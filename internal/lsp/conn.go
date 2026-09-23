// Package lsp implements a minimal Language Server Protocol endpoint over
// stdio: Content-Length framed JSON-RPC, document sync, and the
// textDocument/inlineCompletion method from LSP 3.18, plus the classic
// textDocument/completion for clients without inline support.
package lsp

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
)

// Conn is a JSON-RPC connection over LSP base protocol framing.
type Conn struct {
	r  *bufio.Reader
	w  io.Writer
	wm sync.Mutex
}

func NewConn(r io.Reader, w io.Writer) *Conn {
	return &Conn{r: bufio.NewReaderSize(r, 1<<16), w: w}
}

// Message is a raw JSON-RPC request or notification.
type Message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

func (c *Conn) Read() (*Message, error) {
	length := -1
	for {
		line, err := c.r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if strings.HasPrefix(line, "Content-Length:") {
			v := strings.TrimSpace(line[len("Content-Length:"):])
			length, err = strconv.Atoi(v)
			if err != nil {
				return nil, fmt.Errorf("bad Content-Length: %w", err)
			}
		}
	}
	if length < 0 {
		return nil, fmt.Errorf("missing Content-Length")
	}
	if length > 1<<26 {
		// Oversized but length-delimited: discard the body so the
		// stream stays in sync, then let the caller skip it.
		if _, err := io.CopyN(io.Discard, c.r, int64(length)); err != nil {
			return nil, err
		}
		return nil, ErrOversized
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(c.r, buf); err != nil {
		return nil, err
	}
	var m Message
	if err := json.Unmarshal(buf, &m); err != nil {
		// The frame was consumed. A bad body is skippable.
		return nil, fmt.Errorf("%w: %v", ErrBadFrame, err)
	}
	return &m, nil
}

// ErrOversized marks a frame that was discarded for exceeding the size
// cap. The stream remains synchronized.
var ErrOversized = errors.New("frame exceeds size cap")

// ErrBadFrame marks a consumed frame whose body did not parse.
var ErrBadFrame = errors.New("bad frame body")

func (c *Conn) write(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	c.wm.Lock()
	defer c.wm.Unlock()
	if _, err := fmt.Fprintf(c.w, "Content-Length: %d\r\n\r\n", len(data)); err != nil {
		return err
	}
	_, err = c.w.Write(data)
	return err
}

// Respond sends a result for a request id.
func (c *Conn) Respond(id json.RawMessage, result any) error {
	return c.write(struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Result  any             `json:"result"`
	}{"2.0", id, result})
}

// RespondError sends an error for a request id.
func (c *Conn) RespondError(id json.RawMessage, code int, msg string) error {
	return c.write(struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Error   any             `json:"error"`
	}{"2.0", id, struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	}{code, msg}})
}
