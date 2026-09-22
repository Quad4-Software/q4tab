// Package lsp implements a minimal Language Server Protocol endpoint over
// stdio: Content-Length framed JSON-RPC, document sync, and the
// textDocument/inlineCompletion method from LSP 3.18, plus the classic
// textDocument/completion for clients without inline support.
package lsp

import (
	"bufio"
	"encoding/json"
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
	if length < 0 || length > 1<<26 {
		return nil, fmt.Errorf("bad frame length %d", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(c.r, buf); err != nil {
		return nil, err
	}
	var m Message
	if err := json.Unmarshal(buf, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

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

// Notify sends a server-to-client notification.
func (c *Conn) Notify(method string, params any) error {
	return c.write(struct {
		JSONRPC string `json:"jsonrpc"`
		Method  string `json:"method"`
		Params  any    `json:"params,omitempty"`
	}{"2.0", method, params})
}
