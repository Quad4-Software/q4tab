package lsp

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"sync"

	"q4complete/internal/engine"
)

// Server routes LSP traffic to the completion engine.
type Server struct {
	eng       *engine.Engine
	conn      *Conn
	docsMu    sync.Mutex // guards docs. HTTP handlers run concurrently
	docs      map[string]string
	shutdown  bool
	deltaPath string // model.bin.delta for q4/reindex
	user      string // tenant id (TCP: set by q4/auth. HTTP: per-request)
	reqToken  string // when set, TCP clients must q4/auth before requests
}

func NewServer(eng *engine.Engine, conn *Conn) *Server {
	return &Server{eng: eng, conn: conn, docs: make(map[string]string)}
}

// SetDeltaPath records where the incremental-index delta lives so
// q4/reindex can fold fresh corpus changes into a running server.
func (s *Server) SetDeltaPath(p string) { s.deltaPath = p }

// capabilities advertised in initialize.
var serverCaps = json.RawMessage(`{
	"capabilities": {
		"positionEncoding": "utf-16",
		"textDocumentSync": {"openClose": true, "change": 2},
		"inlineCompletionProvider": true,
		"completionProvider": {"triggerCharacters": [".", " ", "(", ">", ":"]},
		"executeCommandProvider": {"commands": ["q4complete.learn"]}
	}
}`)

// Run processes messages until exit or EOF.
func (s *Server) Run() error {
	for {
		m, err := s.conn.Read()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			// A discarded oversized frame or a single unparseable body
			// must not kill the language server.
			if errors.Is(err, ErrOversized) || errors.Is(err, ErrBadFrame) {
				continue
			}
			return err
		}
		if len(m.ID) == 0 {
			s.notify(m)
			continue
		}
		s.request(m)
	}
}

// rpcError carries a JSON-RPC error back through transports that
// synthesize responses (HTTP) instead of writing to a Conn.
type rpcError struct {
	code int
	msg  string
}

func (s *Server) request(m *Message) {
	res, rerr := s.dispatchAs(s.user, m)
	if rerr != nil {
		s.conn.RespondError(m.ID, rerr.code, rerr.msg)
		return
	}
	s.conn.Respond(m.ID, res)
}

// dispatchAs routes a request and returns its result or error without
// touching the transport, so TCP and HTTP frontends share the logic.
// user selects the tenant overlay (empty = shared/global).
func (s *Server) dispatchAs(user string, m *Message) (any, *rpcError) {
	if s.reqToken != "" && s.user == "" {
		switch m.Method {
		case "initialize", "q4/auth", "q4/status":
		default:
			return nil, &rpcError{-32001, "authentication required: send q4/auth"}
		}
	}
	switch m.Method {
	case "q4/auth":
		// Per-connection auth for the TCP transport. HTTP callers
		// authenticate per request via the Authorization header.
		var p struct {
			Token string `json:"token"`
		}
		if json.Unmarshal(m.Params, &p) != nil ||
			subtle.ConstantTimeCompare([]byte(p.Token), []byte(s.reqToken)) != 1 {
			return nil, &rpcError{-32001, "bad token"}
		}
		sum := sha256.Sum256([]byte(p.Token))
		s.user = "t:" + hex.EncodeToString(sum[:8])
		return map[string]bool{"ok": true}, nil
	case "initialize":
		return serverCaps, nil
	case "shutdown":
		s.shutdown = true
		return nil, nil
	case "textDocument/inlineCompletion", "q4/inlineCompletion":
		var p InlineCompletionParams
		if err := json.Unmarshal(m.Params, &p); err != nil {
			return nil, &rpcError{-32602, err.Error()}
		}
		return s.inlineCompletionAs(user, p), nil
	case "textDocument/completion":
		var p CompletionParams
		if err := json.Unmarshal(m.Params, &p); err != nil {
			return nil, &rpcError{-32602, err.Error()}
		}
		return s.completionAs(user, p), nil
	case "q4/status":
		return s.eng.Stats(), nil
	case "q4/learn":
		var p struct {
			Text string `json:"text"`
			URI  string `json:"uri"`
			Line int    `json:"line"`
		}
		if err := json.Unmarshal(m.Params, &p); err != nil {
			return nil, &rpcError{-32602, err.Error()}
		}
		s.eng.LearnFor(user, p.URI, p.Text, p.Line)
		return nil, nil
	case "q4/reject":
		var p struct {
			URI  string `json:"uri"`
			Line int    `json:"line"`
		}
		if err := json.Unmarshal(m.Params, &p); err != nil {
			return nil, &rpcError{-32602, err.Error()}
		}
		s.eng.Reject(p.URI, p.Line)
		return nil, nil
	case "q4/lookupSymbol":
		var p struct {
			Name  string `json:"name"`
			Limit int    `json:"limit"`
		}
		if json.Unmarshal(m.Params, &p) != nil || p.Name == "" {
			return nil, &rpcError{-32602, "invalid params"}
		}
		if p.Limit <= 0 || p.Limit > 50 {
			p.Limit = 10
		}
		return s.eng.LookupSymbol(p.Name, p.Limit), nil
	case "q4/reindex":
		if s.deltaPath == "" {
			return nil, &rpcError{-32602, "no delta path configured"}
		}
		files, err := engine.LoadDelta(s.deltaPath)
		if errors.Is(err, os.ErrNotExist) {
			files = nil // no delta on disk: clear any prior overlay
		} else if err != nil {
			return nil, &rpcError{-32602, err.Error()}
		}
		s.eng.InstallDelta(files)
		return map[string]int{"files": len(files)}, nil
	case "workspace/executeCommand":
		var p struct {
			Command   string            `json:"command"`
			Arguments []json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(m.Params, &p); err == nil && p.Command == "q4complete.learn" && len(p.Arguments) > 0 {
			var text string
			var uri string
			var line int
			json.Unmarshal(p.Arguments[0], &text)
			if len(p.Arguments) > 1 {
				json.Unmarshal(p.Arguments[1], &uri)
			}
			if len(p.Arguments) > 2 {
				json.Unmarshal(p.Arguments[2], &line)
			}
			if text != "" {
				s.eng.Learn(uri, text, line)
			}
		}
		return nil, nil
	default:
		if strings.HasPrefix(m.Method, "$/") {
			return nil, nil // spec-mandated ignore
		}
		return nil, &rpcError{-32601, "method not found: " + m.Method}
	}
}

func (s *Server) notify(m *Message) {
	switch m.Method {
	case "exit":
		if s.conn == nil {
			return // shared transports (HTTP) ignore exit
		}
		if s.shutdown {
			os.Exit(0)
		}
		os.Exit(1) // spec: exit without shutdown is an error
	case "textDocument/didOpen":
		var p DidOpenTextDocumentParams
		if json.Unmarshal(m.Params, &p) == nil {
			s.docsMu.Lock()
			s.docs[p.TextDocument.URI] = p.TextDocument.Text
			s.docsMu.Unlock()
			s.eng.UpdateDoc(p.TextDocument.URI, p.TextDocument.Text)
		}
	case "textDocument/didChange":
		var p DidChangeTextDocumentParams
		if json.Unmarshal(m.Params, &p) == nil {
			uri := p.TextDocument.URI
			s.docsMu.Lock()
			text, ok := s.docs[uri]
			for _, c := range p.ContentChanges {
				if c.Range == nil {
					text = c.Text // full sync fallback
					ok = true
					continue
				}
				if !ok {
					continue // change for a doc we never opened
				}
				start := offsetAt(text, c.Range.Start)
				end := offsetAt(text, c.Range.End)
				if start > end || start > len(text) || end > len(text) {
					continue
				}
				text = text[:start] + c.Text + text[end:]
			}
			if len(p.ContentChanges) > 0 {
				s.docs[uri] = text
			}
			s.docsMu.Unlock()
			if len(p.ContentChanges) > 0 {
				s.eng.UpdateDoc(uri, text)
			}
		}
	case "textDocument/didClose":
		var p DidCloseTextDocumentParams
		if json.Unmarshal(m.Params, &p) == nil {
			s.docsMu.Lock()
			delete(s.docs, p.TextDocument.URI)
			s.docsMu.Unlock()
			s.eng.CloseDoc(p.TextDocument.URI)
		}
	case "textDocument/didSave":
		var p DidSaveTextDocumentParams
		if json.Unmarshal(m.Params, &p) == nil && p.Text != "" {
			s.docsMu.Lock()
			s.docs[p.TextDocument.URI] = p.Text
			s.docsMu.Unlock()
			s.eng.UpdateDoc(p.TextDocument.URI, p.Text)
		}
	}
}

func (s *Server) docText(uri string) string {
	s.docsMu.Lock()
	defer s.docsMu.Unlock()
	return s.docs[uri]
}

func (s *Server) inlineCompletionAs(user string, p InlineCompletionParams) InlineCompletionList {
	text := s.docText(p.TextDocument.URI)
	// Empty text is still a valid request: thin-context priors
	// (file-start lines, package clauses) answer it.
	off := offsetAt(text, p.Position)
	items := s.eng.CompleteFor(user, p.TextDocument.URI, text, off)
	// The completed-line text is computed server-side so generic clients
	// that fire the attached command still deliver learnable context.
	lineStart := strings.LastIndexByte(text[:off], '\n') + 1
	linePrefix := text[lineStart:off]
	out := make([]InlineCompletionItem, 0, len(items))
	for _, it := range items {
		item := InlineCompletionItem{InsertText: it.Text}
		item.Command = &Command{
			Title:     "accepted",
			Command:   "q4complete.learn",
			Arguments: []any{linePrefix + it.Text, p.TextDocument.URI, p.Position.Line},
		}
		if it.ReplaceToEOL {
			// The suggestion diverges from the existing line tail.
			// the client must replace to end-of-line, not splice.
			eol := strings.IndexByte(text[off:], '\n')
			endOff := len(text)
			if eol >= 0 {
				endOff = off + eol
				if endOff > 0 && text[endOff-1] == '\r' {
					endOff--
				}
			}
			item.Range = &Range{
				Start: p.Position,
				End:   Position{Line: p.Position.Line, Character: p.Position.Character + utf16Len(text[off:endOff])},
			}
		}
		out = append(out, item)
	}
	return InlineCompletionList{Items: out}
}

// utf16Len counts UTF-16 code units in s: BMP runes are one unit,
// astral runes are two.
func utf16Len(s string) int {
	n := 0
	for i := 0; i < len(s); {
		c := s[i]
		w := 1
		switch {
		case c >= 0xf0:
			w = 4
		case c >= 0xe0:
			w = 3
		case c >= 0xc0:
			w = 2
		}
		if i+w > len(s) {
			w = 1
		}
		if w == 4 {
			n += 2
		} else {
			n++
		}
		i += w
	}
	return n
}

func (s *Server) completionAs(user string, p CompletionParams) CompletionList {
	text := s.docText(p.TextDocument.URI)
	// Empty text is still a valid request: thin-context priors
	// (file-start lines, package clauses) answer it.
	off := offsetAt(text, p.Position)
	items := s.eng.CompleteFor(user, p.TextDocument.URI, text, off)
	out := make([]CompletionItem, 0, len(items))
	for _, it := range items {
		label := it.Text
		if len(label) > 60 {
			// Rune-safe truncation for the display label.
			r := []rune(label)
			if len(r) > 60 {
				label = string(r[:60]) + "..."
			}
		}
		out = append(out, CompletionItem{
			Label:      label,
			InsertText: it.Text,
			Kind:       15, // Snippet-ish. Plain text insert
			Detail:     "q4 " + it.Source,
		})
	}
	return CompletionList{IsIncomplete: false, Items: out}
}
