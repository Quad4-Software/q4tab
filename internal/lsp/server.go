package lsp

import (
	"encoding/json"
	"io"
	"os"
	"strings"

	"q4complete/internal/engine"
)

// Server routes LSP traffic to the completion engine.
type Server struct {
	eng  *engine.Engine
	conn *Conn
	docs map[string]string
}

func NewServer(eng *engine.Engine, conn *Conn) *Server {
	return &Server{eng: eng, conn: conn, docs: make(map[string]string)}
}

// capabilities advertised in initialize.
var serverCaps = json.RawMessage(`{
	"capabilities": {
		"textDocumentSync": 1,
		"inlineCompletionProvider": true,
		"completionProvider": {"triggerCharacters": [".", " ", "(", ">", ":"]}
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
			return err
		}
		if len(m.ID) == 0 {
			s.notify(m)
			continue
		}
		s.request(m)
	}
}

func (s *Server) request(m *Message) {
	switch m.Method {
	case "initialize":
		s.conn.Respond(m.ID, serverCaps)
	case "shutdown":
		s.conn.Respond(m.ID, nil)
	case "textDocument/inlineCompletion", "q4/inlineCompletion":
		var p InlineCompletionParams
		if err := json.Unmarshal(m.Params, &p); err != nil {
			s.conn.RespondError(m.ID, -32602, err.Error())
			return
		}
		s.conn.Respond(m.ID, s.inlineCompletion(p))
	case "textDocument/completion":
		var p CompletionParams
		if err := json.Unmarshal(m.Params, &p); err != nil {
			s.conn.RespondError(m.ID, -32602, err.Error())
			return
		}
		s.conn.Respond(m.ID, s.completion(p))
	case "q4/status":
		s.conn.Respond(m.ID, s.eng.Stats())
	default:
		if strings.HasPrefix(m.Method, "$/") {
			return // spec-mandated ignore
		}
		s.conn.RespondError(m.ID, -32601, "method not found: "+m.Method)
	}
}

func (s *Server) notify(m *Message) {
	switch m.Method {
	case "exit":
		os.Exit(0)
	case "textDocument/didOpen":
		var p DidOpenTextDocumentParams
		if json.Unmarshal(m.Params, &p) == nil {
			s.docs[p.TextDocument.URI] = p.TextDocument.Text
			s.eng.UpdateDoc(p.TextDocument.URI, p.TextDocument.Text)
		}
	case "textDocument/didChange":
		var p DidChangeTextDocumentParams
		if json.Unmarshal(m.Params, &p) == nil {
			// Full sync: last change event carries whole document.
			if n := len(p.ContentChanges); n > 0 {
				text := p.ContentChanges[n-1].Text
				s.docs[p.TextDocument.URI] = text
				s.eng.UpdateDoc(p.TextDocument.URI, text)
			}
		}
	case "textDocument/didClose":
		var p DidCloseTextDocumentParams
		if json.Unmarshal(m.Params, &p) == nil {
			delete(s.docs, p.TextDocument.URI)
			s.eng.CloseDoc(p.TextDocument.URI)
		}
	case "textDocument/didSave":
		var p DidSaveTextDocumentParams
		if json.Unmarshal(m.Params, &p) == nil && p.Text != "" {
			s.docs[p.TextDocument.URI] = p.Text
			s.eng.UpdateDoc(p.TextDocument.URI, p.Text)
		}
	}
}

func (s *Server) docText(uri string) string {
	return s.docs[uri]
}

func (s *Server) inlineCompletion(p InlineCompletionParams) InlineCompletionList {
	text := s.docText(p.TextDocument.URI)
	if text == "" {
		return InlineCompletionList{Items: []InlineCompletionItem{}}
	}
	off := offsetAt(text, p.Position)
	items := s.eng.Complete(p.TextDocument.URI, text, off)
	out := make([]InlineCompletionItem, 0, len(items))
	for _, it := range items {
		out = append(out, InlineCompletionItem{InsertText: it.Text})
	}
	return InlineCompletionList{Items: out}
}

func (s *Server) completion(p CompletionParams) CompletionList {
	text := s.docText(p.TextDocument.URI)
	if text == "" {
		return CompletionList{IsIncomplete: false, Items: []CompletionItem{}}
	}
	off := offsetAt(text, p.Position)
	items := s.eng.Complete(p.TextDocument.URI, text, off)
	out := make([]CompletionItem, 0, len(items))
	for _, it := range items {
		label := it.Text
		if len(label) > 60 {
			label = label[:60] + "..."
		}
		out = append(out, CompletionItem{
			Label:      label,
			InsertText: it.Text,
			Kind:       15, // Snippet-ish; plain text insert
			Detail:     "q4 " + it.Source,
		})
	}
	return CompletionList{IsIncomplete: false, Items: out}
}

// LogSend writes a window/logMessage notification.
func (s *Server) LogSend(msg string) {
	s.conn.Notify("window/logMessage", struct {
		Type    int    `json:"type"`
		Message string `json:"message"`
	}{3, msg})
}
