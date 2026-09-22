package lsp

// Minimal Model Context Protocol server. Two transports:
//
//   - ServeMCP: stdio, newline-delimited JSON-RPC (MCP stdio framing is
//     one message per line, NOT the Content-Length LSP framing)
//   - ServeHTTP: POST endpoint per the Streamable HTTP shape (single
//     JSON-RPC message in, application/json response out)
//
// Both lifecycle styles are tolerated: the legacy initialize handshake
// (<= 2025-11-25) and stateless requests (2026-07-28), so any client
// that can call tools/call works.
//
// The tool surface turns q4complete into a retrieval backend for coding
// agents: ground an LLM in the corpus's actual idioms instead of letting
// it guess APIs.

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"q4complete/internal/engine"
	"q4complete/internal/tokenize"
)

// MCP protocol versions this server speaks. Newest first.
var mcpVersions = []string{"2025-06-18", "2025-03-26", "2024-11-05"}

var mcpServerInfo = map[string]any{"name": "q4complete", "version": "0.1.0"}

// MCPServer exposes engine features as MCP tools.
type MCPServer struct {
	eng     *engine.Engine
	allowFS bool     // allow tools to read files by path (stdio/local)
	roots   []string // when allowFS, paths must live under one of these
}

func NewMCPServer(eng *engine.Engine) *MCPServer {
	return &MCPServer{eng: eng}
}

// ServeMCP runs the stdio transport: newline-delimited JSON-RPC on
// r/w. Run it as `q4complete mcp` and point any MCP client at the binary.
// Stdio is a local trust boundary, so file-path reads are allowed.
func (m *MCPServer) ServeMCP(r io.Reader, w io.Writer) error {
	m.allowFS = true
	br := bufio.NewReaderSize(r, 1<<16)
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 {
			m.handleFrame("", line, enc, bw)
		}
		if err != nil {
			return err
		}
	}
}

// ServeHTTP handles one JSON-RPC message per POST and answers with
// application/json (valid for every MCP request type we support).
func (m *MCPServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.serveHTTPAs("", w, r)
}

func (m *MCPServer) serveHTTPAs(user string, w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<22))
	if err != nil {
		http.Error(w, "read body", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	m.handleFrame(user, body, json.NewEncoder(w), nil)
}

// handleFrame dispatches one raw JSON-RPC message and writes the
// response for requests (notifications get none).
func (m *MCPServer) handleFrame(user string, raw []byte, enc *json.Encoder, bw *bufio.Writer) {
	var msg Message
	if err := json.Unmarshal(raw, &msg); err != nil {
		enc.Encode(map[string]any{
			"jsonrpc": "2.0", "id": nil,
			"error": map[string]any{"code": -32700, "message": "parse error"},
		})
		if bw != nil {
			bw.Flush()
		}
		return
	}
	if len(msg.ID) == 0 {
		return // notification: nothing to write
	}
	res, rerr := m.dispatchAs(user, &msg)
	resp := map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(msg.ID)}
	if rerr != nil {
		resp["error"] = map[string]any{"code": rerr.code, "message": rerr.msg}
	} else {
		resp["result"] = res
	}
	enc.Encode(resp)
	if bw != nil {
		bw.Flush()
	}
}

func (m *MCPServer) dispatch(msg *Message) (any, *rpcError) {
	return m.dispatchAs("", msg)
}

// dispatchAs is dispatch plus a tenant id for per-user learning.
func (m *MCPServer) dispatchAs(user string, msg *Message) (any, *rpcError) {
	switch msg.Method {
	case "initialize":
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		json.Unmarshal(msg.Params, &p)
		ver := mcpVersions[0]
		for _, v := range mcpVersions {
			if p.ProtocolVersion == v {
				ver = v
				break
			}
		}
		return map[string]any{
			"protocolVersion": ver,
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      mcpServerInfo,
		}, nil
	case "server/discover":
		return map[string]any{
			"protocolVersions": mcpVersions,
			"capabilities":     map[string]any{"tools": map[string]any{}},
			"serverInfo":       mcpServerInfo,
		}, nil
	case "ping":
		return map[string]any{}, nil
	case "tools/list":
		return map[string]any{"tools": mcpTools}, nil
	case "tools/call":
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(msg.Params, &p); err != nil {
			return nil, &rpcError{-32602, err.Error()}
		}
		return m.callToolAs(user, p.Name, p.Arguments), nil
	default:
		if strings.HasPrefix(msg.Method, "notifications/") || strings.HasPrefix(msg.Method, "$/") {
			return nil, nil
		}
		return nil, &rpcError{-32601, "method not found: " + msg.Method}
	}
}

// Tool result per spec: content blocks plus structuredContent for
// clients that parse it.
func toolResult(v any) map[string]any {
	data, _ := json.Marshal(v)
	return map[string]any{
		"content":           []any{map[string]any{"type": "text", "text": string(data)}},
		"structuredContent": v,
	}
}

func toolError(msg string) map[string]any {
	return map[string]any{
		"isError": true,
		"content": []any{map[string]any{"type": "text", "text": msg}},
	}
}

// readFile reads a tool-requested path when filesystem access is
// allowed: stdio servers are local so any path goes. HTTP servers only
// permit files under the indexed corpus roots, so a public endpoint
// cannot be used to read arbitrary host files.
func (m *MCPServer) readFile(path string) (string, error) {
	if !m.allowFS {
		return "", errors.New("path access disabled on this transport")
	}
	clean := filepath.Clean(path)
	if len(m.roots) > 0 {
		ok := false
		for _, r := range m.roots {
			if clean == r || strings.HasPrefix(clean, r+string(filepath.Separator)) {
				ok = true
				break
			}
		}
		if !ok {
			return "", errors.New("path outside indexed roots")
		}
	}
	data, err := os.ReadFile(clean)
	if err != nil {
		return "", err
	}
	if len(data) > 4<<20 {
		return "", errors.New("file too large")
	}
	return string(data), nil
}

func (m *MCPServer) callTool(name string, args json.RawMessage) any {
	return m.callToolAs("", name, args)
}

func (m *MCPServer) callToolAs(user, name string, args json.RawMessage) any {
	switch name {
	case "complete":
		var a struct {
			Text      string `json:"text"`
			Path      string `json:"path"`
			URI       string `json:"uri"`
			Offset    int    `json:"offset"`
			Line      int    `json:"line"`
			Character int    `json:"character"`
		}
		json.Unmarshal(args, &a)
		text := a.Text
		uri := a.URI
		if a.Path != "" {
			uri = "file://" + a.Path
		}
		if text == "" && a.Path != "" {
			data, err := m.readFile(a.Path)
			if err != nil {
				return toolError("read: " + err.Error())
			}
			text = data
		}
		if text == "" {
			return toolError("provide text or path")
		}
		if uri == "" {
			uri = "mcp://snippet"
		}
		off := a.Offset
		if off <= 0 && a.Line >= 0 && a.Character >= 0 && (a.Line > 0 || a.Character > 0) {
			off = offsetAt(text, Position{Line: a.Line, Character: a.Character})
		}
		if off <= 0 || off > len(text) {
			off = len(text)
		}
		m.eng.UpdateDoc(uri, text) // doc cache benefits follow-up calls
		items := m.eng.CompleteFor(user, uri, text, off)
		type outItem struct {
			Text         string `json:"insertText"`
			Source       string `json:"source"`
			ReplaceToEOL bool   `json:"replaceToEOL,omitempty"`
		}
		out := make([]outItem, 0, len(items))
		for _, it := range items {
			out = append(out, outItem{it.Text, it.Source, it.ReplaceToEOL})
		}
		return toolResult(map[string]any{"items": out})

	case "lookup_lines":
		var a struct {
			Prefix string `json:"prefix"`
			Limit  int    `json:"limit"`
		}
		json.Unmarshal(args, &a)
		if a.Prefix == "" {
			return toolError("provide prefix")
		}
		if a.Limit <= 0 || a.Limit > 50 {
			a.Limit = 10
		}
		return toolResult(map[string]any{
			"lines": m.eng.LookupLines(a.Prefix, a.Limit),
		})

	case "lookup_symbol":
		var a struct {
			Name  string `json:"name"`
			Limit int    `json:"limit"`
		}
		json.Unmarshal(args, &a)
		if a.Name == "" {
			return toolError("provide name")
		}
		if a.Limit <= 0 || a.Limit > 50 {
			a.Limit = 10
		}
		return toolResult(map[string]any{
			"symbols": m.eng.LookupSymbol(a.Name, a.Limit),
		})

	case "context":
		// Repo-scoped grounding: identifiers near the cursor resolve to
		// their definitions and real usage lines from the corpus, plus
		// what the engine itself would complete at this point. This is
		// what an LLM needs to match house style instead of guessing.
		var a struct {
			Text   string `json:"text"`
			Path   string `json:"path"`
			URI    string `json:"uri"`
			Offset int    `json:"offset"`
		}
		json.Unmarshal(args, &a)
		text := a.Text
		if text == "" && a.Path != "" {
			data, err := m.readFile(a.Path)
			if err != nil {
				return toolError("read: " + err.Error())
			}
			text = data
		}
		if text == "" {
			return toolError("provide text or path")
		}
		off := a.Offset
		if off <= 0 || off > len(text) {
			off = len(text)
		}
		return toolResult(m.context(user, text, off))

	case "learn":
		var a struct {
			Text string `json:"text"`
			URI  string `json:"uri"`
			Line int    `json:"line"`
		}
		json.Unmarshal(args, &a)
		if a.Text == "" {
			return toolError("provide text")
		}
		if len(a.Text) > 4096 {
			return toolError("text too large")
		}
		m.eng.LearnFor(user, a.URI, a.Text, a.Line)
		return toolResult(map[string]any{"ok": true})

	case "status":
		return toolResult(m.eng.Stats())

	default:
		return toolError("unknown tool: " + name)
	}
}

// context assembles grounding for an LLM around a cursor position:
// the identifiers used nearby, their definitions, real usage lines,
// and what the engine would complete here.
func (m *MCPServer) context(user, text string, off int) map[string]any {
	prefix := text[:off]
	// Use the last ~40 lines of context.
	start := off
	for i, n := off-1, 0; i >= 0 && n < 40; i-- {
		if text[i] == '\n' {
			n++
			start = i
		}
	}
	idents := significantIdents(prefix[start:])
	out := map[string]any{"identifiers": idents}
	var defs, usage []any
	seenU := map[string]bool{}
	for _, id := range idents {
		for _, s := range m.eng.LookupSymbol(id, 2) {
			defs = append(defs, s)
		}
		for _, l := range m.eng.LookupLines(id, 3) {
			if !seenU[l] {
				seenU[l] = true
				usage = append(usage, l)
			}
		}
	}
	if len(usage) > 24 {
		usage = usage[:24]
	}
	out["defs"] = defs
	out["usage"] = usage
	var sug []string
	for _, it := range m.eng.CompleteFor(user, "mcp://context", text, off) {
		sug = append(sug, it.Text)
	}
	out["suggestion"] = sug
	return out
}

func isIdentChar(c byte) bool {
	return c == '_' || c == '$' ||
		(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80
}

// significantIdents pulls identifier-shaped tokens from text, dropping
// keywords and very common words, longest first (rare names carry the
// most signal).
func significantIdents(text string) []string {
	seen := map[string]bool{}
	var out []string
	for _, t := range tokenize.Lex([]byte(text)) {
		if len(t) < 4 || !isIdentChar(t[0]) {
			continue
		}
		switch t {
		case "func", "return", "true", "false", "nil", "null", "None",
			"this", "self", "import", "package", "const", "else",
			"with", "from", "then", "when", "case", "break", "continue",
			"string", "error", "int", "bool", "type", "struct", "range",
			"defer", "void", "var", "let":
			continue
		}
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	// Prefer longer (rarer) identifiers, cap at 8.
	for i := 0; i < len(out)-1; i++ {
		for j := i + 1; j < len(out); j++ {
			if len(out[j]) > len(out[i]) {
				out[i], out[j] = out[j], out[i]
			}
		}
	}
	if len(out) > 8 {
		out = out[:8]
	}
	return out
}

var mcpTools = []map[string]any{
	{
		"name": "complete",
		"description": "Return code completions for a position in a file or " +
			"snippet, learned from the indexed corpus. Pass text (full file " +
			"content) plus offset, or path plus line/character. The result is " +
			"grounded in real code: verbatim line continuations and n-gram " +
			"model output, not a language model guess.",
		"inputSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text":      map[string]any{"type": "string", "description": "file content (or the snippet to complete)"},
				"path":      map[string]any{"type": "string", "description": "absolute file path to read and complete in"},
				"uri":       map[string]any{"type": "string", "description": "document uri for cache correlation"},
				"offset":    map[string]any{"type": "integer", "description": "byte offset of the cursor in text"},
				"line":      map[string]any{"type": "integer", "description": "0-based line of the cursor"},
				"character": map[string]any{"type": "integer", "description": "0-based UTF-16 column of the cursor"},
			},
		},
	},
	{
		"name": "lookup_lines",
		"description": "Look up real lines from the indexed corpus by prefix. " +
			"Use it to answer 'how does this codebase do X' with verbatim " +
			"examples instead of inventing API shapes.",
		"inputSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"prefix": map[string]any{"type": "string", "description": "line prefix to match"},
				"limit":  map[string]any{"type": "integer", "description": "max lines, default 10"},
			},
			"required": []string{"prefix"},
		},
	},
	{
		"name": "lookup_symbol",
		"description": "Find where a function, method, type, or variable " +
			"is defined in the indexed corpus. Exact name first, then prefix " +
			"matches. Returns kind, path, line, and the signature line.",
		"inputSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":  map[string]any{"type": "string", "description": "identifier or prefix"},
				"limit": map[string]any{"type": "integer", "description": "max results, default 10"},
			},
			"required": []string{"name"},
		},
	},
	{
		"name": "context",
		"description": "Repo-scoped grounding for a cursor position: " +
			"identifiers used nearby resolved to their definitions, real " +
			"usage lines from the corpus, and what the engine would " +
			"complete here. Use before writing code so edits match " +
			"existing conventions.",
		"inputSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text":   map[string]any{"type": "string", "description": "file content"},
				"path":   map[string]any{"type": "string", "description": "absolute file path to read"},
				"offset": map[string]any{"type": "integer", "description": "byte offset of the cursor (default: end)"},
			},
		},
	},
	{
		"name": "learn",
		"description": "Record an accepted or corrected line so future " +
			"completions rank it higher. Feed accepted suggestions and " +
			"reviewer fixes here.",
		"inputSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text": map[string]any{"type": "string", "description": "the completed line to learn"},
				"uri":  map[string]any{"type": "string", "description": "document uri for accept correlation"},
				"line": map[string]any{"type": "integer", "description": "0-based line where it was shown"},
			},
			"required": []string{"text"},
		},
	},
	{
		"name":        "status",
		"description": "Engine stats: corpus size, vocab, cache fills, accept/reject counters, memory.",
		"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}},
	},
}
