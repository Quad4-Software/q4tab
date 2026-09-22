package lsp

import "unicode/utf8"

// Minimal LSP type subset (LSP 3.18).

type Position struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

type Range struct {
	Start Position `json:"start"`
	End   Position `json:"end"`
}

type TextDocumentIdentifier struct {
	URI string `json:"uri"`
}

type TextDocumentItem struct {
	URI        string `json:"uri"`
	LanguageID string `json:"languageId"`
	Version    int    `json:"version"`
	Text       string `json:"text"`
}

type VersionedTextDocumentIdentifier struct {
	URI     string `json:"uri"`
	Version int    `json:"version"`
}

type DidOpenTextDocumentParams struct {
	TextDocument TextDocumentItem `json:"textDocument"`
}

type TextDocumentContentChangeEvent struct {
	Range *Range `json:"range,omitempty"`
	Text  string `json:"text"`
}

type DidChangeTextDocumentParams struct {
	TextDocument   VersionedTextDocumentIdentifier  `json:"textDocument"`
	ContentChanges []TextDocumentContentChangeEvent `json:"contentChanges"`
}

type DidCloseTextDocumentParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

type DidSaveTextDocumentParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Text         string                 `json:"text,omitempty"`
}

type InlineCompletionContext struct {
	TriggerKind int `json:"triggerKind"`
}

type InlineCompletionParams struct {
	Context      InlineCompletionContext `json:"context"`
	TextDocument TextDocumentIdentifier  `json:"textDocument"`
	Position     Position                `json:"position"`
	Range        *Range                  `json:"range,omitempty"`
}

type InlineCompletionItem struct {
	InsertText string   `json:"insertText"`
	Range      *Range   `json:"range,omitempty"`
	FilterText string   `json:"filterText,omitempty"`
	Command    *Command `json:"command,omitempty"`
}

type InlineCompletionList struct {
	Items []InlineCompletionItem `json:"items"`
}

type CompletionParams struct {
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	Position     Position               `json:"position"`
}

type CompletionItem struct {
	Label      string `json:"label"`
	InsertText string `json:"insertText,omitempty"`
	Kind       int    `json:"kind,omitempty"`
	Detail     string `json:"detail,omitempty"`
}

type CompletionList struct {
	IsIncomplete bool             `json:"isIncomplete"`
	Items        []CompletionItem `json:"items"`
}

// Command is the LSP command attached to an accepted completion.
type Command struct {
	Title     string `json:"title"`
	Command   string `json:"command"`
	Arguments []any  `json:"arguments,omitempty"`
}

// offsetAt converts an LSP position to a byte offset in text. LSP
// positions are UTF-16 code units: BMP runes count as one unit, others
// as two. We walk runes and accumulate units, so non-ASCII columns land
// correctly.
func offsetAt(text string, p Position) int {
	line := 0
	i := 0
	for i < len(text) && line < p.Line {
		if text[i] == '\n' {
			line++
		}
		i++
	}
	col := 0
	for i < len(text) && text[i] != '\n' {
		if col >= p.Character {
			break
		}
		// CRLF: the carriage return is not part of the line's content,
		// so a position past end-of-line clamps before it.
		if text[i] == '\r' && i+1 < len(text) && text[i+1] == '\n' {
			break
		}
		// UTF-16 columns: astral runes count as two code units.
		r, w := utf8.DecodeRuneInString(text[i:])
		i += w
		if r > 0xFFFF {
			col += 2
		} else {
			col++
		}
	}
	return i
}
