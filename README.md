# q4complete

Fully local code completion trained on your codebase. Statistical model,
no cloud, no GPU, no network calls, no LLM. Sub-millisecond suggestions.

It works the way pre-2020 TabNine did: an n-gram model over your repos
plus verbatim line retrieval, plus a per-file cache so it learns as you
type. Trained once over the Quad4 corpus it suggests your real internal
APIs, not generic ones.

## Install

```sh
go build -o bin/q4complete ./cmd/q4complete
cp bin/q4complete ~/.local/bin/
```

## Train

Point it at directories, or mirror the whole org first:

```sh
q4complete collect --org Quad4-Software --dest ~/corpus   # optional
q4complete index --root ~/projects --root ~/corpus
```

The model lands in `~/.local/share/q4complete/model.bin`. Re-run `index`
nightly or after big changes. Indexing a few thousand files takes seconds.

Config is `~/.config/q4complete/config.json`:

```json
{
  "roots": ["/home/you/projects", "/home/you/corpus"],
  "order": 6,
  "maxLines": 3
}
```

## VS Code

```sh
cd vscode
npm install
npm run package
code --install-extension q4complete-0.1.0.vsix
```

The extension spawns `q4complete serve` and wires ghost-text inline
completions. Settings: `q4complete.serverPath`, `q4complete.modelPath`,
`q4complete.maxSuggestions`.

## Other editors

The server speaks LSP 3.18 including `textDocument/inlineCompletion`.
Neovim 0.12+ supports that method natively: register `q4complete serve`
as a stdio language server and ghost text works without a plugin.

## Try it without an editor

```sh
q4complete complete -f somefile.go -line 12 -col 20
q4complete stats
```

## How it works

- `internal/tokenize` language-agnostic lexer
- `internal/model` order-6 n-gram with interpolated backoff and scoped
  caches (Hellendoorn and Devanbu 2017 style locality)
- `internal/lines` sorted unique-line index for verbatim continuations
- `internal/engine` merges the layers, tracks open docs, greedy decode
- `internal/lsp` JSON-RPC stdio server

Prefix completion only: it finishes lines and blocks, it does not fill
the middle of existing code. What it knows is exactly what your corpus
contains, which is the point.

## License

0BSD
