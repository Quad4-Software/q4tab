# q4complete

Fully local code completion trained on your codebase. Statistical model,
no cloud, no GPU, no network calls, no LLM. Single-digit-millisecond
suggestions, single and multi-line.

It works the way pre-2020 TabNine did: an n-gram model over your repos
plus verbatim line retrieval, plus per-file and session caches so it
learns as you type. Trained over the Quad4 corpus it suggests your real
internal APIs, not generic ones.

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

The model lands in `~/.local/share/q4complete/model.bin` plus a
`model.bin.manifest` used for incremental updates. Re-run `index`
nightly or after big changes. For scale reference: 53k files / 79M
tokens of the org's code indexes in about 4 minutes on a desktop CPU and
produces an ~680MB packed model (format v2, varint-compressed). The file
is mmap'd read-only, so it loads in well under a second and only ~100MB
stays resident. The rest faults in on demand and can be reclaimed by the
OS under pressure.

### Incremental index

```sh
q4complete index -incr --root ~/projects --root ~/corpus
```

Diffs the tree against the manifest: unchanged files are skipped,
new/changed files are folded into `model.bin.delta` (a compact overlay
the server merges at startup), deleted files drop out of the delta. Run
it freely. When the delta grows large a full `index` rebuild folds
everything back into the base. Lines from deleted files linger in the
base model until that rebuild.

## QA

```sh
q4complete eval -root ~/projects/somerepo -files 50 -pos 10
```

`eval` samples random mid-line cursor positions in real files, masks the
tail as if you were typing, and reports hit@1 / hit@k and p50/p95
latency per language. Add `-v` to print misses. On the org corpus it
measures ~75% hit@1, ~90% hit@k, ~1ms p50.

`go test -fuzz=FuzzLex ./internal/tokenize/` fuzzes the lexer.

Config is `~/.config/q4complete/config.json`:

```json
{
  "roots": ["/home/you/projects", "/home/you/corpus"],
  "order": 6,
  "maxLines": 4
}
```

## VS Code

```sh
cd vscode
npm install
npm run package
code --install-extension q4complete-0.1.0.vsix
```

The extension spawns `q4complete serve` (resolved in order: the
configured `q4complete.serverPath`, the binary bundled in the vsix,
then PATH) and provides two completion surfaces: ghost-text inline
suggestions that appear as you type, and the classic dropdown via
Ctrl+Space. Tab accepts either one. Esc dismisses. Ctrl+Right accepts
one word at a time.

Status bar shows model size when running, an error icon when the
server fails (click for the log). Commands on the palette:

- `q4complete: Trigger Suggestion` (Alt+\\) - force a suggestion
- `q4complete: Index Workspace` - folds open folders into the
  incremental delta and hot-reloads it into the running server
- `q4complete: Restart Server`, `q4complete: Toggle Completions`,
  `q4complete: Show Status`, `q4complete: Show Log`

Settings: `q4complete.serverPath`, `q4complete.modelPath`,
`q4complete.enabled`, `q4complete.maxSuggestions`,
`q4complete.requestTimeout`.

## Neovim

Requires Neovim 0.12+, which has native `vim.lsp.inline_completion`.

Put `nvim/` on your runtimepath (plugin manager of choice, or symlink
`nvim/` into `~/.config/nvim/pack/*/start/`). Then:

```lua
require("q4complete").setup()
```

Ghost text appears as you type. `Tab` accepts (falls through to a
normal Tab when nothing is shown), `Alt-]` / `Alt-[` cycle candidates.
Accepting a suggestion sends the attached learn command back to the
server automatically. `:lua require("q4complete").toggle()` flips
suggestions on and off. `setup({ completion = true })` additionally
enables the builtin popup completion against the same server.

No plugin? This minimal config still works on 0.12+:

```lua
vim.lsp.config('q4complete', { cmd = { 'q4complete', 'serve' } })
vim.lsp.enable('q4complete')
vim.lsp.inline_completion.enable()
```

## Server deployment

Completions are one small request per keystroke, not a token stream,
so the transport is plain request/response. No WebSockets, no SSE, no
WebTransport: a persistent TCP socket with the normal LSP framing is
the lowest-latency option, and HTTP covers everything else.

```sh
# editor-facing: LSP over TCP (same framing as stdio)
q4complete serve -listen 127.0.0.1:7917

# operational/agent-facing: HTTP
q4complete serve -http 127.0.0.1:7918
#   POST /rpc      JSON-RPC, same methods as the LSP transport
#   POST /mcp      MCP endpoint (see below)
#   GET  /status   engine stats as JSON
#   GET  /healthz  liveness
```

Both flags can be combined on one process. Each TCP connection gets an
isolated document session. `/rpc` shares one session so didOpen and
didChange state persists across requests.

VS Code remote mode:

```json
{ "q4complete.serverAddr": "10.0.0.5:7917" }
```

Neovim remote mode:

```lua
require("q4complete").setup({ addr = "10.0.0.5:7917" })
```

Warning: the model contains verbatim source lines. Bind to loopback or
put the listener behind a VPN/TLS terminator before exposing it.

## MCP (LLM / agent integration)

q4complete doubles as an MCP server so coding agents can ground
themselves in the corpus instead of guessing APIs.

```sh
q4complete mcp   # stdio transport, one JSON-RPC message per line
```

Point any MCP client at that command (Claude Code: `claude mcp add
q4complete -- q4complete mcp`), or POST to `/mcp` on the HTTP listener
for remote agents. Speaks MCP `2025-06-18`, `2025-03-26`, and
`2024-11-05`, plus `server/discover` for newer clients.

Tools:

| tool | args | returns |
|---|---|---|
| `complete` | `text`+`offset` or `path`+`line`/`character` | ranked completion items |
| `lookup_lines` | `prefix`, `limit` | verbatim corpus lines |
| `learn` | `text`, `uri`, `line` | feeds the accept-learning loop |
| `status` | none | corpus size, counters, memory |

The division of labor: q4complete is the fast deterministic path in the
editor, the LLM uses `lookup_lines`/`complete` to fetch real idioms for
explanation, refactoring, and code review, and `learn` feeds accepted
results back into the ranking.

## Other editors

The server speaks LSP 3.18 including `textDocument/inlineCompletion`
(UTF-16 positions, per spec). Any client that implements that method
works with `q4complete serve` over stdio.

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
- `internal/engine` merges the layers, tracks open docs, greedy
  multi-line decode with bracket-depth and indentation awareness
- `internal/lsp` JSON-RPC stdio server

Ranking: same-file repetition first, then lines from other open files
and accepted completions, then corpus verbatim matches, then n-gram
generation.

### Multi-line completion

When the cursor is at end of line the engine may emit a whole block
(e.g. `if err !=` -> the full `nil { return err }` body plus the
closing brace). Generation stops at the configured line cap, at a
closing dedent below the cursor's bracket depth, when it would
reproduce code already after the cursor, or when confidence drops below
the multi-line floor. Generated indentation is rewritten into the
document's own indent unit (tabs vs spaces).

### Learning from accepts

Accepted completions are sent back via `q4/learn` (the extension wires
this to the accept event automatically. The server also accepts
`workspace/executeCommand` with `q4complete.learn`). Accepted text is
appended to `~/.local/share/q4complete/learned.jsonl`, replayed on
startup, mixed into the learned n-gram cache, and indexed into the
dynamic line index so the same context retrieves it next time. The
journal self-compacts at 4MB. Set `Q4COMPLETE_JOURNAL` to relocate it.

Shown-but-not-accepted suggestions are tracked too: every displayed item
is recorded per (document, line), an accept marks its match and counts
the rest as implicit rejects, and clients can send `q4/reject`
explicitly. Per-source accept rates drive a bounded adaptive floor on
the model's probability threshold, and repeat accepts boost the same
completed line's rank next time. Counters are visible in `q4/status`.

### Fill-in-the-middle (light)

When the cursor sits mid-line, a suggestion that already ends with the
existing tail becomes a zero-width insert of only the missing middle
(`f(a, |b)` gets `x, ` before `b)`, not a clobbered line). Shorter
suggestions are allowed to insert when the spliced line is attested in
the static or dynamic index. Everything else keeps a replace-to-EOL
range. No neural FIM model, only verification.

### Directory scope

Files in the same directory share a per-directory cache, so idioms from
sibling files surface even before the session cache warms up. Capped at
64 directories.

### Edge cases handled

- UTF-16 position mapping incl. astral characters and CRLF files
- Incremental `didChange` sync (ranged edits, not full-document resends)
- Suffix-aware dedupe: never suggests text already after the cursor
- Divergent mid-line suggestions carry a replace-to-EOL range instead
  of splicing into existing text
- Documents larger than 128KB index in a background worker. Keystrokes
  on huge files never block the completion path
- Corrupt or truncated model files fail with an error, not a panic,
  and a panic anywhere in Complete degrades to no suggestions rather
  than killing the server
- Malformed/oversized JSON-RPC frames are skipped, not fatal
- Journal entries written before a model loads are queued and replayed

### Honest limits

It finishes lines and blocks and can fill a verified middle, but it
cannot invent APIs it has never seen. What it knows is exactly what your
corpus plus your open files contain, which is the point.

## License

0BSD
