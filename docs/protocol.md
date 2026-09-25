# Wire surface

One server, three transports. All of them carry the same JSON-RPC
methods.

## LSP over stdio

`q4tab serve` with no flags. This is what editors spawn.

## LSP over TCP

`q4tab serve -listen 127.0.0.1:7917`. Same framing as stdio. Each
connection gets an isolated document session, so didOpen/didChange on
one socket cannot leak into another.

## HTTP

`q4tab serve -http 127.0.0.1:7918`:

| route | method | purpose |
|---|---|---|
| `/rpc` | POST | JSON-RPC, same methods as LSP; one shared session |
| `/mcp` | POST | MCP endpoint |
| `/status` | GET | engine stats as JSON |
| `/healthz` | GET | liveness |

## Methods

`textDocument/inlineCompletion` (LSP 3.18, UTF-16 positions) is the
editor-facing path. `q4/learn` and `q4/reject` feed the accept loop.
`q4/status` exposes counters. `workspace/executeCommand` accepts
`q4tab.learn`. `q4/nextEdit` returns predicted next edit sites -
lines still carrying the old side of a recent rename - for the
VS Code q4tab.nextEdit jump command (alt+]).

## MCP

`q4tab mcp` speaks MCP over NDJSON stdio, one message per line.
Protocol versions `2025-06-18`, `2025-03-26`, `2024-11-05`, plus
`server/discover`. Tools: `complete`, `lookup_lines`, `learn`,
`status`.

## Auth and limits

`Q4TAB_TOKEN` sets a bearer token checked on `X-Q4-Token`.
`X-Q4-User` tags the per-user journal. Without a token the server
trusts the connection: bind to loopback or put it behind a
terminator, because the model contains verbatim source lines.
