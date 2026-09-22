import { spawn } from "node:child_process";

const srv = spawn("bin/q4complete", ["serve"], { env: { ...process.env, Q4COMPLETE_MODEL: "bin/model.bin" } });
let buf = Buffer.alloc(0);
const pending = new Map();
let id = 0;

srv.stdout.on("data", (d) => {
  buf = Buffer.concat([buf, d]);
  for (;;) {
    const m = buf.toString("latin1").match(/Content-Length: (\d+)\r\n\r\n/);
    if (!m) break;
    const len = +m[1];
    const start = m.index + m[0].length;
    if (buf.length < start + len) break;
    const body = JSON.parse(buf.subarray(start, start + len).toString());
    buf = buf.subarray(start + len);
    if (body.id !== undefined && pending.has(body.id)) {
      pending.get(body.id)(body);
      pending.delete(body.id);
    }
  }
});
srv.stderr.on("data", (d) => process.stderr.write("[srv] " + d));

function send(method, params, isReq = true) {
  const msg = { jsonrpc: "2.0", method, params };
  if (isReq) msg.id = ++id;
  const body = JSON.stringify(msg);
  srv.stdin.write(`Content-Length: ${body.length}\r\n\r\n${body}`);
  if (!isReq) return Promise.resolve(null);
  return new Promise((res) => pending.set(id, res));
}

const doc = "package main\n\nfunc check() error {\n\tif err := doThing(); e";

await send("initialize", { processId: process.pid, capabilities: {}, rootUri: null });
await send("initialized", {}, false);
await send("textDocument/didOpen", { textDocument: { uri: "file:///t.go", languageId: "go", version: 1, text: doc } }, false);
await new Promise((r) => setTimeout(r, 50));
const res = await send("q4/inlineCompletion", {
  textDocument: { uri: "file:///t.go" },
  position: { line: 4, character: 23 },
});
console.log("RESULT", JSON.stringify(res.result));
const res2 = await send("q4/status", {});
console.log("STATUS", JSON.stringify(res2.result));
await send("shutdown", {});
await send("exit", {}, false);
srv.kill();
process.exit(0);
