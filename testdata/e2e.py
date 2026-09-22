import json, subprocess, os

def frame(m):
    b = json.dumps(m).encode()
    return b"Content-Length: %d\r\n\r\n" % len(b) + b

def read_msg(f):
    while True:
        line = f.readline()
        if not line: return None
        if line.startswith(b"Content-Length:"):
            length = int(line.split(b":")[1])
        elif line in (b"\r\n", b"\n"):
            break
    return json.loads(f.read(length))

doc = "package x\n\nfunc h() error {\n\terr := work()\n\tif err !="
e2 = dict(os.environ)
e2.update({"Q4COMPLETE_MODEL": "bin/model.bin", "Q4COMPLETE_JOURNAL": "testdata/journal_test.jsonl"})
p = subprocess.Popen(["./bin/q4complete", "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=e2)
p.stdin.write(frame({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}))
p.stdin.write(frame({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///x.go","languageId":"go","version":1,"text":doc}}}))
p.stdin.write(frame({"jsonrpc":"2.0","id":2,"method":"textDocument/inlineCompletion","params":{"textDocument":{"uri":"file:///x.go"},"position":{"line":4,"character":10}}}))
p.stdin.write(frame({"jsonrpc":"2.0","id":3,"method":"q4/learn","params":{"text":"\tif err != nil {","uri":"file:///x.go","line":4}}))
p.stdin.write(frame({"jsonrpc":"2.0","id":4,"method":"textDocument/inlineCompletion","params":{"textDocument":{"uri":"file:///x.go"},"position":{"line":4,"character":10}}}))
p.stdin.write(frame({"jsonrpc":"2.0","id":6,"method":"q4/reject","params":{"uri":"file:///x.go","line":4}}))
p.stdin.write(frame({"jsonrpc":"2.0","id":7,"method":"q4/status","params":{}}))
p.stdin.write(frame({"jsonrpc":"2.0","id":5,"method":"shutdown","params":{}}))
p.stdin.write(frame({"jsonrpc":"2.0","method":"exit","params":{}}))
p.stdin.flush()
while True:
    m = read_msg(p.stdout)
    if m is None: break
    print(json.dumps(m)[:400])
p.wait(timeout=30)
print("server exit:", p.returncode)
