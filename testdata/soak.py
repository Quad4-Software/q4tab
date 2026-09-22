import json, subprocess, os, random

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

e2 = dict(os.environ)
e2.update({"Q4COMPLETE_MODEL": "bin/model.bin", "Q4COMPLETE_JOURNAL": "testdata/soak_journal.jsonl"})
p = subprocess.Popen(["./bin/q4complete", "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=e2)

def rss():
    try:
        with open(f"/proc/{p.pid}/status") as f:
            for l in f:
                if l.startswith("VmRSS"):
                    return int(l.split()[1]) // 1024
    except FileNotFoundError:
        pass
    return -1

p.stdin.write(frame({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}))
p.stdin.flush()
read_msg(p.stdout)
print("RSS after init: %d MB" % rss(), flush=True)

rng = random.Random(7)
docs = []
for i in range(30):
    body = "".join(f"func fn_{i}_{j}() error {{\n\terr := call_{j}(ctx)\n\tif err != nil {{\n\t\treturn err\n\t}}\n}}\n" for j in range(40))
    doc = f"package m{i}\n\n" + body
    docs.append(doc)
    p.stdin.write(frame({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":f"file:///m{i}.go","languageId":"go","version":1,"text":doc}}}))
p.stdin.flush()
print("RSS after 30 opens: %d MB" % rss(), flush=True)

req = 2
lat = []
import time
for i in range(300):
    d = rng.randrange(30)
    doc = docs[d] + f"\n// edit {i}\n" + "x" * rng.randrange(200)
    docs[d] = doc
    p.stdin.write(frame({"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":f"file:///m{d}.go","version":i},"contentChanges":[{"text":doc}]}}))
    p.stdin.write(frame({"jsonrpc":"2.0","id":req,"method":"textDocument/inlineCompletion","params":{"textDocument":{"uri":f"file:///m{d}.go"},"position":{"line":rng.randrange(30),"character":rng.randrange(20)}}}))
    p.stdin.flush()
    # Read each request response so neither pipe fills. Server is
    # sequential, so responses arrive in id order.
    t0 = time.monotonic()
    m = read_msg(p.stdout)
    lat.append(time.monotonic() - t0)
    assert m and m.get("id") == req, f"lost response for id {req}"
    req += 1
    if i % 7 == 0:
        p.stdin.write(frame({"jsonrpc":"2.0","id":req,"method":"q4/learn","params":{"text":"\tif err != nil { return wrap(err) }"}}))
        p.stdin.flush()
        read_msg(p.stdout)
        req += 1
    if i % 75 == 74:
        print("RSS after %d edits+completes: %d MB" % (i+1, rss()), flush=True)

lat.sort()
print("req latency p50=%.1fms p95=%.1fms max=%.1fms" % (lat[len(lat)//2]*1e3, lat[int(len(lat)*.95)]*1e3, lat[-1]*1e3), flush=True)

p.stdin.write(frame({"jsonrpc":"2.0","id":req,"method":"q4/status","params":{}}))
p.stdin.flush()
# drain responses
deadline = 0
while deadline < 400:
    m = read_msg(p.stdout)
    if m is None: break
    if m.get("id") == req:
        print("status:", json.dumps(m["result"])[:300], flush=True)
        break
    deadline += 1
print("final RSS: %d MB" % rss(), flush=True)
p.stdin.write(frame({"jsonrpc":"2.0","id":req+1,"method":"shutdown","params":{}}))
p.stdin.write(frame({"jsonrpc":"2.0","method":"exit","params":{}}))
p.stdin.flush()
p.wait(timeout=30)
print("exit:", p.returncode, flush=True)
