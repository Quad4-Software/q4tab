# Serving and memory profiles

q4tab serve runs the LSP server. Flags:

- `-listen addr` - LSP over TCP
- `-http addr` - HTTP endpoints: /rpc, /mcp, /status, /healthz
- `-token` - bearer auth (or Q4TAB_TOKEN)
- `-rate`, `-burst`, `-maxconc` - abuse controls
- `-mem auto|low|max` - memory profile (or Q4TAB_MEM, or config key mem)

## Memory profiles

The model file is mmap'd read-only, so most of what top shows as RSS
is page cache the kernel can drop under pressure. Three levers decide
how much heap the process keeps anyway:

- Dynamic n-gram caches (session, learned, delta, per-user) store
  context rows as Go maps. CacheRows bounds them; the cap resets the
  table and it refills from current work.
- The vocab resolves tokens through an open-addressed hash table
  verified against the model blob, 8 bytes per entry, not a Go map
  of strings.
- The delta overlay and aux sidecar unpack to heap maps; their size
  tracks the sidecar files, not the full model.

auto (default): caches bounded at 4M rows, kernel reclaims pages on
demand. Measured: ~2GB RSS warm, sub-millisecond p50.

low: caches bounded at 256k rows, GC target 60, and a timer every
three minutes runs FreeOSMemory plus a MADV_DONTNEED sweep over the
mapped model so cold pages go back to disk. NVMe re-fault is cheap;
measured idle RSS settles near 1GB and oscillates with query traffic.

max: unbounded caches, no sweeps. For machines where resident speed
matters more than the number.

## Model updates without retraining

- `q4tab index -aux -root ... -o model.bin` writes model.bin.aux
  (member tables, call tables, signatures, aliases) in minutes with
  no spill disk.
- `q4tab fetch -sha256 <hash> <url>` validates and atomically installs
  a published sidecar; a running server hot-swaps it within seconds.
- `q4tab index -incr` folds changed files into model.bin.delta, which
  the server also hot-reloads.
