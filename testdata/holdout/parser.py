"""Fixed eval holdout file; see server.go for the rules."""

import json
import os


def load_config(path, defaults=None):
    cfg = dict(defaults or {})
    if not os.path.exists(path):
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return cfg
    if isinstance(data, dict):
        cfg.update(data)
    return cfg


def parse_hosts(text):
    hosts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        hosts.append((parts[0], parts[1]))
    return hosts


def chunk(items, size):
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i:i + size] for i in range(0, len(items), size)]
