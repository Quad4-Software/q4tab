// Fixed eval holdout file; see server.go for the rules.

"use strict";

const DEFAULT_TIMEOUT_MS = 5000;
const MAX_RETRIES = 3;

async function fetchJSON(url, opts = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), opts.timeout || DEFAULT_TIMEOUT_MS);
  try {
    const res = await fetch(url, { signal: ctrl.signal });
    if (!res.ok) {
      throw new Error(`http ${res.status}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function fetchRetry(url, opts = {}) {
  let lastErr = null;
  for (let i = 0; i < MAX_RETRIES; i++) {
    try {
      return await fetchJSON(url, opts);
    } catch (err) {
      lastErr = err;
      await new Promise((r) => setTimeout(r, 100 * (i + 1)));
    }
  }
  throw lastErr;
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    if (t !== null) {
      clearTimeout(t);
    }
    t = setTimeout(() => fn(...args), ms);
  };
}

module.exports = { fetchJSON, fetchRetry, debounce };
