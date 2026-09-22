-- LSP config for q4complete. Neovim 0.11+ discovers files under lsp/
-- on the runtimepath automatically. vim.lsp.enable('q4complete') then
-- attaches the server to every file buffer.
return {
  cmd = { "q4complete", "serve" },
  root_markers = { ".git" },
}
