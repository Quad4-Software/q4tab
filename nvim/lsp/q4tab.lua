-- LSP config for q4tab. Neovim 0.11+ discovers files under lsp/
-- on the runtimepath automatically. vim.lsp.enable('q4tab') then
-- attaches the server to every file buffer.
return {
  cmd = { "q4tab", "serve" },
  root_markers = { ".git" },
}
