-- q4tab: local statistical code completion for Neovim 0.12+.
--
-- Usage:
--   require("q4tab").setup()          -- sensible defaults
--   require("q4tab").setup({ cmd = { "/path/to/q4tab", "serve" } })
--
-- Accepting a suggestion fires the server-attached command
-- (q4tab.learn) automatically, so accepted lines feed the
-- learned cache with no extra wiring.

local M = {}

local ic = vim.lsp.inline_completion

local function on_attach(args)
  local client = vim.lsp.get_client_by_id(args.data.client_id)
  if not client or client.name ~= "q4tab" then
    return
  end
  local buf = args.buf

  -- Ghost text refreshes automatically in insert mode.
  ic.enable(true, { bufnr = buf })

  local opts = { buffer = buf, silent = true }

  -- Tab accepts the ghost text, otherwise behaves like a normal Tab.
  vim.keymap.set("i", "<Tab>", function()
    if not ic.get() then
      return "<Tab>"
    end
  end, vim.tbl_extend("force", opts, {
    expr = true,
    desc = "q4tab: accept inline completion",
  }))

  -- Cycle between the candidates the server returned.
  vim.keymap.set("i", "<M-]>", function()
    ic.select({ bufnr = buf, count = 1 })
  end, vim.tbl_extend("force", opts, { desc = "q4tab: next suggestion" }))
  vim.keymap.set("i", "<M-[>", function()
    ic.select({ bufnr = buf, count = -1 })
  end, vim.tbl_extend("force", opts, { desc = "q4tab: prev suggestion" }))
end

---@class q4tab.Opts
---@field cmd? string[]              server command, default { "q4tab", "serve" }
---@field addr? string               remote server "host:port" (deploy with `q4tab serve -listen host:7917`). Overrides cmd
---@field token? string              bearer token when the server was started with -token. Sent as q4/auth after connect
---@field filetypes? string[]        restrict attachment to these filetypes
---@field root_markers? string[]     default { ".git" }
---@field keymaps? boolean           install the default keymaps (default true)
---@field completion? boolean        also enable builtin popup completion (default false)

---@param opts q4tab.Opts?
function M.setup(opts)
  opts = opts or {}

  local cmd = opts.cmd or { "q4tab", "serve" }
  if opts.addr then
    -- Remote mode: persistent TCP carrying the same LSP framing.
    local host, port = opts.addr:match("^([^:]+):(%d+)$")
    if host then
      cmd = vim.lsp.rpc.connect(host, tonumber(port))
    else
      vim.notify("q4tab: bad addr '" .. opts.addr .. "' (want host:port)", vim.log.levels.ERROR)
      return
    end
  end

  vim.lsp.config("q4tab", {
    cmd = cmd,
    root_markers = opts.root_markers or { ".git" },
    filetypes = opts.filetypes,
    on_attach = function(client, buf)
      -- Authenticate once per connection when a token is configured.
      if opts.token and not client._q4authed then
        client._q4authed = true
        client:request("q4/auth", { token = opts.token }, function(err)
          if err then
            vim.notify("q4tab: auth rejected by server", vim.log.levels.ERROR)
          end
        end)
      end
      if opts.completion then
        vim.lsp.completion.enable(true, client.id, buf, { autotrigger = false })
      end
    end,
  })
  vim.lsp.enable("q4tab")

  if opts.keymaps ~= false then
    vim.api.nvim_create_autocmd("LspAttach", { callback = on_attach })
  else
    vim.api.nvim_create_autocmd("LspAttach", {
      callback = function(args)
        local client = vim.lsp.get_client_by_id(args.data.client_id)
        if client and client.name == "q4tab" then
          ic.enable(true, { bufnr = args.buf })
        end
      end,
    })
  end
end

---Toggle ghost text for the current buffer or globally.
---@param bufnr integer? 0 or nil toggles globally
function M.toggle(bufnr)
  local filter = bufnr and { bufnr = bufnr } or {}
  ic.enable(not ic.is_enabled(filter), filter)
end

return M
