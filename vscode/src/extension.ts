import * as cp from "child_process";
import * as fs from "fs";
import * as net from "net";
import * as path from "path";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  StreamInfo,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;
let statusBar: vscode.StatusBarItem | undefined;
let log: vscode.LogOutputChannel;
let enabled = true;
let stats: Record<string, any> = {};

interface InlineCompletionParams {
  textDocument: { uri: string };
  position: { line: number; character: number };
  context?: { triggerKind: number };
}

interface CompletionItemWire {
  insertText: string;
  range?: {
    start: { line: number; character: number };
    end: { line: number; character: number };
  };
}

interface CompletionResultWire {
  items: CompletionItemWire[];
}

const serverExe =
  process.platform === "win32" ? "q4tab.exe" : "q4tab";

function executable(p: string): boolean {
  try {
    fs.accessSync(p, fs.constants.X_OK);
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
}

// resolveServerPath picks the server binary in priority order:
// the configured path (a directory is searched for the binary inside),
// the binary bundled inside the extension, then the bare name resolved
// through PATH. Returns the command to spawn plus what was tried for
// error reporting.
function resolveServerPath(
  context: vscode.ExtensionContext,
  configured: string,
): { command: string; tried: string[] } {
  const tried: string[] = [];
  const candidates: string[] = [];

  if (configured && configured !== serverExe && configured !== "q4tab") {
    try {
      if (fs.statSync(configured).isDirectory()) {
        candidates.push(
          path.join(configured, "bin", serverExe),
          path.join(configured, serverExe),
        );
      }
    } catch {
      // stat failed. Treat as a file path anyway
    }
    candidates.push(configured);
  }
  candidates.push(path.join(context.extensionPath, "bin", serverExe));

  for (const c of candidates) {
    tried.push(c);
    if (executable(c)) {
      return { command: c, tried };
    }
    // A bundled binary can lose its exec bit through zip extraction
    // or filesystem quirks. Repair it once rather than failing.
    try {
      fs.chmodSync(c, 0o755);
      if (executable(c)) {
        return { command: c, tried };
      }
    } catch {
      // not present or not fixable. Try the next candidate
    }
  }
  // Bare name: let spawn search PATH.
  tried.push(serverExe);
  return { command: serverExe, tried };
}

function modelPath(): string {
  const cfg = vscode.workspace.getConfiguration("q4tab");
  const p = cfg.get<string>("modelPath", "");
  if (p) {
    return p;
  }
  return path.join(
    process.env.HOME || "",
    ".local/share/q4tab/model.bin",
  );
}

function setStatus(text: string, tooltip: string) {
  if (!statusBar) {
    return;
  }
  statusBar.text = text;
  statusBar.tooltip = tooltip;
  statusBar.show();
}

async function startClient(context: vscode.ExtensionContext) {
  const cfg = vscode.workspace.getConfiguration("q4tab");
  const serverAddr = cfg.get<string>("serverAddr", "");

  let serverOptions: ServerOptions;
  if (serverAddr) {
    // Remote mode: persistent TCP socket carrying the same LSP
    // framing. Deploy with `q4tab serve -listen host:7917`.
    const m = /^(?:tcp:\/\/)?([^:]+):(\d+)$/.exec(serverAddr.trim());
    if (!m) {
      vscode.window.showErrorMessage(
        `q4tab: bad serverAddr '${serverAddr}' (want host:port)`,
      );
      return;
    }
    const host = m[1];
    const port = parseInt(m[2], 10);
    log.appendLine(`server: tcp ${host}:${port}`);
    serverOptions = () =>
      new Promise<StreamInfo>((resolve, reject) => {
        const socket = net.connect(port, host);
        socket.once("connect", () =>
          resolve({ reader: socket, writer: socket }),
        );
        socket.once("error", reject);
      });
  } else {
    const { command: serverPath, tried } = resolveServerPath(
      context,
      cfg.get<string>("serverPath", "q4tab"),
    );
    log.appendLine(`server: ${serverPath} (tried: ${tried.join(", ")})`);
    const env = { ...process.env };
    const configuredModel = cfg.get<string>("modelPath", "");
    if (configuredModel) {
      env.Q4TAB_MODEL = configuredModel;
    }
    serverOptions = {
      command: serverPath,
      args: ["serve"],
      options: { env },
    };
  }

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file" }],
    synchronize: {},
    outputChannel: log,
  };

  const c = new LanguageClient(
    "q4tab",
    "q4tab",
    serverOptions,
    clientOptions,
  );

  setStatus("$(sync~spin) q4", "q4tab: starting server");
  try {
    await c.start();
  } catch (err) {
    client = undefined;
    setStatus(
      "$(error) q4",
      `q4tab: server failed to start. Click for log.`,
    );
    statusBar!.command = "q4tab.openLog";
    log.appendLine(`start failed: ${err}`);
    const target = serverAddr
      ? `server at ${serverAddr}`
      : "local server binary";
    const pick = await vscode.window.showErrorMessage(
      `q4tab: failed to start ${target}. See the q4tab output for details.`,
      "Show Log",
      "Retry",
    );
    if (pick === "Show Log") {
      log.show();
    } else if (pick === "Retry") {
      await startClient(context);
    }
    return;
  }
  client = c;
  statusBar!.command = "q4tab.menu";

  // Remote servers may require a bearer token (q4tab serve
  // -token). Authenticate this connection before any completions.
  if (serverAddr) {
    const token = cfg.get<string>("serverToken", "");
    if (token) {
      try {
        await c.sendRequest("q4/auth", { token });
      } catch (err) {
        log.appendLine(`q4/auth failed: ${err}`);
        setStatus("$(error) q4", "q4tab: auth rejected");
        vscode.window.showErrorMessage(
          "q4tab: server rejected the configured token",
        );
        c.stop();
        client = undefined;
        return;
      }
    }
  }

  try {
    stats = (await c.sendRequest("q4/status", {})) as Record<string, any>;
    const lines = stats.lines ?? 0;
    setStatus(
      lines > 0 ? `q4: ${fmtK(lines)}` : "q4: on",
      `q4tab: ${lines} corpus lines, ${stats.vocab ?? 0} vocab, ${stats.contexts ?? 0} contexts`,
    );
    if ((stats.contexts ?? 0) === 0) {
      const pick = await vscode.window.showWarningMessage(
        "q4tab: no trained model found. Index your workspace to get completions.",
        "Index Workspace",
        "Dismiss",
      );
      if (pick === "Index Workspace") {
        vscode.commands.executeCommand("q4tab.indexWorkspace");
      }
    }
  } catch (err) {
    setStatus("q4: on", "q4tab: running (status unavailable)");
    log.appendLine(`status request failed: ${err}`);
  }
}

function fmtK(n: number): string {
  if (n >= 1e6) {
    return `${(n / 1e6).toFixed(1)}M`;
  }
  if (n >= 1e3) {
    return `${(n / 1e3).toFixed(0)}k`;
  }
  return `${n}`;
}

async function stopClient() {
  if (client) {
    const c = client;
    client = undefined;
    try {
      await c.stop();
    } catch {
      // already dead
    }
  }
}

async function requestCompletion(
  document: vscode.TextDocument,
  position: vscode.Position,
): Promise<CompletionItemWire[]> {
  if (!enabled || !client) {
    return [];
  }
  const cfg = vscode.workspace.getConfiguration("q4tab");
  const timeout = cfg.get<number>("requestTimeout", 1000);
  const params: InlineCompletionParams = {
    textDocument: { uri: document.uri.toString() },
    position: { line: position.line, character: position.character },
  };
  let result: CompletionResultWire;
  try {
    result = (await Promise.race([
      client.sendRequest("q4/inlineCompletion", params),
      new Promise<never>((_, rej) =>
        setTimeout(() => rej(new Error("timeout")), timeout),
      ),
    ])) as CompletionResultWire;
  } catch (err) {
    if (`${err}`.includes("timeout")) {
      log.appendLine(`completion timeout at ${document.uri}:${position.line}`);
    }
    return [];
  }
  return result?.items ?? [];
}

class Q4InlineProvider implements vscode.InlineCompletionItemProvider {
  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    _token: vscode.CancellationToken,
  ): Promise<vscode.InlineCompletionItem[]> {
    const items = await requestCompletion(document, position);
    const cfg = vscode.workspace.getConfiguration("q4tab");
    const max = cfg.get<number>("maxSuggestions", 4);
    // The completed line text: what the user typed on this line plus
    // the accepted suggestion. Learned lines are retrievable by the
    // context that produced them.
    const linePrefix = document
      .lineAt(position.line)
      .text.substring(0, position.character);
    return items.slice(0, max).map((it) => {
      // Server-provided range means "replace to end of line": the
      // suggestion diverges from text already after the cursor.
      const range = it.range
        ? new vscode.Range(
            it.range.start.line,
            it.range.start.character,
            it.range.end.line,
            it.range.end.character,
          )
        : new vscode.Range(position, position);
      const item = new vscode.InlineCompletionItem(it.insertText, range);
      // Fires when the user accepts this ghost text. The args carry the
      // completed line and its location so the server can correlate the
      // accept with what was shown.
      item.command = {
        command: "q4tab.accepted",
        title: "accepted",
        arguments: [
          linePrefix + it.insertText,
          document.uri.toString(),
          position.line,
        ],
      };
      return item;
    });
  }
}

// Classic dropdown completions: Ctrl+Space opens the list, Tab or
// Enter accepts. Same server endpoint as the ghost-text path.
class Q4CompletionProvider implements vscode.CompletionItemProvider {
  async provideCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _token: vscode.CancellationToken,
    _context: vscode.CompletionContext,
  ): Promise<vscode.CompletionItem[]> {
    const items = await requestCompletion(document, position);
    const linePrefix = document
      .lineAt(position.line)
      .text.substring(0, position.character);
    return items.map((it, i) => {
      const ci = new vscode.CompletionItem(
        it.insertText.trim().split("\n")[0] || it.insertText,
        vscode.CompletionItemKind.Snippet,
      );
      ci.insertText = it.insertText;
      ci.detail = "q4";
      ci.sortText = `q4${String(i).padStart(3, "0")}`;
      ci.filterText = it.insertText.trimStart();
      if (it.range) {
        ci.range = new vscode.Range(
          it.range.start.line,
          it.range.start.character,
          it.range.end.line,
          it.range.end.character,
        );
      }
      ci.command = {
        command: "q4tab.accepted",
        title: "accepted",
        arguments: [
          linePrefix + it.insertText,
          document.uri.toString(),
          position.line,
        ],
      };
      return ci;
    });
  }
}

// indexWorkspace folds the open workspace folders into the model's
// incremental delta and hot-reloads it into the running server.
async function indexWorkspace(context: vscode.ExtensionContext) {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    vscode.window.showWarningMessage("q4tab: no workspace folder open");
    return;
  }
  const mp = modelPath();
  const hasModel = fs.existsSync(mp);
  const { command: bin } = resolveServerPath(
    context,
    vscode.workspace
      .getConfiguration("q4tab")
      .get<string>("serverPath", "q4tab"),
  );
  const args = ["index"];
  if (hasModel) {
    args.push("-incr");
  }
  args.push("-o", mp);
  for (const f of folders) {
    args.push("-root", f.uri.fsPath);
  }
  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: `q4tab: ${hasModel ? "updating" : "building"} model on workspace`,
      cancellable: false,
    },
    async (progress) => {
      await new Promise<void>((resolve) => {
        const proc = cp.spawn(bin, args, { env: process.env });
        proc.stderr.on("data", (d) => log.append(d.toString()));
        proc.stdout.on("data", (d) => {
          const line = d.toString().trim();
          if (line) {
            progress.report({ message: line.slice(-80) });
            log.appendLine(line);
          }
        });
        proc.on("error", (err) => {
          vscode.window.showErrorMessage(
            `q4tab: index failed to start: ${err}`,
          );
          resolve();
        });
        proc.on("close", async (code) => {
          if (code !== 0) {
            vscode.window.showErrorMessage(
              `q4tab: index exited with code ${code}`,
            );
            resolve();
            return;
          }
          if (client) {
            try {
              const r = (await client.sendRequest("q4/reindex", {})) as {
                files: number;
              };
              vscode.window.showInformationMessage(
                `q4tab: workspace indexed (${r.files} changed files merged)`,
              );
            } catch (err) {
              vscode.window.showInformationMessage(
                `q4tab: indexed. Restart the server to load it.`,
              );
              log.appendLine(`reindex failed: ${err}`);
            }
          } else {
            vscode.window.showInformationMessage(
              "q4tab: workspace indexed. Server will load it on start.",
            );
          }
          resolve();
        });
      });
    },
  );
}

async function showMenu() {
  const pick = await vscode.window.showQuickPick(
    [
      { label: "$(info) Status", cmd: "q4tab.status" },
      { label: "$(sync) Restart Server", cmd: "q4tab.restart" },
      {
        label: `$(index) ${enabled ? "Disable" : "Enable"} Completions`,
        cmd: "q4tab.toggle",
      },
      {
        label: "$(database) Index Workspace",
        cmd: "q4tab.indexWorkspace",
      },
      { label: "$(output) Show Log", cmd: "q4tab.openLog" },
    ],
    { title: "q4tab" },
  );
  if (pick) {
    vscode.commands.executeCommand(pick.cmd);
  }
}

export async function activate(context: vscode.ExtensionContext) {
  log = vscode.window.createOutputChannel("q4tab", { log: true });
  context.subscriptions.push(log);
  log.appendLine(`q4tab activating, pid=${process.pid}`);

  statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100,
  );
  statusBar.command = "q4tab.menu";
  context.subscriptions.push(statusBar);

  enabled = vscode.workspace
    .getConfiguration("q4tab")
    .get<boolean>("enabled", true);

  await startClient(context);

  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider(
      { pattern: "**" },
      new Q4InlineProvider(),
    ),
  );

  context.subscriptions.push(
    vscode.languages.registerCompletionItemProvider(
      { pattern: "**" },
      new Q4CompletionProvider(),
      ".",
      ":",
      ">",
      "(",
      "=",
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "q4tab.accepted",
      async (text: string, uri?: string, line?: number) => {
        if (!client || typeof text !== "string") {
          return;
        }
        try {
          await client.sendRequest("q4/learn", { text, uri, line });
        } catch {
          // learning is best-effort
        }
      },
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("q4tab.status", async () => {
      if (!client) {
        vscode.window.showInformationMessage("q4tab: server not running");
        return;
      }
      try {
        stats = (await client.sendRequest("q4/status", {})) as Record<
          string,
          any
        >;
      } catch (err) {
        vscode.window.showErrorMessage(`q4tab: status failed: ${err}`);
        return;
      }
      const shown = Object.entries(stats.shownN ?? {})
        .map(([k, v]) => `${k}:${v}`)
        .join(" ");
      const accept = Object.entries(stats.acceptN ?? {})
        .map(([k, v]) => `${k}:${v}`)
        .join(" ");
      vscode.window.showInformationMessage(
        `q4tab: ${stats.lines ?? 0} lines, ${stats.vocab ?? 0} vocab, ${stats.contexts ?? 0} contexts, ${stats.docs ?? 0} docs, heap ${stats.heapMB ?? 0}MB` +
          (shown ? ` | shown ${shown} accepted ${accept}` : ""),
      );
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("q4tab.restart", async () => {
      await stopClient();
      await startClient(context);
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("q4tab.toggle", async () => {
      enabled = !enabled;
      await vscode.workspace
        .getConfiguration("q4tab")
        .update("enabled", enabled, vscode.ConfigurationTarget.Global);
      setStatus(
        enabled ? "q4: on" : "q4: off",
        enabled ? "q4tab enabled" : "q4tab disabled",
      );
      vscode.window.showInformationMessage(
        `q4tab: ${enabled ? "enabled" : "disabled"}`,
      );
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("q4tab.indexWorkspace", () =>
      indexWorkspace(context),
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("q4tab.trigger", () =>
      vscode.commands.executeCommand("editor.action.inlineSuggest.trigger"),
    ),
  );

  // Next-edit jump: ask the server where a recent rewrite still
  // needs applying, move the cursor there, and let inline suggest
  // offer the rewritten line.
  context.subscriptions.push(
    vscode.commands.registerCommand("q4tab.nextEdit", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!client || !editor) {
        return;
      }
      try {
        const res = (await client.sendRequest("q4/nextEdit", {
          uri: editor.document.uri.toString(),
          limit: 8,
        })) as { items?: { uri: string; line: number; char: number }[] };
        const hint = res?.items?.[0];
        if (!hint) {
          return;
        }
        let doc = editor.document;
        if (hint.uri && hint.uri !== doc.uri.toString()) {
          doc = await vscode.workspace.openTextDocument(
            vscode.Uri.parse(hint.uri),
          );
          await vscode.window.showTextDocument(doc);
        }
        const pos = new vscode.Position(hint.line, hint.char);
        const ed = vscode.window.activeTextEditor;
        if (!ed) {
          return;
        }
        ed.selection = new vscode.Selection(pos, pos);
        ed.revealRange(
          new vscode.Range(pos, pos),
          vscode.TextEditorRevealType.InCenter,
        );
        await vscode.commands.executeCommand(
          "editor.action.inlineSuggest.trigger",
        );
      } catch (err) {
        log.appendLine(`nextEdit: ${err}`);
      }
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("q4tab.menu", showMenu),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("q4tab.openLog", () => log.show()),
  );

  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(async (e) => {
      if (
        e.affectsConfiguration("q4tab.serverPath") ||
        e.affectsConfiguration("q4tab.modelPath") ||
        e.affectsConfiguration("q4tab.serverAddr")
      ) {
        await stopClient();
        await startClient(context);
      }
      if (e.affectsConfiguration("q4tab.enabled")) {
        enabled = vscode.workspace
          .getConfiguration("q4tab")
          .get<boolean>("enabled", true);
        setStatus(enabled ? "q4: on" : "q4: off", "q4tab");
      }
    }),
  );
}

export async function deactivate() {
  statusBar?.dispose();
  await stopClient();
}
