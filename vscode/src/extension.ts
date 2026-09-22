import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;
let statusBar: vscode.StatusBarItem | undefined;

interface InlineCompletionParams {
  textDocument: { uri: string };
  position: { line: number; character: number };
  context?: { triggerKind: number };
}

interface InlineCompletionResult {
  items: { insertText: string }[];
}

async function startClient(context: vscode.ExtensionContext) {
  const cfg = vscode.workspace.getConfiguration("q4complete");
  const serverPath = cfg.get<string>("serverPath", "q4complete");
  const modelPath = cfg.get<string>("modelPath", "");

  const env = { ...process.env };
  if (modelPath) {
    env.Q4COMPLETE_MODEL = modelPath;
  }

  const serverOptions: ServerOptions = {
    command: serverPath,
    args: ["serve"],
    options: { env },
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file" }],
    synchronize: {},
    outputChannelName: "q4complete",
  };

  client = new LanguageClient(
    "q4complete",
    "q4complete",
    serverOptions,
    clientOptions,
  );

  try {
    await client.start();
  } catch (err) {
    vscode.window.showErrorMessage(
      `q4complete: failed to start server '${serverPath}': ${err}`,
    );
    return;
  }

  const stats = (await client.sendRequest("q4/status", {})) as Record<
    string,
    number
  >;
  statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100,
  );
  statusBar.text = `q4: ${stats.lines ?? 0} lines`;
  statusBar.tooltip = `q4complete model: ${stats.vocab ?? 0} tokens, ${stats.contexts ?? 0} contexts, ${stats.lines ?? 0} lines`;
  statusBar.command = "q4complete.status";
  statusBar.show();
  context.subscriptions.push(statusBar);
}

class Q4InlineProvider implements vscode.InlineCompletionItemProvider {
  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken,
  ): Promise<vscode.InlineCompletionItem[]> {
    const cfg = vscode.workspace.getConfiguration("q4complete");
    if (!cfg.get<boolean>("enabled", true) || !client) {
      return [];
    }

    const params: InlineCompletionParams = {
      textDocument: { uri: document.uri.toString() },
      position: { line: position.line, character: position.character },
    };

    let result: InlineCompletionResult;
    try {
      result = (await Promise.race([
        client.sendRequest("q4/inlineCompletion", params, token),
        new Promise<never>((_, rej) =>
          setTimeout(() => rej(new Error("timeout")), 400),
        ),
      ])) as InlineCompletionResult;
    } catch {
      return [];
    }

    if (!result || !result.items) {
      return [];
    }
    const max = cfg.get<number>("maxSuggestions", 4);
    return result.items.slice(0, max).map(
      (it) =>
        new vscode.InlineCompletionItem(
          it.insertText,
          new vscode.Range(position, position),
        ),
    );
  }
}

export async function activate(context: vscode.ExtensionContext) {
  await startClient(context);

  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider(
      { pattern: "**" },
      new Q4InlineProvider(),
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("q4complete.status", async () => {
      if (!client) {
        vscode.window.showInformationMessage("q4complete: server not running");
        return;
      }
      const stats = (await client.sendRequest("q4/status", {})) as Record<
        string,
        number
      >;
      vscode.window.showInformationMessage(
        `q4complete: ${stats.lines ?? 0} corpus lines, ${stats.vocab ?? 0} vocab, ${stats.contexts ?? 0} contexts, ${stats.docs ?? 0} open docs`,
      );
    }),
  );
}

export async function deactivate() {
  statusBar?.dispose();
  if (client) {
    await client.stop();
  }
}
