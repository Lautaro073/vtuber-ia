import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { once } from "node:events";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { AgyClient, ClaudeClient, CodexClient, OpenCodeClient, createClientFactory } from "../lib/ai/cli-clients.js";

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "cli-fixture.mjs");
const WINDOWS_FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "cli-fixture.cmd");

function clientConfig(protocol, overrides = {}) {
  const workDir = mkdtempSync(join(tmpdir(), `nyra-${protocol}-client-`));
  return {
    path: process.execPath,
    argsPrefix: [FIXTURE],
    workDir,
    timeoutMs: 1000,
    coldMs: 0,
    turnsPerProcess: 20,
    env: { ...process.env, FIXTURE_PROTOCOL: protocol },
    ...overrides,
  };
}

function cleanupClient(t, client) {
  t.after(async () => {
    const children = [...client.children].filter(child => child.exitCode === null);
    client.stop();
    await Promise.all(children.map(child => once(child, "exit")));
    rmSync(client.config.workDir, { recursive: true, force: true });
  });
}

async function collect(stream) {
  let text = "";
  for await (const chunk of stream) text += chunk;
  return text;
}

test("AgyClient streams the existing Antigravity event shape", async (t) => {
  const client = new AgyClient("chat", "gemini-test", clientConfig("agy"));
  cleanupClient(t, client);

  assert.equal(await collect(client.stream("hola")), "reply:hola");
});

test("ClaudeClient streams the existing Claude event shape", async (t) => {
  const client = new ClaudeClient("chat", "claude-test-low", clientConfig("claude"));
  cleanupClient(t, client);

  assert.equal(await collect(client.stream("hola")), "reply:hola");
});

test("persistent clients serialize concurrent jobs", async (t) => {
  const client = new AgyClient("chat", "gemini-test", clientConfig("agy", {
    env: { ...process.env, FIXTURE_PROTOCOL: "agy", FIXTURE_DELAY_MS: "30" },
  }));
  cleanupClient(t, client);

  assert.deepEqual(await Promise.all([client.run("primero"), client.run("segundo")]), [
    "reply:primero",
    "reply:segundo",
  ]);
});

test("persistent timeout kills the hung child", async (t) => {
  const client = new AgyClient("chat", "gemini-test", clientConfig("agy", {
    timeoutMs: 30,
    env: { ...process.env, FIXTURE_PROTOCOL: "agy", FIXTURE_BEHAVIOR: "hang" },
  }));
  cleanupClient(t, client);

  await assert.rejects(client.run("sin respuesta"), /timeout/);
  assert.equal(client.proc, null);
});

test("persistent stop kills both the active and warm replacement processes", async (t) => {
  const client = new AgyClient("chat", "gemini-test", clientConfig("agy", { turnsPerProcess: 1 }));
  cleanupClient(t, client);
  await client.run("hola");
  for (let i = 0; i < 40 && !client.next; i++) await new Promise(resolve => setTimeout(resolve, 5));
  const active = client.proc;
  const warm = client.next;

  assert.ok(active);
  assert.ok(warm);
  client.stop();
  await Promise.all([active, warm].map(child => child.exitCode === null ? once(child, "exit") : null));

  assert.notEqual(active.exitCode, null);
  assert.notEqual(warm.exitCode, null);
  assert.equal(client.proc, null);
  assert.equal(client.next, null);
});

test("CodexClient parses agent_message events and clears its child", async (t) => {
  const client = new CodexClient("chat", "default", clientConfig("codex"));
  cleanupClient(t, client);

  assert.equal(await collect(client.stream("hola")), "reply:hola");
  assert.equal(client.children.size, 0);
});

test("OpenCodeClient parses text parts and clears its child", async (t) => {
  const client = new OpenCodeClient("chat", "default", clientConfig("opencode"));
  cleanupClient(t, client);

  assert.equal(await collect(client.stream("hola")), "reply:hola");
  assert.equal(client.children.size, 0);
});

test("OpenCodeClient can launch a Windows command shim", { skip: process.platform !== "win32" }, async (t) => {
  const config = clientConfig("opencode", {
    path: WINDOWS_FIXTURE,
    argsPrefix: [],
    shell: true,
    env: { ...process.env, NODE_EXE: process.execPath, FIXTURE_PROTOCOL: "opencode" },
  });
  const client = new OpenCodeClient("chat", "default", config);
  cleanupClient(t, client);

  assert.equal(await client.run("hola"), "reply:hola");
});

test("stopping a Windows command shim terminates its descendant process tree", { skip: process.platform !== "win32" }, async (t) => {
  const config = clientConfig("opencode", {
    path: WINDOWS_FIXTURE,
    argsPrefix: [],
    shell: true,
    env: {
      ...process.env,
      NODE_EXE: process.execPath,
      FIXTURE_PROTOCOL: "opencode",
      FIXTURE_BEHAVIOR: "hang",
    },
  });
  const client = new OpenCodeClient("chat", "default", config);
  let descendants = [];
  const descendantPids = parentPid => {
    const output = execFileSync("powershell.exe", [
      "-NoProfile",
      "-Command",
      `Get-CimInstance Win32_Process | Where-Object ParentProcessId -eq ${parentPid} | Select-Object -ExpandProperty ProcessId`,
    ], { encoding: "utf8" }).trim();
    return output ? output.split(/\s+/).map(Number) : [];
  };
  const isAlive = pid => {
    try { process.kill(pid, 0); return true; } catch { return false; }
  };
  t.after(() => {
    client.stop();
    for (const pid of descendants.filter(isAlive)) {
      try { execFileSync("taskkill.exe", ["/PID", String(pid), "/T", "/F"], { windowsHide: true }); } catch { }
    }
    rmSync(config.workDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  });

  const running = client.run("hola");
  for (let i = 0; i < 80 && client.children.size === 0; i++) await new Promise(resolve => setTimeout(resolve, 10));
  const wrapper = [...client.children][0];
  assert.ok(wrapper?.pid);
  for (let i = 0; i < 40 && descendants.length === 0; i++) {
    await new Promise(resolve => setTimeout(resolve, 25));
    descendants = descendantPids(wrapper.pid);
  }
  assert.ok(descendants.length > 0);
  const interrupted = assert.rejects(running, /detenido/);

  client.stop();
  await interrupted;
  for (let i = 0; i < 40 && descendants.some(isAlive); i++) await new Promise(resolve => setTimeout(resolve, 25));

  assert.deepEqual(descendants.filter(isAlive), []);
});

test("one-shot clients send long prompts through stdin", async (t) => {
  const client = new CodexClient("chat", "default", clientConfig("codex"));
  cleanupClient(t, client);
  const prompt = "ñ".repeat(40000);

  assert.equal(await client.run(prompt), `reply:${prompt}`);
});

test("one-shot clients reject malformed JSON without hanging", async (t) => {
  const client = new OpenCodeClient("chat", "default", clientConfig("opencode", {
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "malformed" },
  }));
  cleanupClient(t, client);

  await assert.rejects(client.run("hola"), /respuesta válida/);
  assert.equal(client.children.size, 0);
});

test("one-shot nonzero errors keep stderr bounded", async (t) => {
  const client = new CodexClient("chat", "default", clientConfig("codex", {
    env: { ...process.env, FIXTURE_PROTOCOL: "codex", FIXTURE_BEHAVIOR: "error" },
  }));
  cleanupClient(t, client);

  const error = await client.run("hola").then(() => null, reason => reason);
  assert.ok(error);
  assert.match(error.message, /^codex-cli terminó con código 2:/);
  assert.ok(error.message.length <= 550);
});

test("one-shot errors prefer the reason reported in the CLI JSON over stderr noise", async (t) => {
  const client = new CodexClient("chat", "default", clientConfig("codex", {
    env: { ...process.env, FIXTURE_PROTOCOL: "codex", FIXTURE_BEHAVIOR: "limit" },
  }));
  cleanupClient(t, client);

  await assert.rejects(client.run("hola"), /código 1: You've hit your usage limit\.$/);
});

test("one-shot stop kills the exact in-flight child and rejects the request", async (t) => {
  const client = new OpenCodeClient("chat", "default", clientConfig("opencode", {
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "hang" },
  }));
  cleanupClient(t, client);
  const running = client.run("hola");
  for (let i = 0; i < 40 && client.children.size === 0; i++) await new Promise(resolve => setTimeout(resolve, 5));
  const child = [...client.children][0];
  const interrupted = assert.rejects(running, /detenido/);

  client.stop();

  await interrupted;
  assert.ok(child);
  if (child.exitCode === null) await once(child, "exit");
  assert.notEqual(child.exitCode, null);
  assert.equal(client.children.size, 0);
});

test("client factory creates the adapter requested by provider ID", () => {
  const factory = createClientFactory({
    "codex-cli": { path: "codex", workDir: "C:/temp/nyra" },
  });

  const client = factory({ providerId: "codex-cli", role: "chat", model: "default" });

  assert.ok(client instanceof CodexClient);
  assert.equal(client.name, "chat");
  assert.equal(client.model, "default");
});

test("non-Claude adapters include role system prompts in their input", async (t) => {
  const agy = new AgyClient("scribe", "gemini-test", clientConfig("agy"));
  const codex = new CodexClient("scribe", "default", clientConfig("codex"));
  cleanupClient(t, agy);
  cleanupClient(t, codex);
  agy.systemPrompt = "REGLAS";
  codex.systemPrompt = "REGLAS";

  assert.equal(await agy.run("hola"), "reply:REGLAS\n\nhola");
  assert.equal(await codex.run("hola"), "reply:REGLAS\n\nhola");
});

test("Claude sends role system prompts only through its dedicated CLI flag", async (t) => {
  const claude = new ClaudeClient("scribe", "claude-test-low", clientConfig("claude"));
  cleanupClient(t, claude);
  claude.systemPrompt = "REGLAS";

  assert.equal(await claude.run("hola"), "reply:hola");
});

test("OpenCodeClient uses nyra as default agent and allows overriding or disabling it", () => {
  const def = new OpenCodeClient("chat", "nyra-ollama/nyra", clientConfig("opencode"));
  assert.deepEqual(def._cmd()[1].slice(-4), ["--agent", "nyra", "--model", "nyra-ollama/nyra"]);

  const custom = new OpenCodeClient("chat", "default", clientConfig("opencode", { agent: "build" }));
  assert.deepEqual(custom._cmd()[1].slice(-2), ["--agent", "build"]);

  const none = new OpenCodeClient("chat", "default", clientConfig("opencode", { agent: null }));
  assert.ok(!none._cmd()[1].includes("--agent"));
});

test("OpenCodeClient rejects invalid agent names before spawning", async (t) => {
  const client = new OpenCodeClient("chat", "default", clientConfig("opencode", { agent: "nyra & calc" }));
  cleanupClient(t, client);

  await assert.rejects(client.run("hola"), /Agente de OpenCode inválido/);
  assert.equal(client.children.size, 0);
});

test("OpenCodeClient preserves inherited env while applying child-only env overrides", async (t) => {
  const client = new OpenCodeClient("chat", "default", clientConfig("opencode", {
    env: { ...process.env, OPENCODE_DISABLE_PROJECT_CONFIG: "1", FIXTURE_PROTOCOL: "opencode" },
  }));
  cleanupClient(t, client);

  const [, , env] = client._cmd();
  assert.equal(env.OPENCODE_DISABLE_PROJECT_CONFIG, "1");
  assert.equal(env.PATH, process.env.PATH);
  assert.equal(process.env.OPENCODE_DISABLE_PROJECT_CONFIG, undefined);
});

test("OpenCodeClient sends multiline Unicode, quotes, and special chars through stdin", async (t) => {
  const client = new OpenCodeClient("chat", "default", clientConfig("opencode"));
  cleanupClient(t, client);
  const prompt = "Línea 1: «hola» & \"chau\" $100\nLínea 2: áéíóú ñ " + "x".repeat(12000);

  assert.equal(await client.run(prompt), `reply:${prompt}`);
});

test("OpenCodeClient rejects empty responses and timeouts cleanly", async (t) => {
  const empty = new OpenCodeClient("chat", "default", clientConfig("opencode", {
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "empty" },
  }));
  cleanupClient(t, empty);
  await assert.rejects(empty.run("hola"), /sin respuesta válida/);

  const slow = new OpenCodeClient("chat", "default", clientConfig("opencode", {
    timeoutMs: 30,
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "hang" },
  }));
  cleanupClient(t, slow);
  await assert.rejects(slow.run("hola"), /opencode timeout/);
  const [child] = [...slow.children];
  if (child && child.exitCode === null) await once(child, "exit");
  assert.equal(slow.children.size, 0);
});

test("OpenCodeClient deduplicates duplicate events, handles cumulative snapshots, and keeps incremental deltas", async (t) => {
  const dup = new OpenCodeClient("chat", "default", clientConfig("opencode", {
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "dup_events" },
  }));
  cleanupClient(t, dup);
  assert.equal(await dup.run("hola"), "Hola chat");

  const cum = new OpenCodeClient("chat", "default", clientConfig("opencode", {
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "cumulative" },
  }));
  cleanupClient(t, cum);
  assert.equal(await cum.run("hola"), "Hola chat!");

  const inc = new OpenCodeClient("chat", "default", clientConfig("opencode", {
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "incremental" },
  }));
  cleanupClient(t, inc);
  assert.equal(await collect(inc.stream("hola")), "Hola chat!");
});

test("Codex attaches screenshots with -i before reading the prompt from stdin", () => {
  const client = new CodexClient("vision", "default", { path: "codex", workDir: tmpdir() });
  const [, args] = client._cmd(null, ["C:\tmp\screen.jpg"]);
  assert.deepEqual(args.slice(-3), ["-i", "C:\tmp\screen.jpg", "-"]);
  assert.equal(client._cmd()[1].includes("-i"), false);
});
