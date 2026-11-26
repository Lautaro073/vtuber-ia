import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { once } from "node:events";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { AgyClient, ClaudeClient, CodexClient, OpenCodeClient, createClientFactory } from "../lib/ai/cli-clients.js";

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "cli-fixture.mjs");

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

  assert.equal(active.killed, true);
  assert.equal(warm.killed, true);
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
  await once(child, "exit");
  assert.equal(child.killed, true);
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
