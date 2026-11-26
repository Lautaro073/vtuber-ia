import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { OpenCodeServerManager } from "../lib/ai/opencode-server.js";
import { OpenCodeClient } from "../lib/ai/cli-clients.js";

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "fixtures", "cli-fixture.mjs");

function makeManager(overrides = {}) {
  const workDir = mkdtempSync(join(tmpdir(), "nyra-oc-srv-"));
  const manager = new OpenCodeServerManager({
    path: process.execPath,
    argsPrefix: [FIXTURE],
    workDir,
    startTimeoutMs: 3000,
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode" },
    ...overrides,
  });
  return { manager, workDir };
}

test("OpenCodeServerManager starts on 127.0.0.1 with ephemeral auth and prevents duplicates", async (t) => {
  const { manager, workDir } = makeManager();
  t.after(async () => { await manager.stop(); rmSync(workDir, { recursive: true, force: true }); });

  const [u1, u2] = await Promise.all([manager.ensureStarted(), manager.ensureStarted()]);
  assert.ok(u1);
  assert.equal(u1, u2);
  assert.match(u1, /^http:\/\/127\.0\.0\.1:\d+$/);
  assert.ok(manager.isRunning());
  assert.ok(manager._password && manager._password.length >= 32);

  // Unauthenticated request to fixture server must fail 401
  const unauth = await fetch(`${u1}/api/session`);
  assert.equal(unauth.status, 401);
});

test("OpenCodeServerManager handles startup failures, timeouts, and bounded restarts", async (t) => {
  const { manager: failMgr, workDir: w1 } = makeManager({
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "serve_fail" },
    maxRestarts: 2,
    restartWindowMs: 10000,
  });
  t.after(async () => { await failMgr.stop(); rmSync(w1, { recursive: true, force: true }); });

  assert.equal(await failMgr.ensureStarted(), null);
  assert.equal(await failMgr.ensureStarted(), null);
  // Third attempt is blocked by bounded restart cap without spawning
  assert.equal(await failMgr.ensureStarted(), null);

  const { manager: hangMgr, workDir: w2 } = makeManager({
    startTimeoutMs: 150,
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "serve_hang" },
  });
  t.after(async () => { await hangMgr.stop(); rmSync(w2, { recursive: true, force: true }); });
  assert.equal(await hangMgr.ensureStarted(), null);
  assert.equal(hangMgr.isRunning(), false);
});

test("OpenCodeServerManager cancels in-flight startup on stop()", async (t) => {
  const { manager, workDir } = makeManager({
    startTimeoutMs: 2000,
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "serve_hang" },
  });
  t.after(async () => { await manager.stop(); rmSync(workDir, { recursive: true, force: true }); });

  const starting = manager.ensureStarted();
  await new Promise(r => setTimeout(r, 40));
  await manager.stop();
  assert.equal(await starting, null);
  assert.equal(manager.isRunning(), false);
});

test("OpenCodeClient uses --server when manager is active, cleans NYRA sessions, and falls back to standalone", async (t) => {
  const { manager, workDir } = makeManager();
  t.after(async () => { await manager.stop(); rmSync(workDir, { recursive: true, force: true }); });

  const client = new OpenCodeClient("chat", "nyra-ollama/nyra", {
    path: process.execPath,
    argsPrefix: [FIXTURE],
    workDir,
    timeoutMs: 3000,
    serverManager: manager,
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode" },
  });
  t.after(() => client.stop());

  assert.equal(await client.run("hola"), "reply:hola");
  await new Promise(r => setTimeout(r, 40));
  const check = await fetch(`${manager.url}/api/session`, {
    headers: { Authorization: manager._authHeader() },
  }).then(r => r.json());
  assert.equal(check.deleted.length, 1);
  assert.match(check.deleted[0], /^ses_nyra_srv_/);

  // Fallback to standalone when --server fails with ECONNREFUSED
  const fbClient = new OpenCodeClient("chat", "nyra-ollama/nyra", {
    path: process.execPath,
    argsPrefix: [FIXTURE],
    workDir,
    timeoutMs: 3000,
    serverManager: manager,
    env: { ...process.env, FIXTURE_PROTOCOL: "opencode", FIXTURE_BEHAVIOR: "server_down" },
  });
  t.after(() => fbClient.stop());
  assert.equal(await fbClient.run("respaldo"), "reply:respaldo");
});
