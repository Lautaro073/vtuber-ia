import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ProviderDisabledError, ProviderRegistry } from "../lib/ai/provider-registry.js";

const ORDER = ["agy", "claude-code", "codex-cli", "opencode"];

function definitions(overrides = {}) {
  return ORDER.map(id => ({
    id,
    label: id,
    capabilities: id === "agy" || id === "claude-code" ? ["text", "vision"] : ["text"],
    onDemand: id === "codex-cli" || id === "opencode",
    checkAvailable: async () => true,
    discoverModels: async () => [`${id}:${id}-model`],
    createClient: async ({ role, model }) => ({ name: role, model, stop() {} }),
    ...overrides[id],
  }));
}

function registryAt(t, options = {}) {
  const root = mkdtempSync(join(tmpdir(), "nyra-provider-registry-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const settingsFile = join(root, "providers.json");
  return {
    settingsFile,
    registry: new ProviderRegistry({
      definitions: definitions(options.overrides),
      settingsFile,
      preferredOrder: ORDER,
      initialProvider: options.initialProvider || "agy",
      initialModel: options.initialModel,
    }),
  };
}

test("initialization enables only the configured provider without saved settings", async (t) => {
  const { registry } = registryAt(t, { initialProvider: "claude-code" });

  await registry.initialize();
  const state = registry.snapshot();

  assert.deepEqual(state.providers.filter(provider => provider.enabled).map(provider => provider.id), ["claude-code"]);
  assert.equal(state.currentModel, "claude-code:claude-code-model");
});

test("a corrupt settings file falls back to the configured provider", async (t) => {
  const { registry, settingsFile } = registryAt(t, { initialProvider: "codex-cli" });
  writeFileSync(settingsFile, "{not json");

  await registry.initialize();

  assert.deepEqual(registry.snapshot().providers.filter(provider => provider.enabled).map(provider => provider.id), ["codex-cli"]);
  assert.doesNotThrow(() => JSON.parse(readFileSync(settingsFile, "utf8")));
});

test("saved provider booleans survive a second registry instance", async (t) => {
  const { registry, settingsFile } = registryAt(t);
  await registry.initialize();
  await registry.setEnabled("opencode", true);
  await registry.setEnabled("agy", false);

  const restored = new ProviderRegistry({
    definitions: definitions(),
    settingsFile,
    preferredOrder: ORDER,
    initialProvider: "agy",
  });
  await restored.initialize();

  assert.deepEqual(restored.snapshot().providers.filter(provider => provider.enabled).map(provider => provider.id), ["opencode"]);
});

test("unknown saved provider keys are discarded", async (t) => {
  const { registry, settingsFile } = registryAt(t);
  writeFileSync(settingsFile, JSON.stringify({
    enabled: { agy: false, "claude-code": true, mystery: true },
    currentModel: "claude-code:claude-code-model",
  }));

  await registry.initialize();

  const saved = JSON.parse(readFileSync(settingsFile, "utf8"));
  assert.deepEqual(Object.keys(saved.enabled).sort(), [...ORDER].sort());
  assert.equal(Object.hasOwn(saved.enabled, "mystery"), false);
});

test("disabling a provider stops every owned client and removes its models", async (t) => {
  const clients = [];
  const { registry } = registryAt(t, {
    overrides: {
      agy: {
        createClient: async ({ role, model }) => {
          const client = {
            name: role,
            model,
            stopped: false,
            warmReplacementAlive: true,
            stop() { this.stopped = true; this.warmReplacementAlive = false; },
          };
          clients.push(client);
          return client;
        },
      },
    },
  });
  await registry.initialize();
  await registry.createClient("chat", "agy:agy-model");
  await registry.createClient("ahead", "agy:agy-model");

  await registry.setEnabled("agy", false);

  assert.equal(clients.length, 2);
  assert.equal(clients.every(client => client.stopped && !client.warmReplacementAlive), true);
  assert.deepEqual(registry.snapshot().models, []);
  assert.equal(registry.snapshot().providers.find(provider => provider.id === "agy").running, false);
  await assert.rejects(() => registry.createClient("chat", "agy:agy-model"), ProviderDisabledError);
});

test("a disabled provider is never selected as fallback", async (t) => {
  const { registry } = registryAt(t);
  await registry.initialize();
  await registry.setEnabled("claude-code", true);
  await registry.setEnabled("agy", false);

  assert.equal(registry.selectFallbackModel(null, "text"), "claude-code:claude-code-model");
  assert.equal(registry.selectFallbackModel("claude-code:claude-code-model", "text"), null);
});

test("registerClient subjects externally created roles to provider shutdown", async (t) => {
  const { registry } = registryAt(t);
  await registry.initialize();
  const client = { stopCalled: 0, stop() { this.stopCalled++; } };

  registry.registerClient("agy", client);
  await registry.stopProvider("agy");

  assert.equal(client.stopCalled, 1);
  assert.equal(registry.snapshot().providers.find(provider => provider.id === "agy").running, false);
});

test("enabling rechecks a CLI that became available after startup", async (t) => {
  let installed = false;
  const { registry } = registryAt(t, {
    overrides: {
      opencode: { checkAvailable: async () => installed },
    },
  });
  await registry.initialize();
  installed = true;

  await registry.setEnabled("opencode", true);

  const provider = registry.snapshot().providers.find(item => item.id === "opencode");
  assert.equal(provider.available, true);
  assert.equal(provider.enabled, true);
  assert.deepEqual(provider.models, ["opencode:opencode-model"]);
});

test("the last enable or disable request wins when availability checks overlap", async (t) => {
  let checkCount = 0;
  let releaseCheck;
  let notifyCheckStarted;
  const checkStarted = new Promise(resolve => { notifyCheckStarted = resolve; });
  const delayedCheck = new Promise(resolve => { releaseCheck = resolve; });
  const { registry } = registryAt(t, {
    overrides: {
      opencode: {
        checkAvailable: async () => {
          checkCount++;
          if (checkCount === 1) return true;
          notifyCheckStarted();
          await delayedCheck;
          return true;
        },
      },
    },
  });
  await registry.initialize();

  const enabling = registry.setEnabled("opencode", true);
  const started = await Promise.race([
    checkStarted.then(() => true),
    new Promise(resolve => setTimeout(() => resolve(false), 50)),
  ]);
  if (!started) {
    releaseCheck();
    await enabling;
    assert.equal(checkCount, 2);
    return;
  }
  const disabling = registry.setEnabled("opencode", false);
  releaseCheck();
  await Promise.allSettled([enabling, disabling]);

  assert.equal(checkCount, 2);
  assert.equal(registry.snapshot().providers.find(provider => provider.id === "opencode").enabled, false);
});

test("disabling a provider interrupts its in-flight request", async (t) => {
  let rejectRun;
  const { registry } = registryAt(t, {
    overrides: {
      agy: {
        createClient: async ({ role, model }) => ({
          name: role,
          model,
          run: () => new Promise((_resolve, reject) => { rejectRun = reject; }),
          stop: () => rejectRun?.(new ProviderDisabledError("agy")),
        }),
      },
    },
  });
  await registry.initialize();
  const client = await registry.createClient("chat", "agy:agy-model");
  const running = client.run("hola");
  const interrupted = assert.rejects(running, ProviderDisabledError);

  await registry.setEnabled("agy", false);

  await interrupted;
});

test("disabling the active provider selects the first enabled provider in order", async (t) => {
  const { registry } = registryAt(t);
  await registry.initialize();
  await registry.setEnabled("opencode", true);
  await registry.setEnabled("claude-code", true);

  await registry.setEnabled("agy", false);

  assert.equal(registry.snapshot().currentModel, "claude-code:claude-code-model");
});

test("disabling every provider leaves NYRA in IA apagada state", async (t) => {
  const { registry } = registryAt(t);
  await registry.initialize();

  await registry.setEnabled("agy", false);

  assert.equal(registry.snapshot().currentModel, null);
  assert.equal(registry.snapshot().status, "IA apagada");
});
