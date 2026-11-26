import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { createProviderController } from "../lib/ai/provider-controller.js";
import { ProviderRegistry } from "../lib/ai/provider-registry.js";

function setup(t, availability = {}) {
  const root = mkdtempSync(join(tmpdir(), "nyra-provider-controller-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const ids = ["agy", "claude-code", "codex-cli", "opencode"];
  const registry = new ProviderRegistry({
    definitions: ids.map(id => ({
      id,
      label: id,
      capabilities: id === "agy" || id === "claude-code" ? ["text", "vision"] : ["text"],
      onDemand: id === "codex-cli" || id === "opencode",
      checkAvailable: async () => availability[id] ?? true,
      discoverModels: async () => [`${id}:${id}-model`],
      createClient: async ({ role, model }) => ({ name: role, model, stop() {} }),
    })),
    settingsFile: join(root, "providers.json"),
    preferredOrder: ids,
    initialProvider: "agy",
  });
  const syncs = [];
  const ai = {
    visionModel: "agy:agy-model",
    async syncProviders(snapshot) { syncs.push(snapshot.currentModel); },
  };
  return { registry, ai, syncs, controller: createProviderController({ registry, ai }) };
}

test("provider controller enables and disables a valid provider", async (t) => {
  const { registry, controller, syncs } = setup(t);
  await registry.initialize();

  const enabled = await controller.setProvider("codex-cli", true);
  const disabled = await controller.setProvider("codex-cli", false);

  assert.equal(enabled.status, 200);
  assert.equal(enabled.body.providers.find(provider => provider.id === "codex-cli").enabled, true);
  assert.equal(disabled.body.providers.find(provider => provider.id === "codex-cli").enabled, false);
  assert.deepEqual(syncs, ["agy:agy-model", "agy:agy-model"]);
});

test("provider controller rejects invalid IDs and non-boolean values", async (t) => {
  const { registry, controller } = setup(t);
  await registry.initialize();

  assert.equal((await controller.setProvider("mystery", true)).status, 400);
  assert.equal((await controller.setProvider("agy", "false")).status, 400);
});

test("provider controller returns conflict when an unavailable CLI is enabled", async (t) => {
  const { registry, controller } = setup(t, { opencode: false });
  await registry.initialize();

  const response = await controller.setProvider("opencode", true);

  assert.equal(response.status, 409);
  assert.match(response.body.error, /no encontrado/i);
});

test("disabling the active provider switches the controller to the first enabled engine", async (t) => {
  const { registry, controller, syncs } = setup(t);
  await registry.initialize();
  await controller.setProvider("claude-code", true);

  const response = await controller.setProvider("agy", false);

  assert.equal(response.body.currentModel, "claude-code:claude-code-model");
  assert.equal(syncs.at(-1), "claude-code:claude-code-model");
});

test("controller status reports IA apagada and both model lists when all providers are off", async (t) => {
  const { registry, controller } = setup(t);
  await registry.initialize();
  await controller.setProvider("agy", false);

  const response = controller.getStatus();

  assert.equal(response.status, 200);
  assert.equal(response.body.status, "IA apagada");
  assert.deepEqual(response.body.models, []);
  assert.deepEqual(response.body.visionModels, []);
  assert.equal(response.body.currentModel, null);
});

test("model changes reject disabled providers and accept enabled ones", async (t) => {
  const { registry, controller } = setup(t);
  await registry.initialize();

  assert.equal((await controller.setModel("codex-cli:codex-cli-model")).status, 409);
  await controller.setProvider("codex-cli", true);
  const changed = await controller.setModel("codex-cli:codex-cli-model");

  assert.equal(changed.status, 200);
  assert.equal(changed.body.currentModel, "codex-cli:codex-cli-model");
});

test("overlapping controller updates cannot leave AI roles on a stale provider", async (t) => {
  const { registry, ai } = setup(t);
  await registry.initialize();
  let releaseFirst;
  let notifyFirst;
  let first = true;
  const firstStarted = new Promise(resolve => { notifyFirst = resolve; });
  const firstGate = new Promise(resolve => { releaseFirst = resolve; });
  ai.activeModel = null;
  ai.syncProviders = async snapshot => {
    if (first) {
      first = false;
      notifyFirst();
      await firstGate;
    }
    ai.activeModel = snapshot.currentModel;
  };
  const controller = createProviderController({ registry, ai });

  const enabling = controller.setProvider("claude-code", true);
  await firstStarted;
  const disabling = controller.setProvider("agy", false);
  await new Promise(resolve => setTimeout(resolve, 10));
  releaseFirst();
  await Promise.all([enabling, disabling]);

  assert.equal(registry.snapshot().currentModel, "claude-code:claude-code-model");
  assert.equal(ai.activeModel, "claude-code:claude-code-model");
});
