import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const panel = readFileSync(join(root, "public", "panel.html"), "utf8");

test("panel exposes accessible controls for all four AI providers", () => {
  assert.match(panel, /id="providerControls"/);
  for (const id of ["agy", "claude-code", "codex-cli", "opencode"]) {
    assert.match(panel, new RegExp(`id="provider-${id}"`));
    assert.match(panel, new RegExp(`id="provider-status-${id}"`));
    assert.match(panel, new RegExp(`for="provider-${id}"`));
  }
});

test("provider toggles call the lifecycle endpoint and restore server state on failure", () => {
  assert.match(panel, /async function toggleProvider\(id, enabled\)/);
  assert.match(panel, /fetch\(`\/api\/providers\/\$\{encodeURIComponent\(id\)\}`/);
  assert.match(panel, /if \(!res\.ok\)/);
  assert.match(panel, /renderProviders\(window\._providerState\)/);
});

test("model rendering uses capability-filtered lists and disables empty controls", () => {
  assert.match(panel, /function refreshModelSelectors\(data\)/);
  assert.match(panel, /data\.visionModels/);
  assert.match(panel, /modelSelect\.disabled = models\.length === 0/);
  assert.match(panel, /visionModelSelect\.disabled = visionModels\.length === 0/);
  assert.match(panel, /IA apagada/);
});
