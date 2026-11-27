import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  discoverProviderModels,
  parseModelId,
  formatModelId,
  normalizeModelId,
  providerSupports,
} from "../lib/ai/model-catalog.js";

test("parses all four provider prefixes", () => {
  assert.deepEqual(parseModelId("agy:gemini-4-neon-low"), {
    providerId: "agy",
    model: "gemini-4-neon-low",
  });
  assert.deepEqual(parseModelId("claude-code:claude-haiku-5-5-low"), {
    providerId: "claude-code",
    model: "claude-haiku-5-5-low",
  });
  assert.deepEqual(parseModelId("codex-cli:gpt-5.6-luna"), {
    providerId: "codex-cli",
    model: "gpt-5.6-luna",
  });
  assert.deepEqual(parseModelId("opencode:openai/gpt-5.2"), {
    providerId: "opencode",
    model: "openai/gpt-5.2",
  });
});

test("normalizes an unprefixed legacy agy model", () => {
  assert.equal(normalizeModelId("gemini-4-neon-low"), "agy:gemini-4-neon-low");
  assert.equal(formatModelId("agy", "gemini-4-neon-low"), "agy:gemini-4-neon-low");
});

test("keeps provider/model slashes inside an OpenCode model", () => {
  assert.deepEqual(parseModelId("opencode:anthropic/claude-sonnet-4"), {
    providerId: "opencode",
    model: "anthropic/claude-sonnet-4",
  });
});

test("rejects unknown prefixes", () => {
  assert.throws(() => parseModelId("ollama:llama3"), /Proveedor desconocido/);
  assert.throws(() => formatModelId("ollama", "llama3"), /Proveedor desconocido/);
});

test("advertises text for all providers and vision only for current image providers", () => {
  for (const id of ["agy", "claude-code", "codex-cli", "opencode"])
    assert.equal(providerSupports(id, "text"), true);
  assert.equal(providerSupports("agy", "vision"), true);
  assert.equal(providerSupports("claude-code", "vision"), true);
  assert.equal(providerSupports("codex-cli", "vision"), true);
  assert.equal(providerSupports("opencode", "vision"), false);
});

test("discovers visible Codex models and always includes the configured default", async (t) => {
  const root = mkdtempSync(join(tmpdir(), "nyra-codex-models-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  writeFileSync(join(root, "models_cache.json"), JSON.stringify({
    models: [
      { slug: "gpt-visible", visibility: "list" },
      { slug: "gpt-hidden", visibility: "hide" },
      { slug: "gpt-also-visible" },
    ],
  }));

  const models = await discoverProviderModels({
    id: "codex-cli",
    defaultModel: "default",
    codexHome: root,
  });

  assert.deepEqual(models, [
    "codex-cli:default",
    "codex-cli:gpt-visible",
    "codex-cli:gpt-also-visible",
  ]);
});

test("discovers OpenCode provider/model lines without damaging slashes", async () => {
  const execFileImpl = (_command, _args, _options, callback) => callback(null,
    "anthropic/claude-sonnet-4\nopenai/gpt-5.2\n");

  const models = await discoverProviderModels({
    id: "opencode",
    path: "opencode",
    defaultModel: "default",
  }, { execFileImpl });

  assert.deepEqual(models, [
    "opencode:default",
    "opencode:anthropic/claude-sonnet-4",
    "opencode:openai/gpt-5.2",
  ]);
});

test("returns only the provider default when command discovery fails", async () => {
  const execFileImpl = (_command, _args, _options, callback) => callback(new Error("plugin activation failed"), "");
  if (process.platform === "win32") { // opencode es un .cmd/.ps1: se tiene que lanzar con shell o no aparece ningún modelo
    let seen;
    await discoverProviderModels({ id: "opencode", path: "C:\\Program Files\\x\\opencode", defaultModel: "default" },
      { execFileImpl: (command, args, options, callback) => { seen = { command, args, shell: options.shell }; callback(null, ""); } });
    assert.deepEqual(seen, { command: '"C:\\Program Files\\x\\opencode" models', args: [], shell: true });
  }

  assert.deepEqual(await discoverProviderModels({
    id: "opencode",
    path: "opencode",
    defaultModel: "default",
  }, { execFileImpl }), ["opencode:default"]);
});
