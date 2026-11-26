const PROVIDERS = new Set(["agy", "claude-code", "codex-cli", "opencode"]);
const CAPABILITIES = {
  agy: new Set(["text", "vision"]),
  "claude-code": new Set(["text", "vision"]),
  "codex-cli": new Set(["text"]),
  opencode: new Set(["text"]),
};

function assertProvider(providerId) {
  if (!PROVIDERS.has(providerId)) throw new Error(`Proveedor desconocido: ${providerId}`);
}

export function parseModelId(value) {
  const text = String(value || "");
  const separator = text.indexOf(":");
  if (separator <= 0 || separator === text.length - 1) throw new Error(`Modelo inválido: ${text}`);
  const providerId = text.slice(0, separator);
  assertProvider(providerId);
  return { providerId, model: text.slice(separator + 1) };
}

export function formatModelId(providerId, model) {
  assertProvider(providerId);
  if (!model) throw new Error("Falta el modelo");
  return `${providerId}:${model}`;
}

export function normalizeModelId(value, legacyProviderId = "agy") {
  if (!String(value || "").includes(":")) return formatModelId(legacyProviderId, value);
  const { providerId, model } = parseModelId(value);
  return formatModelId(providerId, model);
}

export function providerSupports(providerId, capability) {
  assertProvider(providerId);
  return CAPABILITIES[providerId].has(capability);
}

function run(execFileImpl, command, args) {
  return new Promise((resolve, reject) => {
    execFileImpl(command, args, { windowsHide: true, timeout: 30000 }, (error, stdout = "") =>
      error ? reject(error) : resolve(String(stdout)));
  });
}

function uniqueModelIds(providerId, defaultModel, models) {
  return [...new Set([defaultModel, ...models].filter(Boolean).map(model => formatModelId(providerId, model)))];
}

export async function discoverProviderModels(definition, dependencies = {}) {
  const { id: providerId, defaultModel = "default" } = definition;
  assertProvider(providerId);
  const execFileImpl = dependencies.execFileImpl || execFile;
  const readFileImpl = dependencies.readFileImpl || readFile;

  try {
    if (Array.isArray(definition.models))
      return uniqueModelIds(providerId, defaultModel, definition.models);

    if (providerId === "codex-cli") {
      const home = definition.codexHome
        || process.env.CODEX_HOME
        || join(process.env.USERPROFILE || "", ".codex");
      const cache = JSON.parse(await readFileImpl(join(home, "models_cache.json"), "utf8"));
      const models = (cache.models || [])
        .filter(model => model?.slug && model.visibility !== "hide")
        .map(model => model.slug);
      return uniqueModelIds(providerId, defaultModel, models);
    }

    const stdout = await run(execFileImpl, definition.path, ["models"]);
    const models = stdout.split(/\r?\n/)
      .map(line => providerId === "agy" ? line.split("\t")[0].trim() : line.trim())
      .filter(model => /^[\w./-]+$/.test(model));
    return uniqueModelIds(providerId, defaultModel, models);
  } catch {
    return uniqueModelIds(providerId, defaultModel, []);
  }
}
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
