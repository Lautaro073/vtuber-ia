import { readFile, rename, writeFile } from "node:fs/promises";

import { normalizeModelId, parseModelId } from "./model-catalog.js";

export class ProviderDisabledError extends Error {
  constructor(providerId) {
    super(`Proveedor desactivado: ${providerId}`);
    this.name = "ProviderDisabledError";
    this.providerId = providerId;
  }
}

export class ProviderRegistry {
  constructor({ definitions, settingsFile, preferredOrder, initialProvider = "agy", initialModel = null }) {
    this.definitions = new Map(definitions.map(definition => [definition.id, definition]));
    this.settingsFile = settingsFile;
    this.preferredOrder = preferredOrder;
    this.initialProvider = initialProvider;
    this.initialModel = initialModel;
    this.states = new Map();
    this.currentModel = null;
    this.persistQueue = Promise.resolve();
  }

  async initialize() {
    let saved = null;
    try { saved = JSON.parse(await readFile(this.settingsFile, "utf8")); } catch { }

    for (const id of this.preferredOrder) {
      const definition = this.definitions.get(id);
      if (!definition) continue;
      let available = false;
      let error = null;
      try { available = await definition.checkAvailable(); } catch (reason) { error = reason?.message || String(reason); }
      let models = [];
      if (available) {
        try { models = await definition.discoverModels(); } catch (reason) { error = reason?.message || String(reason); }
      }
      this.states.set(id, {
        definition,
        available,
        enabled: !!saved?.enabled?.[id] && available,
        models,
        error,
        clients: new Set(),
        revision: 0,
      });
    }

    if (!saved) {
      const preferred = this.states.get(this.initialProvider);
      const selected = preferred?.available ? preferred : this._orderedStates().find(state => state.available);
      if (selected) selected.enabled = true;
    }

    this.currentModel = this._validCurrentModel(saved?.currentModel || this.initialModel)
      || this._firstEnabledModel("text");
    await this._persist();
    return this.snapshot();
  }

  _orderedStates() {
    return this.preferredOrder.map(id => this.states.get(id)).filter(Boolean);
  }

  _validCurrentModel(modelId) {
    if (!modelId) return null;
    let normalized;
    try { normalized = normalizeModelId(modelId); } catch { return null; }
    const { providerId } = parseModelId(normalized);
    const state = this.states.get(providerId);
    return state?.enabled && state.available && state.models.includes(normalized) ? normalized : null;
  }

  _firstEnabledModel(capability) {
    for (const state of this._orderedStates()) {
      if (!state.enabled || !state.available || !state.definition.capabilities.includes(capability)) continue;
      if (state.models[0]) return state.models[0];
    }
    return null;
  }

  async _persist() {
    const save = async () => {
      const enabled = Object.fromEntries(this._orderedStates().map(state => [state.definition.id, state.enabled]));
      const data = JSON.stringify({ enabled, currentModel: this.currentModel }, null, 2) + "\n";
      const temporary = `${this.settingsFile}.tmp`;
      await writeFile(temporary, data);
      await rename(temporary, this.settingsFile);
    };
    this.persistQueue = this.persistQueue.then(save, save);
    return this.persistQueue;
  }

  async setEnabled(providerId, enabled) {
    const state = this.states.get(providerId);
    if (!state) throw new Error(`Proveedor desconocido: ${providerId}`);
    const revision = ++state.revision;
    if (enabled) {
      try {
        const available = await state.definition.checkAvailable();
        if (state.revision !== revision) return this.snapshot();
        state.available = !!available;
        state.error = available ? null : `Proveedor no encontrado: ${providerId}`;
        if (!available) {
          state.enabled = false;
          throw new Error(state.error);
        }
        state.models = await state.definition.discoverModels();
        if (state.revision !== revision) return this.snapshot();
        state.enabled = true;
      } catch (reason) {
        if (state.revision !== revision) return this.snapshot();
        state.available = false;
        state.enabled = false;
        state.error = reason?.message || String(reason);
        await this._persist();
        throw reason;
      }
    } else {
      state.enabled = false;
      await this.stopProvider(providerId);
    }
    if (!this._validCurrentModel(this.currentModel)) this.currentModel = this._firstEnabledModel("text");
    await this._persist();
    return this.snapshot();
  }

  registerClient(providerId, client) {
    const state = this.states.get(providerId);
    if (!state?.enabled) {
      client?.stop?.();
      throw new ProviderDisabledError(providerId);
    }
    state.clients.add(client);
    return client;
  }

  async createClient(role, modelId, options = {}) {
    const { providerId, model } = parseModelId(normalizeModelId(modelId));
    const state = this.states.get(providerId);
    if (!state?.enabled) throw new ProviderDisabledError(providerId);
    const revision = state.revision;
    const client = await state.definition.createClient({ role, model, modelId, options });
    if (!state.enabled || state.revision !== revision) {
      await client?.stop?.();
      throw new ProviderDisabledError(providerId);
    }
    return this.registerClient(providerId, client);
  }

  selectFallbackModel(excludedModelId = null, capability = "text") {
    for (const state of this._orderedStates()) {
      if (!state.enabled || !state.available || !state.definition.capabilities.includes(capability)) continue;
      const model = state.models.find(candidate => candidate !== excludedModelId);
      if (model) return model;
    }
    return null;
  }

  async stopProvider(providerId) {
    const state = this.states.get(providerId);
    if (!state) throw new Error(`Proveedor desconocido: ${providerId}`);
    const clients = [...state.clients];
    state.clients.clear();
    await Promise.allSettled(clients.map(client => client?.stop?.()));
  }

  async stopAll() {
    await Promise.all(this.preferredOrder.filter(id => this.states.has(id)).map(id => this.stopProvider(id)));
  }

  snapshot() {
    const providers = this._orderedStates().map(state => {
      const running = state.clients.size > 0;
      const status = !state.available ? "no encontrado"
        : !state.enabled ? "apagado"
          : running ? "activo"
            : state.definition.onDemand ? "listo bajo demanda" : "listo";
      return {
        id: state.definition.id,
        label: state.definition.label,
        enabled: state.enabled,
        available: state.available,
        running,
        status,
        error: state.error,
        models: [...state.models],
      };
    });
    const models = providers.filter(provider => provider.enabled && provider.available).flatMap(provider => provider.models);
    const visionModels = this._orderedStates()
      .filter(state => state.enabled && state.available && state.definition.capabilities.includes("vision"))
      .flatMap(state => state.models);
    return {
      providers,
      models,
      visionModels,
      currentModel: this.currentModel,
      status: this.currentModel ? "activo" : "IA apagada",
    };
  }
}
