import { ProviderDisabledError } from "./provider-registry.js";

export function createProviderController({ registry, ai }) {
  let syncQueue = Promise.resolve();
  const syncProviders = snapshot => {
    const task = syncQueue.then(() => ai.syncProviders(snapshot));
    syncQueue = task.catch(() => {});
    return task;
  };

  const bodyFrom = snapshot => ({
    ok: true,
    ...snapshot,
    visionModel: ai.visionModel || null,
  });

  const errorResponse = (error, fallbackStatus = 400) => ({
    status: error instanceof ProviderDisabledError ? 409 : fallbackStatus,
    body: { ok: false, error: error?.message || String(error) },
  });

  return {
    getStatus() {
      return { status: 200, body: bodyFrom(registry.snapshot()) };
    },

    async setProvider(providerId, enabled) {
      if (typeof enabled !== "boolean")
        return { status: 400, body: { ok: false, error: "enabled debe ser booleano" } };
      if (!registry.snapshot().providers.some(provider => provider.id === providerId))
        return { status: 400, body: { ok: false, error: `Proveedor desconocido: ${providerId}` } };
      try {
        const snapshot = await registry.setEnabled(providerId, enabled);
        await syncProviders(snapshot);
        return { status: 200, body: bodyFrom(registry.snapshot()) };
      } catch (error) {
        return errorResponse(error, enabled ? 409 : 400);
      }
    },

    async setModel(modelId) {
      try {
        const snapshot = await registry.setCurrentModel(modelId);
        await syncProviders(snapshot);
        return { status: 200, body: bodyFrom(registry.snapshot()) };
      } catch (error) {
        return errorResponse(error);
      }
    },
  };
}
