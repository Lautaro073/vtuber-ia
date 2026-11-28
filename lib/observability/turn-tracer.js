// Fase 4A: módulo central de trazas e instrumentación de latencia de NYRA.
// Garantías:
//  - Desactivado por defecto para detalle de eventos (enabled=false): costo mínimo en el camino crítico.
//  - Reloj monotónico inyectable (performance.now() en Node). Nunca resta relojes de procesos distintos.
//  - Identificadores de correlación únicos, nunca derivados del texto del mensaje.
//  - Memoria estrictamente acotada (historial, trazas activas, eventos por turno, expiración de abandonadas).
//  - Aislamiento de fallos: ningún error de telemetría puede propagar ni bloquear LLM, TTS o reproducción.
//  - Privacidad y seguridad: no guarda prompts, texto privado, memorias, referencias vocales ni credenciales.
import { randomUUID } from "node:crypto";

export const TRACE_SCHEMA_VERSION = "1.0.0";

export const VALID_CATEGORIES = new Set([
  "mock",
  "simulation",
  "historical_replay",
  "isolated_browser",
  "real_gateway",
  "real_obs",
]);

export const TERMINAL_STATES = new Set([
  "completed",
  "cancelled",
  "expired",
  "failed",
  "partial_failure",
  "discarded",
  "abandoned",
]);

// Claves que nunca deben entrar a metadatos/exportaciones
const FORBIDDEN_KEY_RE = /^(?:api[_-]?key|token|secret|password|authorization|cookie|prompt|systemprompt|rawprompt|memory|memories|text|fulltext|filtered|said|transcript|message|body|content|referencepath|voicepath|usertext|chattext|username|uniqueid|userid|email|phone|ip|url|endpoint|host|hostname)$/i;
const SECRET_VALUE_RE = /(?:bearer\s+\S+|sk-[a-z0-9_-]{8,}|https?:\/\/\S+|file:\/\/\S+|[a-z]:\\[^\s]+|\/(?:Users|home|content)\/\S+)/i;

export const VALID_START_METHODS = new Set([
  "worklet_first_render",
  "html_audio_onplaying",
  "webaudio_buffer_start",
  "unknown",
]);

export function sanitizeValue(val, depth = 0) {
  if (val === null || val === undefined) return null;
  if (typeof val === "number") return Number.isFinite(val) ? val : null;
  if (typeof val === "boolean") return val;
  if (typeof val === "string") {
    if (SECRET_VALUE_RE.test(val)) return "[REDACTED]";
    return val.length > 160 ? `${val.slice(0, 157)}...` : val;
  }
  if (depth >= 4) return null;
  if (Array.isArray(val)) return val.slice(0, 32).map(v => sanitizeValue(v, depth + 1));
  if (typeof val === "object") {
    const out = {};
    let count = 0;
    for (const [k, v] of Object.entries(val)) {
      if (count >= 48) break;
      if (FORBIDDEN_KEY_RE.test(k)) continue;
      out[k] = sanitizeValue(v, depth + 1);
      count += 1;
    }
    return out;
  }
  return null;
}

export function sanitizeReasonCode(raw, fallback = "unknown") {
  const s = String(raw || "").trim();
  if (!s || SECRET_VALUE_RE.test(s)) return fallback;
  return s.slice(0, 96);
}

const r2 = v => (Number.isFinite(v) ? Number(v.toFixed(2)) : null);
const r3 = v => (Number.isFinite(v) ? Number(v.toFixed(3)) : null);
const nonNeg = (a, b) => (Number.isFinite(a) && Number.isFinite(b) ? Math.max(0, r2(b - a)) : null);

export class TurnTracer {
  constructor({
    enabled = false,
    category = "real_gateway",
    now = () => performance.now(),
    wallNow = () => Date.now(),
    maxHistory = 200,
    maxActive = 100,
    maxEventsPerTurn = 128,
    abandonTimeoutMs = 300000,
    onError = null,
  } = {}) {
    this.enabled = Boolean(enabled);
    this.category = VALID_CATEGORIES.has(category) ? category : "mock";
    this.now = typeof now === "function" ? now : () => performance.now();
    this.wallNow = typeof wallNow === "function" ? wallNow : () => Date.now();
    this.maxHistory = Math.max(1, Number(maxHistory) || 200);
    this.maxActive = Math.max(1, Number(maxActive) || 100);
    this.maxEventsPerTurn = Math.max(8, Number(maxEventsPerTurn) || 128);
    this.abandonTimeoutMs = Math.max(100, Number(abandonTimeoutMs) || 300000);
    this.onError = typeof onError === "function" ? onError : null;

    this._seq = 0;
    this.active = new Map();       // turnId -> trace
    this.history = [];             // completed traces (bounded)
    this.byUrl = new Map();        // tts url ("stream:N" | "/tts/...") -> turnId
    this.byStreamId = new Map();   // numeric streamId -> turnId
    this.byMessageId = new Map();  // messageId -> turnId
    this.droppedEvents = 0;
    this.evictedActive = 0;
    this.internalErrors = 0;
  }

  _safe(fn, fallback = null) {
    try {
      return fn();
    } catch (err) {
      this.internalErrors += 1;
      try { this.onError?.(err); } catch { }
      return fallback;
    }
  }

  nextId(prefix = "id") {
    this._seq += 1;
    const tag = randomUUID().slice(0, 8);
    return `${prefix}_${this._seq}_${tag}`;
  }

  isEnabled() { return this.enabled === true; }
  setEnabled(v) { this.enabled = Boolean(v); }

  // Limpieza perezosa de trazas abandonadas (sin timers que mantengan vivo el proceso)
  sweepAbandoned() {
    return this._safe(() => {
      const t = this.now();
      let swept = 0;
      for (const tr of [...this.active.values()]) {
        const last = tr._lastTouchedMs ?? tr._originMonoMs;
        if (t - last > this.abandonTimeoutMs) {
          this._closeTrace(tr, "abandoned", "abandon_timeout");
          swept += 1;
        }
      }
      return swept;
    }, 0);
  }

  // Inicia una traza de turno en recepción / encolado. Si enabled=false, devuelve null (sin costo).
  // Nota: receptionClockNote documenta explícitamente que t0 es recepción local en NYRA, no envío en TikTok/Discord.
  startTurn({
    messageId = null,
    turnId = null,
    source = "live",
    platform = "unknown",
    priority = 4,
    cancelEpoch = 0,
    category = null,
    warmup = false,
    receivedAtMonoMs = null,
    requestedProvider = null,
    requestedEngine = null,
    playbackMode = null,
  } = {}) {
    if (!this.enabled) return null;
    return this._safe(() => {
      this.sweepAbandoned();
      while (this.active.size >= this.maxActive) {
        const oldest = this.active.values().next().value;
        if (!oldest) break;
        this.evictedActive += 1;
        this._closeTrace(oldest, "abandoned", "max_active_evicted");
      }
      const t0 = Number.isFinite(receivedAtMonoMs) ? receivedAtMonoMs : this.now();
      const mid = messageId || this.nextId("msg");
      const tid = turnId || this.nextId("trn");
      const cat = category && VALID_CATEGORIES.has(category) ? category : this.category;
      const tr = {
        schemaVersion: TRACE_SCHEMA_VERSION,
        ids: {
          messageId: mid,
          turnId: tid,
          llmJobIds: [],
          ttsJobIds: [],
          httpRequestIds: [],
          streamIds: [],
          segmentIds: [],
          playbackIds: [],
        },
        category: cat,
        warmup: Boolean(warmup),
        source: String(source || "live"),
        platform: String(platform || "unknown"),
        priority: Number.isFinite(Number(priority)) ? Number(priority) : 4,
        cancelEpoch: Number.isFinite(Number(cancelEpoch)) ? Number(cancelEpoch) : 0,
        requestedProvider: requestedProvider || null,
        effectiveProvider: requestedProvider || null,
        requestedEngine: requestedEngine || null,
        effectiveEngine: requestedEngine || null,
        playbackMode: playbackMode || null,
        state: "active",
        terminalReason: null,
        startedAtIso: new Date(this.wallNow()).toISOString(),
        endedAtIso: null,
        clocks: {
          serverClock: "node_monotonic_ms",
          browserClock: "browser_monotonic_ms_isolated",
          crossProcessNote: "Las duraciones locales del navegador nunca se restan contra relojes de Node. El ACK de reproducción recibido en el servidor incluye el retraso de retorno unidireccional desde el navegador (no equivale automáticamente a RTT completo ni al sonido físico en OBS/parlantes).",
          receptionClockNote: "receivedAt es la recepción local del evento en NYRA; no equivale al instante en que el usuario envió el mensaje en TikTok o Discord.",
        },
        _originMonoMs: t0,
        _lastTouchedMs: t0,
        // Etapas como intervalos independientes sobre la línea temporal del turno (offset ms desde _originMonoMs)
        queue: {
          receivedRelMs: 0,
          enqueuedRelMs: null,
          holdUntilRelMs: null,
          prepareStartRelMs: null,
          prepareEndRelMs: null,
          playbackTurnStartRelMs: null,
          playbackTurnEndRelMs: null,
          conversationalQueueWaitMs: null,
          prepareAheadWaitMs: null,
          prepareAheadOverlapped: false,
          expired: false,
          cancelled: false,
          replaced: false,
          cancelReason: null,
        },
        llm: {
          attempts: [],
          retries: 0,
          fallbackUsed: false,
          cancelled: false,
          providerRequested: requestedProvider || null,
          providerEffective: requestedProvider || null,
          streamingCapability: "unknown", // "incremental" | "full_only" | "unknown"
          startRelMs: null,
          firstEventRelMs: null,
          firstTextDeltaRelMs: null, // null para proveedores de respuesta completa
          firstSentenceCandidateRelMs: null,
          completeResponseRelMs: null,
          endRelMs: null,
          firstEventLatencyMs: null,
          firstTextLatencyMs: null,
          firstSentenceLatencyMs: null,
          totalMs: null,
          validationMs: null,
          errorCode: null,
        },
        validations: {
          startRelMs: null,
          endRelMs: null,
          totalMs: null,
          sanitizeMs: null,
          filtersMs: null,
          responseValidationMs: null,
          emotionExtractionMs: null,
          phoneticsMs: null,
          ttsTransformMs: null,
          status: null,       // "ok" | "refused" | "blocked" | "empty" | "safety_intervene" | "expired"
          reasonCode: null,
        },
        overlap: {
          enabled: false,
          providerEligible: false,
          backendAvailable: null,
          speculationStarted: false,
          speculationUsed: false,
          speculationDiscarded: false,
          discardReason: null,
          displacedByPriority: false,
          firstSentenceRelMs: null,
          specStartRelMs: null,
          headStartMs: null,
          ttsRequestCount: 0,
          specElapsedMs: null,
          estimatedGpuMs: null,
          measuredGpuMs: null, // null salvo instrumentación real en el backend GPU
          secondSegmentFailed: false,
          partialFailure: false,
          interSegmentGapMs: null,
        },
        tts: {
          jobs: [],           // múltiples trabajos / segmentos / fallbacks correlacionados al MISMO turnId
          httpRequests: [],
          totalRequests: 0,
          fallbackUsed: false,
          cancelled: false,
          partialFailure: false,
          retries429Total: 0,
          backoffMsTotal: 0,
          firstByteRelMs: null,       // primer byte HTTP WAV/MP3 (NO es TTFA de streaming)
          firstValidPcmRelMs: null,   // primer PCM progresivo válido
          httpBodyEndRelMs: null,     // fin de cuerpo HTTP (NO garantiza fin exacto de GPU)
          audioDurationSec: null,
        },
        playback: {
          playbackId: null,
          mode: playbackMode || null, // "wav" | "stream" | "edge" | "system" | "discord"
          dispatchedRelMs: null,
          participants: 0,
          audibleParticipants: 0,
          allViewsMuted: false,
          multiViewAggregation: "first_audible_view",
          startSource: null,                // "first_audible_view" | "muted_only"
          startMethod: null,                // "worklet_first_render" | "html_audio_onplaying" | "webaudio_buffer_start" | "unknown"
          serverToPlaybackStartAckMs: null, // incluye retraso de retorno unidireccional navegador→servidor (no es RTT completo)
          serverToPlaybackEndAckMs: null,   // fin audible confirmado por vista (tts_done)
          estimatedPlaybackStartRelMs: null,
          lastAudibleStartAckRelMs: null,
          mutedPlaybackStartRelMs: null,
          duplicateStartAcks: 0,
          rejectedAcks: 0,
          playbackEndRelMs: null,
          audibleEndDiffFromHttpEndMs: null,
          views: [],
          underrunsTotal: 0,
          underrunTotalMs: 0,
          underrunRecoveries: 0,
          cancelled: false,
          disconnected: false,
          partial: false,
          result: null,
        },
        timeline: [], // intervalos { stage, id, startRelMs, endRelMs, meta } para visualización sin sumar solapes
        events: [],   // eventos acotados por maxEventsPerTurn
        errors: [],
        metrics: {},  // resumen derivado al cerrar
      };
      this.active.set(tid, tr);
      this.byMessageId.set(mid, tid);
      this._pushEvent(tr, "turn_received", { source: tr.source, platform: tr.platform, priority: tr.priority });
      return tid;
    }, null);
  }

  _rel(tr, monoMs = null) {
    const t = Number.isFinite(monoMs) ? monoMs : this.now();
    tr._lastTouchedMs = t;
    return Math.max(0, r2(t - tr._originMonoMs));
  }

  _pushEvent(tr, type, meta = null, monoMs = null) {
    if (!tr) return;
    const relMs = this._rel(tr, monoMs);
    if (tr.events.length >= this.maxEventsPerTurn) {
      this.droppedEvents += 1;
      return;
    }
    tr.events.push({ type, relMs, ...(meta ? { meta: sanitizeValue(meta) } : {}) });
  }

  _pushTimeline(tr, stage, startRelMs, endRelMs, meta = null) {
    if (!tr || startRelMs === null) return;
    tr.timeline.push({
      stage,
      startRelMs: r2(startRelMs),
      endRelMs: endRelMs !== null ? r2(endRelMs) : null,
      durationMs: endRelMs !== null ? nonNeg(startRelMs, endRelMs) : null,
      ...(meta ? { meta: sanitizeValue(meta) } : {}),
    });
  }

  getActive(turnId) { return turnId ? this.active.get(turnId) || null : null; }

  resolveTurnByUrl(url) {
    const tid = url ? this.byUrl.get(url) : null;
    return tid ? this.active.get(tid) || null : null;
  }

  resolveTurnByStreamId(streamId) {
    const tid = streamId != null ? this.byStreamId.get(Number(streamId)) : null;
    return tid ? this.active.get(tid) || null : null;
  }

  linkUrl(turnId, url) {
    if (!this.enabled || !turnId || !url) return;
    this._safe(() => {
      if (this.active.has(turnId)) this.byUrl.set(String(url), turnId);
    });
  }

  linkStream(turnId, streamId, url = null) {
    if (!this.enabled || !turnId || streamId == null) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      const sid = Number(streamId);
      this.byStreamId.set(sid, turnId);
      if (url) this.byUrl.set(String(url), turnId);
      if (!tr.ids.streamIds.includes(sid)) tr.ids.streamIds.push(sid);
    });
  }

  // ── Cola y recepción ──
  markEnqueued(turnId, { priority = null, holdMs = 0, monoMs = null } = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      const rel = this._rel(tr, monoMs);
      tr.queue.enqueuedRelMs = rel;
      if (priority !== null) tr.priority = Number(priority);
      if (holdMs > 0) tr.queue.holdUntilRelMs = r2(rel + holdMs);
      this._pushEvent(tr, "queue_enqueued", { priority: tr.priority, holdMs });
    });
  }

  markPrepareStart(turnId, { overlappedWithPlayback = false, monoMs = null } = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      const rel = this._rel(tr, monoMs);
      tr.queue.prepareStartRelMs = rel;
      tr.queue.prepareAheadOverlapped = Boolean(overlappedWithPlayback);
      const base = tr.queue.enqueuedRelMs ?? 0;
      tr.queue.conversationalQueueWaitMs = nonNeg(base, rel);
      this._pushEvent(tr, "prepare_start", { overlappedWithPlayback: tr.queue.prepareAheadOverlapped });
    });
  }

  markPrepareEnd(turnId, { monoMs = null } = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      const rel = this._rel(tr, monoMs);
      tr.queue.prepareEndRelMs = rel;
      this._pushTimeline(tr, "prepare", tr.queue.prepareStartRelMs ?? 0, rel, { overlappedWithPlayback: tr.queue.prepareAheadOverlapped });
      this._pushEvent(tr, "prepare_end");
    });
  }

  markQueueEvent(turnId, kind, { reason = null, priority = null, monoMs = null } = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      const rel = this._rel(tr, monoMs);
      if (kind === "expired") tr.queue.expired = true;
      if (kind === "cancelled") { tr.queue.cancelled = true; tr.queue.cancelReason = sanitizeReasonCode(reason, "cancelled"); }
      if (kind === "replaced") tr.queue.replaced = true;
      if (kind === "promoted" && priority !== null) tr.priority = Number(priority);
      this._pushEvent(tr, `queue_${kind}`, { reason: reason ? sanitizeReasonCode(reason) : null, priority });
      if (kind === "expired") this._closeTrace(tr, "expired", reason || "ttl_expired", monoMs);
      else if (kind === "cancelled") this._closeTrace(tr, "cancelled", reason || "queue_cancelled", monoMs);
      else if (kind === "replaced") this._closeTrace(tr, "discarded", reason || "replaced", monoMs);
    });
  }

  markPlaybackTurnStart(turnId, { monoMs = null } = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      const rel = this._rel(tr, monoMs);
      tr.queue.playbackTurnStartRelMs = rel;
      if (tr.queue.prepareEndRelMs !== null) {
        tr.queue.prepareAheadWaitMs = nonNeg(tr.queue.prepareEndRelMs, rel);
      }
      this._pushEvent(tr, "playback_turn_start", { prepareAheadWaitMs: tr.queue.prepareAheadWaitMs });
    });
  }

  markPlaybackTurnEnd(turnId, { monoMs = null } = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      const rel = this._rel(tr, monoMs);
      tr.queue.playbackTurnEndRelMs = rel;
      this._pushEvent(tr, "playback_turn_end");
    });
  }

  // ── LLM ──
  startLlmAttempt(turnId, { providerId = null, streamingCapability = "unknown", fallback = false, monoMs = null } = {}) {
    if (!this.enabled || !turnId) return null;
    return this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return null;
      const rel = this._rel(tr, monoMs);
      const llmJobId = this.nextId("llm");
      tr.ids.llmJobIds.push(llmJobId);
      if (tr.llm.startRelMs === null) tr.llm.startRelMs = rel;
      if (!tr.llm.providerRequested && providerId) tr.llm.providerRequested = providerId;
      if (providerId) { tr.llm.providerEffective = providerId; tr.effectiveProvider = providerId; }
      if (streamingCapability) tr.llm.streamingCapability = streamingCapability;
      if (fallback) tr.llm.fallbackUsed = true;
      if (tr.llm.attempts.length > 0) tr.llm.retries += 1;
      const att = {
        llmJobId,
        providerId: providerId || null,
        streamingCapability,
        fallback: Boolean(fallback),
        startRelMs: rel,
        firstEventRelMs: null,
        firstTextDeltaRelMs: null,
        firstSentenceCandidateRelMs: null,
        completeResponseRelMs: null,
        endRelMs: null,
        cancelled: false,
        errorCode: null,
      };
      tr.llm.attempts.push(att);
      this._pushEvent(tr, "llm_start", { llmJobId, providerId, streamingCapability, fallback });
      return llmJobId;
    }, null);
  }

  _findLlm(tr, llmJobId) {
    if (!tr) return null;
    if (llmJobId) return tr.llm.attempts.find(a => a.llmJobId === llmJobId) || null;
    return tr.llm.attempts.at(-1) || null;
  }

  markLlmEvent(turnId, llmJobId, kind, { monoMs = null, errorCode = null } = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      const att = this._findLlm(tr, llmJobId);
      if (!tr || !att) return;
      const rel = this._rel(tr, monoMs);
      if (kind === "first_event" && att.firstEventRelMs === null) {
        att.firstEventRelMs = rel;
        if (tr.llm.firstEventRelMs === null) {
          tr.llm.firstEventRelMs = rel;
          tr.llm.firstEventLatencyMs = nonNeg(tr.llm.startRelMs, rel);
        }
      } else if (kind === "first_text_delta") {
        // Proveedores full_only no inventan TTFT incremental: solo registrar si el proveedor emite deltas incrementales
        if (att.streamingCapability !== "full_only" && att.firstTextDeltaRelMs === null) {
          att.firstTextDeltaRelMs = rel;
          if (tr.llm.firstTextDeltaRelMs === null) {
            tr.llm.firstTextDeltaRelMs = rel;
            tr.llm.firstTextLatencyMs = nonNeg(tr.llm.startRelMs, rel);
          }
        }
      } else if (kind === "first_sentence" && att.firstSentenceCandidateRelMs === null) {
        att.firstSentenceCandidateRelMs = rel;
        if (tr.llm.firstSentenceCandidateRelMs === null) {
          tr.llm.firstSentenceCandidateRelMs = rel;
          tr.llm.firstSentenceLatencyMs = nonNeg(tr.llm.startRelMs, rel);
        }
        if (tr.overlap.firstSentenceRelMs === null) tr.overlap.firstSentenceRelMs = rel;
      } else if (kind === "complete") {
        att.completeResponseRelMs = rel;
        tr.llm.completeResponseRelMs = rel;
      } else if (kind === "end") {
        att.endRelMs = rel;
        tr.llm.endRelMs = rel;
        tr.llm.totalMs = nonNeg(tr.llm.startRelMs, rel);
        this._pushTimeline(tr, "llm", att.startRelMs, rel, { llmJobId: att.llmJobId, providerId: att.providerId });
      } else if (kind === "cancelled") {
        att.cancelled = true;
        att.endRelMs = rel;
        tr.llm.cancelled = true;
        tr.llm.endRelMs = rel;
        tr.llm.totalMs = nonNeg(tr.llm.startRelMs, rel);
        this._pushTimeline(tr, "llm", att.startRelMs, rel, { llmJobId: att.llmJobId, cancelled: true });
      } else if (kind === "error") {
        att.errorCode = sanitizeReasonCode(errorCode, "llm_error");
        att.endRelMs = rel;
        tr.llm.errorCode = att.errorCode;
        tr.errors.push({ stage: "llm", code: att.errorCode, relMs: rel });
      }
      this._pushEvent(tr, `llm_${kind}`, { llmJobId: att.llmJobId, errorCode: att.errorCode });
    });
  }

  recordLlmFallback(turnId, { effectiveProvider = null, fallbackUsed = true, monoMs = null } = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      if (fallbackUsed) tr.llm.fallbackUsed = true;
      if (effectiveProvider) {
        tr.llm.providerEffective = String(effectiveProvider);
        tr.effectiveProvider = String(effectiveProvider);
      }
      this._pushEvent(tr, "llm_fallback", { effectiveProvider: tr.llm.providerEffective, fallbackUsed: tr.llm.fallbackUsed }, monoMs);
    });
  }

  // ── Validaciones post-LLM (sin texto privado ni re-ejecución de filtros) ──
  recordValidations(turnId, {
    startRelMs = null,
    endRelMs = null,
    sanitizeMs = null,
    filtersMs = null,
    responseValidationMs = null,
    emotionExtractionMs = null,
    phoneticsMs = null,
    ttsTransformMs = null,
    status = "ok",
    reasonCode = null,
  } = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      const v = tr.validations;
      if (startRelMs !== null) v.startRelMs = r2(startRelMs);
      if (endRelMs !== null) v.endRelMs = r2(endRelMs);
      if (sanitizeMs !== null) v.sanitizeMs = r2(sanitizeMs);
      if (filtersMs !== null) v.filtersMs = r2(filtersMs);
      if (responseValidationMs !== null) v.responseValidationMs = r2(responseValidationMs);
      if (emotionExtractionMs !== null) v.emotionExtractionMs = r2(emotionExtractionMs);
      if (phoneticsMs !== null) v.phoneticsMs = r2(phoneticsMs);
      if (ttsTransformMs !== null) v.ttsTransformMs = r2(ttsTransformMs);
      v.status = sanitizeReasonCode(status, "ok");
      v.reasonCode = reasonCode ? sanitizeReasonCode(reasonCode) : null;
      if (v.startRelMs !== null && v.endRelMs !== null) {
        v.totalMs = nonNeg(v.startRelMs, v.endRelMs);
        tr.llm.validationMs = v.totalMs;
        this._pushTimeline(tr, "validations", v.startRelMs, v.endRelMs, { status: v.status });
      } else {
        const parts = [v.sanitizeMs, v.filtersMs, v.responseValidationMs, v.emotionExtractionMs, v.phoneticsMs, v.ttsTransformMs].filter(Number.isFinite);
        if (parts.length) {
          v.totalMs = r2(parts.reduce((a, b) => a + b, 0));
          tr.llm.validationMs = v.totalMs;
        }
      }
      this._pushEvent(tr, "validations_recorded", { status: v.status, reasonCode: v.reasonCode, totalMs: v.totalMs });
    });
  }

  // ── Overlap ──
  recordOverlap(turnId, patch = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      const o = tr.overlap;
      if (patch.enabled !== undefined) o.enabled = Boolean(patch.enabled);
      if (patch.providerEligible !== undefined) o.providerEligible = Boolean(patch.providerEligible);
      if (patch.backendAvailable !== undefined) o.backendAvailable = patch.backendAvailable === null ? null : Boolean(patch.backendAvailable);
      if (patch.speculationStarted) {
        o.speculationStarted = true;
        if (o.specStartRelMs === null) o.specStartRelMs = this._rel(tr, patch.monoMs);
      }
      if (patch.speculationUsed) o.speculationUsed = true;
      if (patch.speculationDiscarded) o.speculationDiscarded = true;
      if (patch.discardReason !== undefined) o.discardReason = patch.discardReason ? sanitizeReasonCode(patch.discardReason) : null;
      if (patch.displacedByPriority) o.displacedByPriority = true;
      if (Number.isFinite(patch.headStartMs)) o.headStartMs = r2(patch.headStartMs);
      if (Number.isFinite(patch.ttsRequestCount)) o.ttsRequestCount = Math.max(0, Math.round(patch.ttsRequestCount));
      if (Number.isFinite(patch.specElapsedMs)) o.specElapsedMs = r2(patch.specElapsedMs);
      if (Number.isFinite(patch.estimatedGpuMs)) o.estimatedGpuMs = r2(patch.estimatedGpuMs);
      // measuredGpuMs solo si viene explícitamente medido del backend (por defecto null)
      if (patch.measuredGpuMs !== undefined) o.measuredGpuMs = Number.isFinite(patch.measuredGpuMs) ? r2(patch.measuredGpuMs) : null;
      if (patch.secondSegmentFailed) o.secondSegmentFailed = true;
      if (patch.partialFailure) o.partialFailure = true;
      if (Number.isFinite(patch.interSegmentGapMs)) o.interSegmentGapMs = r2(patch.interSegmentGapMs);
      this._pushEvent(tr, "overlap_update", {
        started: o.speculationStarted, used: o.speculationUsed, discarded: o.speculationDiscarded,
        reason: o.discardReason, displaced: o.displacedByPriority,
      }, patch.monoMs);
    });
  }

  // ── TTS y solicitudes HTTP ──
  startTtsJob(turnId, {
    requestedEngine = null,
    effectiveEngine = null,
    playbackMode = "wav",
    segmentIndex = 0,
    speculative = false,
    fallback = false,
    streamId = null,
    monoMs = null,
  } = {}) {
    if (!this.enabled || !turnId) return null;
    return this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return null;
      const rel = this._rel(tr, monoMs);
      const ttsJobId = this.nextId("tts");
      const httpRequestId = this.nextId("req");
      const segmentId = this.nextId("seg");
      tr.ids.ttsJobIds.push(ttsJobId);
      tr.ids.httpRequestIds.push(httpRequestId);
      tr.ids.segmentIds.push(segmentId);
      if (streamId != null) this.linkStream(turnId, streamId);
      if (requestedEngine && !tr.requestedEngine) tr.requestedEngine = requestedEngine;
      if (effectiveEngine) tr.effectiveEngine = effectiveEngine;
      if (playbackMode) tr.playbackMode = playbackMode;
      const job = {
        ttsJobId,
        httpRequestId,
        streamId: streamId != null ? Number(streamId) : null,
        segmentId,
        segmentIndex,
        speculative: Boolean(speculative),
        fallback: Boolean(fallback),
        requestedEngine: requestedEngine || tr.requestedEngine || null,
        effectiveEngine: effectiveEngine || requestedEngine || null,
        playbackMode,
        queueStartRelMs: rel,
        requestStartRelMs: null,
        headersRelMs: null,
        firstByteRelMs: null,      // WAV/MP3 primer byte HTTP (NO llamado TTFA)
        firstValidPcmRelMs: null,  // PCM progresivo primer bloque válido
        httpBodyEndRelMs: null,    // fin del cuerpo HTTP
        remoteGenEndRelMs: null,   // solo si existe señal fiable del backend
        queueWaitMs: null,
        httpTotalMs: null,
        audioDurationSec: null,
        bytesReceived: 0,
        samplesReceived: 0,
        chunkCount: 0,
        retries429: 0,
        backoffMs: 0,
        fallbackUsed: Boolean(fallback),
        cancelled: false,
        partialFailure: false,
        errorCode: null,
        httpTotalRtf: null,
        result: "pending",
      };
      tr.tts.jobs.push(job);
      tr.tts.httpRequests.push({ httpRequestId, ttsJobId, segmentId, engine: job.effectiveEngine, playbackMode });
      tr.tts.totalRequests = tr.tts.jobs.length;
      tr.overlap.ttsRequestCount = tr.tts.totalRequests;
      if (fallback) tr.tts.fallbackUsed = true;
      this._pushEvent(tr, "tts_job_start", { ttsJobId, httpRequestId, segmentId, segmentIndex, engine: job.effectiveEngine, playbackMode, speculative, fallback }, monoMs);
      return { ttsJobId, httpRequestId, segmentId };
    }, null);
  }

  _findTtsJob(tr, ttsJobId) {
    if (!tr) return null;
    if (ttsJobId) return tr.tts.jobs.find(j => j.ttsJobId === ttsJobId) || null;
    return tr.tts.jobs.at(-1) || null;
  }

  updateTtsJob(turnId, ttsJobId, patch = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      const job = this._findTtsJob(tr, ttsJobId);
      if (!tr || !job) return;
      const rel = this._rel(tr, patch.monoMs);
      if (patch.streamId != null) {
        job.streamId = Number(patch.streamId);
        this.linkStream(turnId, job.streamId, patch.url || null);
      } else if (patch.url) {
        this.linkUrl(turnId, patch.url);
      }
      if (patch.effectiveEngine) { job.effectiveEngine = patch.effectiveEngine; tr.effectiveEngine = patch.effectiveEngine; }
      if (patch.playbackMode) { job.playbackMode = patch.playbackMode; tr.playbackMode = patch.playbackMode; }
      if (patch.requestStart && job.requestStartRelMs === null) {
        job.requestStartRelMs = rel;
        job.queueWaitMs = Number.isFinite(patch.queueWaitMs) ? r2(patch.queueWaitMs) : nonNeg(job.queueStartRelMs, rel);
      }
      if (patch.headers && job.headersRelMs === null) job.headersRelMs = rel;
      if (patch.firstByte && job.firstByteRelMs === null) {
        job.firstByteRelMs = rel;
        if (tr.tts.firstByteRelMs === null) tr.tts.firstByteRelMs = rel;
      }
      if (patch.firstValidPcm && job.firstValidPcmRelMs === null) {
        job.firstValidPcmRelMs = rel;
        if (tr.tts.firstValidPcmRelMs === null) tr.tts.firstValidPcmRelMs = rel;
      }
      if (Number.isFinite(patch.queueWaitMs)) job.queueWaitMs = r2(patch.queueWaitMs);
      if (Number.isFinite(patch.bytesReceived)) job.bytesReceived = Math.max(0, Math.round(patch.bytesReceived));
      if (Number.isFinite(patch.samplesReceived)) job.samplesReceived = Math.max(0, Math.round(patch.samplesReceived));
      if (Number.isFinite(patch.chunkCount)) job.chunkCount = Math.max(0, Math.round(patch.chunkCount));
      if (Number.isFinite(patch.retries429)) {
        job.retries429 = Math.max(0, Math.round(patch.retries429));
        tr.tts.retries429Total = tr.tts.jobs.reduce((s, j) => s + j.retries429, 0);
      }
      if (Number.isFinite(patch.backoffMs)) {
        job.backoffMs = Math.max(0, r2(patch.backoffMs));
        tr.tts.backoffMsTotal = r2(tr.tts.jobs.reduce((s, j) => s + j.backoffMs, 0));
      }
      if (Number.isFinite(patch.audioDurationSec)) {
        job.audioDurationSec = r3(patch.audioDurationSec);
        const sumDur = tr.tts.jobs.filter(j => j.result !== "cancelled" && Number.isFinite(j.audioDurationSec)).reduce((s, j) => s + j.audioDurationSec, 0);
        if (sumDur > 0) tr.tts.audioDurationSec = r3(sumDur);
      }
      if (patch.remoteGenEndRelMs !== undefined) {
        job.remoteGenEndRelMs = Number.isFinite(patch.remoteGenEndRelMs) ? r2(patch.remoteGenEndRelMs) : null;
      }
      if (patch.httpBodyEnd) {
        job.httpBodyEndRelMs = rel;
        tr.tts.httpBodyEndRelMs = rel;
        const startBase = job.requestStartRelMs ?? job.queueStartRelMs;
        job.httpTotalMs = Number.isFinite(patch.httpTotalMs) ? r2(patch.httpTotalMs) : nonNeg(startBase, rel);
        if (job.audioDurationSec > 0 && job.httpTotalMs !== null) {
          job.httpTotalRtf = r3(job.httpTotalMs / (job.audioDurationSec * 1000));
        }
        job.result = patch.result || "ok";
        this._pushTimeline(tr, job.speculative ? "tts_speculative" : "tts", startBase, rel, {
          ttsJobId: job.ttsJobId, segmentIndex: job.segmentIndex, engine: job.effectiveEngine, mode: job.playbackMode,
        });
      }
      if (patch.cancelled) {
        job.cancelled = true;
        job.result = "cancelled";
        tr.tts.cancelled = true;
        this._pushTimeline(tr, "tts_cancelled", job.requestStartRelMs ?? job.queueStartRelMs, rel, { ttsJobId: job.ttsJobId });
      }
      if (patch.partialFailure) {
        job.partialFailure = true;
        job.result = "partial";
        tr.tts.partialFailure = true;
        tr.overlap.partialFailure = true;
        if (job.segmentIndex > 0) tr.overlap.secondSegmentFailed = true;
      }
      if (patch.errorCode) {
        job.errorCode = sanitizeReasonCode(patch.errorCode, "tts_error");
        if (job.result === "pending") job.result = job.partialFailure ? "partial" : "error";
        tr.errors.push({ stage: "tts", ttsJobId: job.ttsJobId, code: job.errorCode, relMs: rel });
      }
      if (patch.fallbackUsed) {
        job.fallbackUsed = true;
        tr.tts.fallbackUsed = true;
      }
    });
  }

  // ── Reproducción y navegador / AudioWorklet ──
  markPlaybackDispatched(turnId, { url = null, mode = "wav", participants = 0, audibleParticipants = null, monoMs = null } = {}) {
    if (!this.enabled || !turnId) return null;
    return this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return null;
      const rel = this._rel(tr, monoMs);
      const playbackId = this.nextId("ply");
      tr.ids.playbackIds.push(playbackId);
      tr.playback.playbackId = playbackId;
      tr.playback.mode = mode;
      tr.playback.dispatchedRelMs = rel;
      const pCount = Math.max(0, Math.round(Number(participants) || 0));
      const aCount = audibleParticipants === null || audibleParticipants === undefined ? pCount : Math.max(0, Math.round(Number(audibleParticipants) || 0));
      tr.playback.participants = pCount;
      tr.playback.audibleParticipants = aCount;
      tr.playback.allViewsMuted = pCount > 0 && aCount === 0;
      if (url) this.linkUrl(turnId, url);
      this._pushEvent(tr, "playback_dispatched", { playbackId, mode, participants: pCount, audibleParticipants: aCount }, monoMs);
      return playbackId;
    }, null);
  }

  // ACK recibido en el servidor cuando una vista reporta primer renderizado efectivo (AudioWorklet started, <audio> onplaying o AudioBufferSourceNode.start)
  recordPlaybackStartAck(turnId, {
    viewId = "view_1",
    audible = true,
    mode = null,
    startMethod = "unknown",
    clientRenderStartMs = null,
    bufferAtStartSec = null,
    contextRate = null,
    contextState = null,
    outputLatencyMs = null,
    monoMs = null,
  } = {}) {
    if (!this.enabled || !turnId) return false;
    return this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return false;
      const rel = this._rel(tr, monoMs);
      const pb = tr.playback;
      if (pb.cancelled || pb.playbackEndRelMs !== null) {
        pb.rejectedAcks += 1;
        return false;
      }
      if (mode && !pb.mode) pb.mode = sanitizeReasonCode(mode, "unknown");
      const safeViewId = sanitizeReasonCode(viewId, "view_1");
      const safeMethod = VALID_START_METHODS.has(startMethod) ? startMethod : "unknown";
      const isAudible = Boolean(audible);
      const clampNum = (v, max = 1e7) => (Number.isFinite(Number(v)) ? Math.max(0, Math.min(max, Number(v))) : null);

      let view = pb.views.find(v => v.viewId === safeViewId);
      if (view && view.serverStartAckRelMs != null) {
        // ACK duplicado de la misma vista: se contabiliza sin sobrescribir
        pb.duplicateStartAcks += 1;
        return false;
      }
      if (!view) {
        view = { viewId: safeViewId, audible: isAudible, startMethod: safeMethod, clientClock: {} };
        pb.views.push(view);
      } else {
        view.audible = isAudible;
        view.startMethod = safeMethod;
      }
      view.serverStartAckRelMs = rel;
      view.clientClock = {
        ...view.clientClock,
        effectiveRenderStartMs: clampNum(clientRenderStartMs),
        bufferAtStartSec: clampNum(bufferAtStartSec, 600),
        contextRate: clampNum(contextRate, 384000),
        contextState: ["running", "suspended", "closed"].includes(contextState) ? contextState : (view.clientClock.contextState ?? null),
        outputLatencyMs: clampNum(outputLatencyMs, 10000),
      };

      if (isAudible) {
        pb.lastAudibleStartAckRelMs = rel;
        // La métrica principal de inicio audible se fija con la primera vista audible (o reemplaza un provisional muted_only)
        if (pb.estimatedPlaybackStartRelMs === null || pb.startSource === "muted_only") {
          pb.estimatedPlaybackStartRelMs = rel;
          pb.serverToPlaybackStartAckMs = pb.dispatchedRelMs !== null ? nonNeg(pb.dispatchedRelMs, rel) : rel;
          pb.startSource = "first_audible_view";
          pb.startMethod = safeMethod;
          pb.allViewsMuted = false;
        }
      } else {
        if (pb.mutedPlaybackStartRelMs === null) pb.mutedPlaybackStartRelMs = rel;
        // Una vista silenciada nunca falsea la métrica principal si hay vistas audibles; solo se usa cuando TODAS están silenciadas
        if (pb.estimatedPlaybackStartRelMs === null && pb.audibleParticipants === 0) {
          pb.estimatedPlaybackStartRelMs = rel;
          pb.serverToPlaybackStartAckMs = pb.dispatchedRelMs !== null ? nonNeg(pb.dispatchedRelMs, rel) : rel;
          pb.startSource = "muted_only";
          pb.startMethod = safeMethod;
          pb.allViewsMuted = true;
        }
      }
      this._pushEvent(tr, "playback_start_ack", {
        viewId: safeViewId, audible: isAudible, mode: pb.mode, startMethod: safeMethod,
        startSource: pb.startSource, serverToAckMs: pb.serverToPlaybackStartAckMs,
      }, monoMs);
      return true;
    }, false);
  }

  recordClientPlaybackStats(turnId, { viewId = "view_1", audible = true, stats = {}, monoMs = null } = {}) {
    if (!this.enabled || !turnId) return false;
    return this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr || !stats || typeof stats !== "object" || Array.isArray(stats)) return false;
      const pb = tr.playback;
      const safeViewId = sanitizeReasonCode(viewId, "view_1");
      let view = pb.views.find(v => v.viewId === safeViewId);
      if (!view) {
        view = { viewId: safeViewId, audible: Boolean(audible), startMethod: "unknown", clientClock: {} };
        pb.views.push(view);
      }
      // Lista explícita de campos permitidos desde el navegador (se descarta cualquier campo arbitrario)
      const num = (v, max = 1e7) => (Number.isFinite(Number(v)) ? Math.max(0, Math.min(max, Number(v))) : null);
      view.clientClock = {
        ...view.clientClock,
        firstChunkMs: num(stats.firstChunkMs),
        bufferingStartMs: num(stats.bufferingStartMs),
        effectiveRenderStartMs: num(stats.effectiveRenderStartMs ?? stats.audibleStartMs),
        bufferAtStartSec: num(stats.bufferAtStartSec, 600),
        contextRate: num(stats.contextRate, 384000),
        contextState: ["running", "suspended", "closed"].includes(stats.contextState) ? stats.contextState : (view.clientClock.contextState ?? null),
        outputLatencyMs: num(stats.outputLatencyMs, 10000),
        receivedSamples: num(stats.receivedSamples, 2e8),
        playedSamples: num(stats.playedSamples, 2e8),
        underruns: num(stats.underruns, 1000) ?? 0,
        underrunMs: num(stats.underrunMs) ?? 0,
        underrunRecoveries: num(stats.underrunRecoveries, 1000) ?? 0,
        streamEndMs: num(stats.streamEndMs),
        playbackEndMs: num(stats.playbackEndMs),
        decodedDurationSec: num(stats.decodedDurationSec, 3600),
        result: ["done", "cancelled", "error", "disconnected"].includes(stats.result) ? stats.result : null,
      };
      const relevant = pb.views.filter(v => v.audible);
      const pool = relevant.length ? relevant : pb.views;
      pb.underrunsTotal = Math.max(0, ...pool.map(v => v.clientClock.underruns || 0));
      pb.underrunTotalMs = Math.max(0, ...pool.map(v => v.clientClock.underrunMs || 0));
      pb.underrunRecoveries = Math.max(0, ...pool.map(v => v.clientClock.underrunRecoveries || 0));
      if (stats.result === "cancelled") pb.cancelled = true;
      if (stats.result === "disconnected") pb.disconnected = true;
      this._pushEvent(tr, "playback_client_stats", { viewId: safeViewId, underruns: view.clientClock.underruns, result: view.clientClock.result }, monoMs);
      return true;
    }, false);
  }

  markPlaybackEnd(turnId, { result = "done", partial = false, disconnected = false, monoMs = null } = {}) {
    if (!this.enabled || !turnId) return;
    this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return;
      const rel = this._rel(tr, monoMs);
      const pb = tr.playback;
      pb.playbackEndRelMs = rel;
      pb.serverToPlaybackEndAckMs = pb.dispatchedRelMs !== null ? nonNeg(pb.dispatchedRelMs, rel) : rel;
      pb.result = sanitizeReasonCode(result, "done");
      pb.partial = Boolean(partial);
      if (result === "cancelled") pb.cancelled = true;
      if (disconnected) pb.disconnected = true;
      if (tr.tts.httpBodyEndRelMs !== null) {
        pb.audibleEndDiffFromHttpEndMs = r2(rel - tr.tts.httpBodyEndRelMs);
      }
      const startLine = pb.estimatedPlaybackStartRelMs ?? pb.dispatchedRelMs;
      if (startLine !== null) {
        this._pushTimeline(tr, "playback", startLine, rel, { mode: pb.mode, result: pb.result, partial: pb.partial });
      }
      this._pushEvent(tr, "playback_end", { result: pb.result, partial: pb.partial, audibleEndDiffFromHttpEndMs: pb.audibleEndDiffFromHttpEndMs }, monoMs);
    });
  }

  // ── Cierre terminal ──
  finishTurn(turnId, { state = "completed", reason = null, monoMs = null } = {}) {
    if (!this.enabled || !turnId) return null;
    return this._safe(() => {
      const tr = this.active.get(turnId);
      if (!tr) return null;
      return this._closeTrace(tr, state, reason, monoMs);
    }, null);
  }

  _closeTrace(tr, state, reason = null, monoMs = null) {
    const finalState = TERMINAL_STATES.has(state) ? state : "completed";
    const endRel = this._rel(tr, monoMs);
    tr.state = finalState;
    tr.terminalReason = reason ? sanitizeReasonCode(reason) : null;
    tr.endedAtIso = new Date(this.wallNow()).toISOString();
    // Métricas derivadas de la línea temporal del propio turno (sin sumar etapas superpuestas)
    tr.metrics = {
      turnTotalMs: endRel,
      queueWaitMs: tr.queue.conversationalQueueWaitMs,
      prepareAheadWaitMs: tr.queue.prepareAheadWaitMs,
      llmFirstTextMs: tr.llm.firstTextLatencyMs,
      llmCompleteMs: tr.llm.totalMs,
      validationsMs: tr.validations.totalMs,
      firstByteRelMs: tr.tts.firstByteRelMs,
      firstPcmRelMs: tr.tts.firstValidPcmRelMs,
      httpBodyEndRelMs: tr.tts.httpBodyEndRelMs,
      estimatedPlaybackStartRelMs: tr.playback.estimatedPlaybackStartRelMs,
      playbackEndRelMs: tr.playback.playbackEndRelMs,
      audibleEndDiffFromHttpEndMs: tr.playback.audibleEndDiffFromHttpEndMs,
      audioDurationSec: tr.tts.audioDurationSec,
      httpTotalRtf: tr.tts.audioDurationSec > 0 && tr.tts.jobs.length
        ? r3(tr.tts.jobs.reduce((s, j) => s + (j.httpTotalMs || 0), 0) / (tr.tts.audioDurationSec * 1000))
        : null,
      ttsRequestsPerTurn: tr.tts.totalRequests,
      retries429: tr.tts.retries429Total,
      backoffMs: tr.tts.backoffMsTotal,
      underruns: tr.playback.underrunsTotal,
      underrunTotalMs: tr.playback.underrunTotalMs,
      fallbackUsed: Boolean(tr.tts.fallbackUsed || tr.llm.fallbackUsed),
      partialFailure: Boolean(tr.tts.partialFailure || tr.playback.partial),
      speculationStarted: tr.overlap.speculationStarted,
      speculationUsed: tr.overlap.speculationUsed,
      speculationDiscarded: tr.overlap.speculationDiscarded,
    };

    // Limpiar índices activos
    this.active.delete(tr.ids.turnId);
    if (this.byMessageId.get(tr.ids.messageId) === tr.ids.turnId) this.byMessageId.delete(tr.ids.messageId);
    for (const sid of tr.ids.streamIds) if (this.byStreamId.get(sid) === tr.ids.turnId) this.byStreamId.delete(sid);
    for (const [u, tid] of [...this.byUrl.entries()]) if (tid === tr.ids.turnId) this.byUrl.delete(u);

    // Remover campos internos de reloj absoluto antes de guardar en historial
    const { _originMonoMs, _lastTouchedMs, ...exported } = tr;
    this.history.push(exported);
    while (this.history.length > this.maxHistory) this.history.shift();
    return exported;
  }

  getHistory({ category = null, limit = null } = {}) {
    return this._safe(() => {
      let list = category ? this.history.filter(t => t.category === category) : [...this.history];
      if (Number.isFinite(limit) && limit > 0) list = list.slice(-limit);
      return list;
    }, []);
  }

  getSummary() {
    return this._safe(() => ({
      schemaVersion: TRACE_SCHEMA_VERSION,
      enabled: this.enabled,
      category: this.category,
      activeCount: this.active.size,
      historyCount: this.history.length,
      maxHistory: this.maxHistory,
      maxActive: this.maxActive,
      maxEventsPerTurn: this.maxEventsPerTurn,
      droppedEvents: this.droppedEvents,
      evictedActive: this.evictedActive,
      internalErrors: this.internalErrors,
    }), { enabled: false, internalErrors: this.internalErrors });
  }

  clear() {
    this.active.clear();
    this.history = [];
    this.byUrl.clear();
    this.byStreamId.clear();
    this.byMessageId.clear();
    this.droppedEvents = 0;
    this.evictedActive = 0;
  }
}
