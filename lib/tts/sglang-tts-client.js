// Fish S2 Pro servido por SGLang-Omni detrás de un gateway autenticado (notebook Nyra_Fish_S2_Pro_SGLang_L4_Operativo).
// Contrato del gateway:
//   GET  /v1/health        → JSON {status:"ok", model, voice, busy, sample_rate:44100, channels:1, encoding:"pcm_s16le"}
//   POST /v1/audio/speech  ← JSON {model, voice, input, stream, response_format}  (nada más; voz y sampling viven en el server)
//        stream:false, response_format:"wav" → 200 audio/wav (PCM 16 bits, mono, 44100 Hz, Cache-Control: no-store)
//        429 + Retry-After cuando está ocupado (1 síntesis a la vez)
// Fase 1: solo WAV completo. stream:true (application/octet-stream PCM con X-Audio-*) queda para Fase 2.
import { lookup as dnsLookup } from "node:dns/promises";
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { applyFishExpressivity, assertSafeFishTarget, pinnedRequest, validateFishUrl, validateWavBuffer } from "./fish-audio-client.js";

export const SGLANG_MODEL = "fishaudio/s2-pro";
export const SGLANG_VOICE = "nyra-clonada";
export const SGLANG_MAX_INPUT_CHARS = 1000;
const MAX_WAV_BYTES = 25 * 1024 * 1024;
const MAX_JSON_BYTES = 64 * 1024;
const MAX_ATTEMPTS = 3;
const MAX_RETRY_WAIT_MS = 3000;
const WAV_MIME = "audio/wav"; // único Content-Type que emite el gateway para stream:false
const WAV_CONTRACT = { audioFormat: 1, channels: 1, sampleRate: 44100, bitsPerSample: 16 };
const SERVICE = { serviceName: "SGLang TTS", urlVar: "SGLANG_TTS_URL" };

const isLoopback = h => ["localhost", "127.0.0.1", "::1"].includes(String(h).toLowerCase().replace(/^\[|\]$/g, ""));
const tagged = (msg, extra) => Object.assign(new Error(msg), extra);
const mime = headers => String(headers?.["content-type"] || "").split(";")[0].trim().toLowerCase();

// Reglas propias encima de las de Fish: HTTPS obligatorio salvo loopback autorizado, y lista de hosts obligatoria si es remoto.
export function validateSglangUrl(rawUrl, { allowedHosts = null, allowLoopbackHttp = false } = {}) {
  const s = String(rawUrl || "").trim();
  if (!s) throw tagged("SGLANG_TTS_URL no configurada", { code: "UNCONFIGURED" });
  let u;
  try { u = new URL(s); } catch { throw new Error("URL de SGLang TTS inválida"); }
  const loop = isLoopback(u.hostname);
  if (loop && !allowLoopbackHttp) throw new Error("Loopback no autorizado para SGLang TTS (SGLANG_TTS_ALLOW_LOOPBACK_HTTP=1 solo en pruebas)");
  if (u.protocol === "http:" && !loop) throw new Error("SGLang TTS remoto exige HTTPS");
  if (!loop && !(Array.isArray(allowedHosts) && allowedHosts.some(h => String(h).trim()))) {
    throw new Error("SGLANG_TTS_ALLOWED_HOSTS vacío: un endpoint remoto debe estar en la lista permitida");
  }
  if (u.pathname !== "/" || u.search || u.hash) throw new Error("La URL de SGLang TTS debe ser solo el origen (sin ruta)");
  validateFishUrl(s, { allowedHosts, ...SERVICE }); // lista de hosts, credenciales embebidas e IP privada literal
  return u;
}

// El gateway garantiza PCM16 mono 44100: cualquier otra cosa es una respuesta corrupta o de otro servicio.
// (Más estricto que validateWavBuffer a propósito; Fish sigue aceptando sus formatos.)
export function validateSglangWav(rawBuf) {
  const wav = validateWavBuffer(rawBuf); // RIFF/WAVE, chunks fmt/data, tamaño máximo
  const bad = msg => tagged(`WAV inválido: ${msg}`, { code: "BAD_WAV" });
  for (const [k, v] of Object.entries(WAV_CONTRACT)) if (wav[k] !== v) throw bad(`${k}=${wav[k]} (se espera ${v})`);
  if (wav.blockAlign !== 2 || wav.byteRate !== 88200) throw bad("blockAlign/byteRate inconsistentes");
  if (wav.truncated || wav.declaredDataSize !== wav.dataSize) throw bad("truncado (faltan muestras)");
  if (wav.dataSize <= 0) throw bad("sin audio");
  if (wav.dataSize % wav.blockAlign !== 0) throw bad("muestras desalineadas");
  if (wav.riffSize + 8 !== wav.buf.length) throw bad("tamaño RIFF inconsistente");
  return wav;
}

export class SGLangTTSClient {
  constructor({ config = {}, settingsFile, allowedHosts = null, allowLoopbackHttp = false, lookupImpl = dnsLookup, maxResponseBytes = MAX_WAV_BYTES, logger = console } = {}) {
    this.settingsFile = resolve(settingsFile || join(process.cwd(), "tts_settings.json"));
    this.allowedHosts = allowedHosts;
    this.allowLoopbackHttp = Boolean(allowLoopbackHttp);
    this.lookupImpl = lookupImpl;
    this.maxResponseBytes = maxResponseBytes;
    this.logger = logger;
    const saved = this._loadSaved().sglang || {};
    const url = String(saved.url || config.url || "").trim();
    this.config = {
      url,
      apiKey: String(config.apiKey || "").trim(), // solo desde .env; nunca se persiste ni se devuelve
      timeoutMs: clampInt(saved.timeoutMs ?? config.timeoutMs, 5000, 180000, 60000),
      expressive: (saved.expressive ?? config.expressive) !== false,
      fallback: (saved.fallback ?? config.fallback) !== false,
    };
    this._active = new Map();
    this.health = null;
    this.metrics = { status: "unknown", failures: 0, cancellations: 0, fallbacks: 0, lastError: null, lastCheckedAt: null, last: null };
  }

  _loadSaved() {
    try {
      if (!existsSync(this.settingsFile)) return {};
      const p = JSON.parse(readFileSync(this.settingsFile, "utf8"));
      return p && typeof p === "object" ? p : {};
    } catch { return {}; }
  }

  _persist() {
    const { apiKey: _omit, ...pub } = this.config;
    const tmp = `${this.settingsFile}.${process.pid}.${Date.now()}.tmp`;
    mkdirSync(resolve(this.settingsFile, ".."), { recursive: true });
    writeFileSync(tmp, JSON.stringify({ ...this._loadSaved(), sglang: pub }, null, 2));
    renameSync(tmp, this.settingsFile);
  }

  isConfigured() { return Boolean(this.config.url && this.config.apiKey); }

  updateConfig(patch = {}) {
    if (patch.url !== undefined) {
      const u = String(patch.url || "").trim();
      this.config.url = u ? validateSglangUrl(u, this._urlOpts()).origin : "";
    }
    if (patch.timeoutMs !== undefined) this.config.timeoutMs = clampInt(patch.timeoutMs, 5000, 180000, this.config.timeoutMs);
    if (patch.expressive !== undefined) this.config.expressive = Boolean(patch.expressive);
    if (patch.fallback !== undefined) this.config.fallback = Boolean(patch.fallback);
    this._persist();
    return this.getPublicConfig();
  }

  getPublicConfig() {
    return {
      url: this.config.url,
      apiKeyConfigured: Boolean(this.config.apiKey),
      configured: this.isConfigured(),
      timeoutMs: this.config.timeoutMs,
      expressive: this.config.expressive,
      fallback: this.config.fallback,
      streaming: false, // Fase 2
      health: this.health,
      metrics: { ...this.metrics, last: this.metrics.last && { ...this.metrics.last } },
    };
  }

  _urlOpts() { return { allowedHosts: this.allowedHosts, allowLoopbackHttp: this.allowLoopbackHttp }; }

  async _target() {
    validateSglangUrl(this.config.url, this._urlOpts());
    return assertSafeFishTarget(this.config.url, { lookupImpl: this.lookupImpl, allowedHosts: this.allowedHosts, ...SERVICE });
  }

  _requireKey() {
    if (!this.config.apiKey) throw tagged("SGLANG_TTS_API_KEY no configurada", { code: "UNCONFIGURED" });
  }

  abortSource(source) {
    const st = this._active.get(source);
    if (!st) return false;
    st.cancelled = true;
    st.controller.abort(new Error("Cancelación manual de TTS"));
    return true;
  }

  abortAll() {
    let any = false;
    for (const s of [...this._active.keys()]) any = this.abortSource(s) || any;
    return any;
  }

  recordFallback() { this.metrics.fallbacks += 1; }

  // Nunca lanza: devuelve un estado clasificado (online | busy | unauthorized | unavailable | timeout | invalid_response | unconfigured | offline).
  async checkHealth() {
    const done = (status, extra = {}) => {
      this.health = { status, checkedAt: Date.now(), ...extra };
      this.metrics.status = status;
      this.metrics.lastCheckedAt = this.health.checkedAt;
      this.metrics.lastError = extra.error || null;
      return { ok: status === "online" || status === "busy", ...this.health };
    };
    let target;
    try { this._requireKey(); target = await this._target(); }
    catch (err) { return done(err.code === "UNCONFIGURED" ? "unconfigured" : "invalid_config", { error: err.message }); }
    const ac = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; ac.abort(new Error("timeout")); }, Math.min(10000, this.config.timeoutMs));
    const t0 = performance.now();
    try {
      const res = await pinnedRequest(target, "/v1/health", {
        method: "GET", headers: { Authorization: `Bearer ${this.config.apiKey}` }, signal: ac.signal, maxBytes: MAX_JSON_BYTES,
      });
      const headersMs = Math.round(performance.now() - t0);
      if (!res.ok) res.arrayBuffer().catch(() => {});
      if (res.status === 401 || res.status === 403) return done("unauthorized", { error: `HTTP ${res.status}`, headersMs });
      if (res.status === 429) return done("busy", { headersMs });
      if (!res.ok) return done(res.status >= 500 ? "unavailable" : "invalid_response", { error: `HTTP ${res.status}`, headersMs });
      if (mime(res.headersRaw) !== "application/json") return done("invalid_response", { error: "Content-Type inesperado", headersMs });
      let body;
      try { body = JSON.parse(Buffer.from(await res.arrayBuffer()).toString("utf8")); }
      catch { return done("invalid_response", { error: "JSON inválido", headersMs }); }
      if (!body || body.status !== "ok" || typeof body.model !== "string") return done("invalid_response", { error: "Respuesta de health inesperada", headersMs });
      const info = {
        headersMs, model: body.model, voice: String(body.voice || ""), sampleRate: Number(body.sample_rate) || null,
        channels: Number(body.channels) || null, encoding: String(body.encoding || ""),
      };
      return done(body.busy === true ? "busy" : "online", info);
    } catch (err) {
      return done(timedOut ? "timeout" : "offline", { error: timedOut ? "Timeout" : "Sin conexión" });
    } finally { clearTimeout(timer); }
  }

  // Sintetiza un WAV completo. Lanza con .cancelled en cancelación voluntaria (sin fallback) o error normal (fallback).
  async synthesizeWav(text, { emotion = null, mood = 50, source = "live", outDir = null, queueMs = 0, expiresAt = null, shouldCancel = null } = {}) {
    const m = { requestedEngine: "sglang", source, queueMs, retries: 0, result: "error", reason: null };
    const t0 = performance.now();
    const since = () => Math.round(performance.now() - t0);
    const input = applyFishExpressivity(text, { emotion, mood, enabled: this.config.expressive });
    let state = null;
    let tmpPath = null;
    try {
      this._requireKey();
      if (!input) throw tagged("Texto vacío para síntesis", { code: "EMPTY_TEXT" });
      // shortcut: sin partir el texto (Fase 1); la respuesta larga cae al fallback existente
      if (input.length > SGLANG_MAX_INPUT_CHARS) throw tagged(`Texto supera ${SGLANG_MAX_INPUT_CHARS} caracteres para SGLang`, { code: "TEXT_TOO_LONG" });
      const target = await this._target();

      this.abortSource(source);
      const controller = new AbortController();
      state = { controller, cancelled: false, timedOut: false };
      this._active.set(source, state);
      const timer = setTimeout(() => { state.timedOut = true; controller.abort(new Error("timeout")); }, this.config.timeoutMs);
      const body = Buffer.from(JSON.stringify({ model: SGLANG_MODEL, voice: SGLANG_VOICE, input, stream: false, response_format: "wav" }));
      try {
        let res;
        for (let attempt = 1; ; attempt++) {
          // Stop pudo llegar mientras se resolvía DNS o durante el backoff: no ocupar el único worker de GPU
          if (shouldCancel?.()) { state.cancelled = true; throw tagged("cancelado", { cancelled: true }); }
          m.requestStartMs = since();
          res = await pinnedRequest(target, "/v1/audio/speech", {
            method: "POST",
            headers: { Authorization: `Bearer ${this.config.apiKey}`, "Content-Type": "application/json", "Content-Length": body.length },
            body, signal: controller.signal, maxBytes: this.maxResponseBytes,
            onFirstChunk: () => { m.firstByteMs = since(); },
          });
          m.headersMs = since();
          if (res.status !== 429 || attempt >= MAX_ATTEMPTS) break;
          res.arrayBuffer().catch(() => {}); // drenar el 429
          m.retries += 1;
          const ra = Number(res.headersRaw?.["retry-after"]);
          const waitMs = Number.isFinite(ra) && ra >= 0 ? Math.min(MAX_RETRY_WAIT_MS, Math.round(ra * 1000)) : attempt * 500;
          // Si el pedido vence durante el backoff, se descarta en vez de reintentar (sin fallback)
          if (expiresAt && Date.now() + waitMs > expiresAt) throw tagged("SGLang TTS: pedido vencido durante el reintento", { expired: true });
          await abortableWait(waitMs, controller.signal);
        }
        if (!res.ok || mime(res.headersRaw) !== WAV_MIME) res.arrayBuffer().catch(() => {}); // drenar el cuerpo descartado
        if (res.status === 429) throw tagged("SGLang TTS ocupado (HTTP 429, reintentos agotados)", { code: "BUSY", status: 429 });
        if (res.status === 401 || res.status === 403) { this.metrics.status = "unauthorized"; throw tagged(`SGLang TTS HTTP ${res.status}`, { status: res.status }); }
        if (!res.ok) throw tagged(`SGLang TTS HTTP ${res.status}`, { status: res.status });
        if (mime(res.headersRaw) !== WAV_MIME) throw tagged("SGLang TTS devolvió un Content-Type que no es audio/wav", { code: "BAD_CONTENT_TYPE" });
        const ab = await res.arrayBuffer();
        m.downloadMs = since();
        if (state.cancelled) throw tagged("cancelado", { cancelled: true });
        const tv = performance.now();
        const wav = validateSglangWav(Buffer.from(ab));
        m.validateMs = Math.round(performance.now() - tv);
        m.audioDurationSec = wav.audioDurationSec;
        m.sampleRate = wav.sampleRate;

        let filePath = null;
        if (outDir) {
          const tw = performance.now();
          mkdirSync(outDir, { recursive: true });
          const id = `${Date.now()}${Math.floor(Math.random() * 1000)}`; // los 13 primeros dígitos = timestamp (limpieza existente)
          tmpPath = join(outDir, `${id}.wav.tmp`);
          writeFileSync(tmpPath, wav.buf);
          if (state.cancelled) throw tagged("cancelado", { cancelled: true });
          filePath = join(outDir, `${id}.wav`);
          renameSync(tmpPath, filePath);
          tmpPath = null;
          m.writeMs = Math.round(performance.now() - tw);
        }
        m.totalMs = since();
        m.rtf = wav.audioDurationSec > 0 ? Number((m.totalMs / (wav.audioDurationSec * 1000)).toFixed(2)) : 0;
        m.result = "ok";
        this.metrics.status = "online";
        this.metrics.lastError = null;
        return { buffer: wav.buf, filePath, synthText: input, metrics: m };
      } finally {
        clearTimeout(timer);
      }
    } catch (err) {
      m.totalMs = since();
      if (err.expired) {
        m.result = "expired";
        throw err;
      }
      if (state?.cancelled || err.cancelled) {
        this.metrics.cancellations += 1;
        m.result = "cancelled";
        m.reason = "manual";
        throw tagged("SGLang TTS cancelado", { cancelled: true });
      }
      this.metrics.failures += 1;
      if (state?.timedOut) {
        m.reason = "timeout";
        this.metrics.lastError = `Timeout (${this.config.timeoutMs}ms)`;
        throw tagged(this.metrics.lastError, { timedOut: true });
      }
      m.reason = err.status ? `http_${err.status}` : (err.code || "network");
      // Errores de sistema de Node (ECONNREFUSED, ERR_...) traen IP/puerto: al público va un mensaje genérico
      const sys = err.syscall || /^(E[A-Z]+|ERR_)/.test(String(err.code || ""));
      this.metrics.lastError = sys ? "Error de conexión con SGLang TTS" : err.message;
      throw tagged(this.metrics.lastError, { code: err.code, status: err.status });
    } finally {
      if (tmpPath) try { unlinkSync(tmpPath); } catch { }
      if (state && this._active.get(source) === state) this._active.delete(source);
      this.metrics.last = m;
      // Telemetría sin texto, claves ni cabeceras
      this.logger?.log?.(`📊 TTS sglang [${source}] ${m.result}${m.reason ? `(${m.reason})` : ""} cola=${m.queueMs}ms cab=${m.headersMs ?? "-"}ms 1erByte=${m.firstByteMs ?? "-"}ms total=${m.totalMs}ms dur=${m.audioDurationSec ?? "-"}s RTF=${m.rtf ?? "-"} reintentos=${m.retries}`);
    }
  }
}

function abortableWait(ms, signal) {
  return new Promise((res, rej) => {
    if (signal.aborted) return rej(signal.reason);
    const onAbort = () => { clearTimeout(t); rej(signal.reason); };
    const t = setTimeout(() => { signal.removeEventListener("abort", onAbort); res(); }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function clampInt(v, min, max, def) {
  const n = Math.round(Number(v));
  if (!Number.isFinite(n)) return def;
  return Math.max(min, Math.min(max, n));
}
