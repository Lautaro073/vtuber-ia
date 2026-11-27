import { lookup as dnsLookup } from "node:dns/promises";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import {
  assertSafeFishTarget,
  pinnedRequest,
  resolveAuthorizedVoicePath,
  validateFishUrl,
} from "../tts/fish-audio-client.js";

export const VALID_STT_MODELS = ["large-v3-turbo", "large-v3", "medium", "small"];
const MAX_AUDIO_SECS = 60;
const MAX_AUDIO_BYTES = 25 * 1024 * 1024;

export function pcmToWavBuffer(pcm, sampleRate = 16000, channels = 1, bitsPerSample = 16) {
  const buf = Buffer.isBuffer(pcm) ? pcm : Buffer.from(pcm);
  const byteRate = sampleRate * channels * (bitsPerSample / 8);
  const blockAlign = channels * (bitsPerSample / 8);
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, 4, "ascii");
  header.writeUInt32LE(36 + buf.length, 4);
  header.write("WAVE", 8, 4, "ascii");
  header.write("fmt ", 12, 4, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write("data", 36, 4, "ascii");
  header.writeUInt32LE(buf.length, 40);
  return Buffer.concat([header, buf]);
}

export function buildMultipartFormData({ fields = {}, fileField = "file", fileName = "audio.wav", fileBuffer, mimeType = "audio/wav" }) {
  const boundary = `----NyraSttBoundary${Date.now().toString(16)}${Math.random().toString(16).slice(2, 8)}`;
  const parts = [];
  for (const [k, v] of Object.entries(fields)) {
    if (v === undefined || v === null || v === "") continue;
    parts.push(Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="${String(k).replace(/"/g, "")}"\r\n\r\n${String(v)}\r\n`,
      "utf8"
    ));
  }
  parts.push(Buffer.from(
    `--${boundary}\r\nContent-Disposition: form-data; name="${fileField}"; filename="${fileName}"\r\nContent-Type: ${mimeType}\r\n\r\n`,
    "utf8"
  ));
  parts.push(Buffer.isBuffer(fileBuffer) ? fileBuffer : Buffer.from(fileBuffer));
  parts.push(Buffer.from(`\r\n--${boundary}--\r\n`, "utf8"));
  return {
    contentType: `multipart/form-data; boundary=${boundary}`,
    body: Buffer.concat(parts),
  };
}

function atomicWriteJson(filePath, data) {
  const dir = resolve(filePath, "..");
  mkdirSync(dir, { recursive: true });
  const tmp = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  writeFileSync(tmp, JSON.stringify(data, null, 2));
  renameSync(tmp, filePath);
}

export class WhisperRemoteClient {
  constructor({
    config = {},
    voicesDir,
    settingsFile,
    fetchImpl = null,
    lookupImpl = dnsLookup,
    allowedHosts = null,
  } = {}) {
    this.fetchImpl = fetchImpl;
    this.lookupImpl = lookupImpl;
    this.allowedHosts = allowedHosts;
    this.voicesDir = resolve(voicesDir || join(process.cwd(), "voices", "stt"));
    this.settingsFile = resolve(settingsFile || join(process.cwd(), "stt_settings.json"));

    const saved = this._loadSavedSettings();
    const merged = { ...config };
    for (const [k, v] of Object.entries(saved)) {
      if (typeof v === "string") {
        if (v.trim() !== "") merged[k] = v;
      } else if (v !== undefined && v !== null) {
        merged[k] = v;
      }
    }

    this.config = {
      engine: merged.engine === "remote" ? "remote" : "local",
      url: String(merged.url || "").trim(),
      apiKey: String(config.apiKey || "").trim(), // nunca desde disco ni frontend
      model: VALID_STT_MODELS.includes(merged.model) ? merged.model : "large-v3-turbo",
      timeoutMs: clampInt(merged.timeoutMs, 2000, 60000, 10000),
      language: String(merged.language || "es").trim() || "es",
      beamSize: clampInt(merged.beamSize, 1, 5, 3),
      temperature: clampNum(merged.temperature, 0.0, 1.0, 0.0),
      vadFilter: merged.vadFilter !== false,
      conditionOnPreviousText: Boolean(merged.conditionOnPreviousText),
      hotwords: String(merged.hotwords ?? "Nyra, Discord, TikTok, Wild Rift").trim(),
      fallbackLocal: merged.fallbackLocal !== false,
      maxQueue: clampInt(merged.maxQueue, 1, 20, 10),
      maxAgeMs: clampInt(merged.maxAgeMs, 2000, 60000, 15000),
    };

    this._queue = [];
    this._running = false;
    this._activeControllers = new Set();
    this._cancelled = false;

    this.metrics = {
      status: "unknown",
      effectiveModel: null,
      effectiveDevice: null,
      effectiveQuantization: null,
      lastTranscript: "",
      httpTotalMs: 0,
      inferenceMs: 0,
      queueMs: 0,
      audioDurationSec: 0,
      rtf: 0,
      discardedCount: 0,
      lastDiscardReason: null,
      fallbacks: 0,
      failures: 0,
      cancellations: 0,
      lastError: null,
      lastCheckedAt: null,
    };
  }

  _loadSavedSettings() {
    try {
      if (!existsSync(this.settingsFile)) return {};
      const parsed = JSON.parse(readFileSync(this.settingsFile, "utf8"));
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      return {};
    }
  }

  _persistSettings() {
    const { apiKey: _omit, ...publicCfg } = this.config;
    atomicWriteJson(this.settingsFile, publicCfg);
  }

  updateConfig(patch = {}) {
    if (patch.engine !== undefined) {
      const e = String(patch.engine).trim().toLowerCase();
      if (e !== "local" && e !== "remote") throw new Error("Motor STT inválido (solo local o remote)");
      this.config.engine = e;
    }
    if (patch.url !== undefined) {
      const u = String(patch.url || "").trim();
      this.config.url = u ? validateFishUrl(u, { allowedHosts: this.allowedHosts, serviceName: "Whisper remoto", urlVar: "WHISPER_REMOTE_URL" }) : "";
    }
    if (patch.model !== undefined) {
      const m = String(patch.model || "").trim();
      if (!VALID_STT_MODELS.includes(m)) throw new Error(`Modelo STT no soportado: ${m}`);
      this.config.model = m;
    }
    if (patch.timeoutMs !== undefined) this.config.timeoutMs = clampInt(patch.timeoutMs, 2000, 60000, this.config.timeoutMs);
    if (patch.language !== undefined) this.config.language = String(patch.language || "es").trim() || "es";
    if (patch.beamSize !== undefined) this.config.beamSize = clampInt(patch.beamSize, 1, 5, this.config.beamSize);
    if (patch.temperature !== undefined) this.config.temperature = clampNum(patch.temperature, 0.0, 1.0, this.config.temperature);
    if (patch.vadFilter !== undefined) this.config.vadFilter = Boolean(patch.vadFilter);
    if (patch.conditionOnPreviousText !== undefined) this.config.conditionOnPreviousText = Boolean(patch.conditionOnPreviousText);
    if (patch.hotwords !== undefined) this.config.hotwords = String(patch.hotwords || "").slice(0, 500).trim();
    if (patch.fallbackLocal !== undefined) this.config.fallbackLocal = Boolean(patch.fallbackLocal);
    if (patch.maxQueue !== undefined) this.config.maxQueue = clampInt(patch.maxQueue, 1, 20, this.config.maxQueue);
    if (patch.maxAgeMs !== undefined) this.config.maxAgeMs = clampInt(patch.maxAgeMs, 2000, 60000, this.config.maxAgeMs);

    this._persistSettings();
    return this.getPublicConfig();
  }

  getPublicConfig() {
    return {
      engine: this.config.engine,
      url: this.config.url,
      apiKeyConfigured: Boolean(this.config.apiKey),
      model: this.config.model,
      timeoutMs: this.config.timeoutMs,
      language: this.config.language,
      beamSize: this.config.beamSize,
      temperature: this.config.temperature,
      vadFilter: this.config.vadFilter,
      conditionOnPreviousText: this.config.conditionOnPreviousText,
      hotwords: this.config.hotwords,
      fallbackLocal: this.config.fallbackLocal,
      maxQueue: this.config.maxQueue,
      maxAgeMs: this.config.maxAgeMs,
      metrics: { ...this.metrics },
    };
  }

  recordDiscard(reason) {
    this.metrics.discardedCount += 1;
    this.metrics.lastDiscardReason = reason;
  }

  recordFallback() {
    this.metrics.fallbacks += 1;
  }

  abortAll() {
    this._cancelled = true;
    while (this._queue.length) {
      const item = this._queue.shift();
      this.metrics.cancellations += 1;
      item.reject(Object.assign(new Error("STT cancelado manualmente"), { cancelled: true }));
    }
    for (const ac of this._activeControllers) {
      ac.abort(new Error("STT cancelado manualmente"));
    }
    this._activeControllers.clear();
  }

  resetCancellation() {
    this._cancelled = false;
  }

  async checkHealth() {
    if (!this.config.apiKey) {
      this.metrics.status = "unconfigured";
      this.metrics.lastError = "WHISPER_REMOTE_API_KEY no configurada";
      throw new Error(this.metrics.lastError);
    }
    const target = await assertSafeFishTarget(this.config.url, {
      lookupImpl: this.lookupImpl,
      allowedHosts: this.allowedHosts,
      serviceName: "Whisper remoto",
      urlVar: "WHISPER_REMOTE_URL",
    });
    const t0 = performance.now();
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(new Error("STT health timeout")), Math.min(8000, this.config.timeoutMs));
    try {
      const reqOpts = {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${this.config.apiKey}`,
          "ngrok-skip-browser-warning": "1",
        },
        signal: ac.signal,
      };
      const res = this.fetchImpl
        ? await this.fetchImpl(`${target.origin}/v1/stt/health`, { ...reqOpts, redirect: "error" })
        : await pinnedRequest(target, "/v1/stt/health", reqOpts);
      const httpTotalMs = Math.round(performance.now() - t0);
      if (!res.ok) {
        this.metrics.status = res.status === 401 ? "unauthorized" : res.status === 429 ? "busy" : "error";
        this.metrics.lastError = `STT Health HTTP ${res.status}`;
        this.metrics.lastCheckedAt = Date.now();
        throw new Error(this.metrics.lastError);
      }
      let body = null;
      try { body = await res.json(); } catch { body = { status: "ok" }; }
      this.metrics.status = "online";
      this.metrics.effectiveModel = body?.model || this.metrics.effectiveModel;
      this.metrics.effectiveDevice = body?.device || this.metrics.effectiveDevice;
      this.metrics.effectiveQuantization = body?.compute_type || body?.quantization || this.metrics.effectiveQuantization;
      this.metrics.lastError = null;
      this.metrics.lastCheckedAt = Date.now();
      return { ok: true, status: res.status, httpTotalMs, body };
    } catch (err) {
      if (this.metrics.status !== "unauthorized") this.metrics.status = "offline";
      this.metrics.lastError = err.message;
      this.metrics.lastCheckedAt = Date.now();
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  transcribePcm(pcm, { speakerId = "unknown", capturedAt = Date.now(), model = null } = {}) {
    const secs = pcm.length / 32000;
    if (secs > MAX_AUDIO_SECS) {
      this.recordDiscard("audio_too_long");
      return Promise.reject(new Error(`Audio excede el límite de ${MAX_AUDIO_SECS}s`));
    }
    const wav = pcmToWavBuffer(pcm, 16000, 1, 16);
    return this._enqueueAudio({
      fileBuffer: wav,
      fileName: "audio.wav",
      mimeType: "audio/wav",
      durationSec: Number(secs.toFixed(2)),
      speakerId,
      capturedAt,
      model,
    });
  }

  transcribeAuthorizedFile(filePath) {
    const info = resolveAuthorizedVoicePath(this.voicesDir, filePath);
    const buf = readFileSync(info.path);
    const ext = extname(info.path).toLowerCase();
    const mimeMap = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".flac": "audio/flac" };
    return this._enqueueAudio({
      fileBuffer: buf,
      fileName: `test${ext}`,
      mimeType: mimeMap[ext] || "application/octet-stream",
      durationSec: 0,
      speakerId: "panel_test",
      capturedAt: Date.now(),
      model: null,
    });
  }

  _enqueueAudio(job) {
    if (this._cancelled) {
      return Promise.reject(Object.assign(new Error("STT cancelado manualmente"), { cancelled: true }));
    }
    if (job.fileBuffer.length > MAX_AUDIO_BYTES) {
      this.recordDiscard("file_too_large");
      return Promise.reject(new Error("Archivo de audio excede el máximo permitido"));
    }
    if (Date.now() - job.capturedAt > this.config.maxAgeMs) {
      this.recordDiscard("stale_before_queue");
      return Promise.reject(Object.assign(new Error("Fragmento expirado antes de entrar a cola"), { stale: true }));
    }
    if (this._queue.length >= this.config.maxQueue) {
      this.recordDiscard("queue_full");
      return Promise.reject(Object.assign(new Error("Cola STT remota llena"), { queueFull: true }));
    }
    return new Promise((resolve, reject) => {
      this._queue.push({ ...job, enqueuedAt: Date.now(), resolve, reject });
      this._drain();
    });
  }

  async _drain() {
    if (this._running) return;
    this._running = true;
    try {
      while (this._queue.length > 0) {
        const job = this._queue.shift();
        if (this._cancelled) {
          this.metrics.cancellations += 1;
          job.reject(Object.assign(new Error("STT cancelado manualmente"), { cancelled: true }));
          continue;
        }
        const ageMs = Date.now() - job.capturedAt;
        if (ageMs > this.config.maxAgeMs) {
          this.recordDiscard("stale_in_queue");
          job.reject(Object.assign(new Error("Fragmento expirado en cola"), { stale: true }));
          continue;
        }
        try {
          const res = await this._sendTranscription(job);
          job.resolve(res);
        } catch (err) {
          job.reject(err);
        }
      }
    } finally {
      this._running = false;
    }
  }

  async _sendTranscription(job, attempt = 0) {
    if (!this.config.apiKey) {
      this.metrics.failures += 1;
      this.metrics.lastError = "WHISPER_REMOTE_API_KEY no configurada";
      throw Object.assign(new Error(this.metrics.lastError), { code: "UNCONFIGURED" });
    }
    const target = await assertSafeFishTarget(this.config.url, {
      lookupImpl: this.lookupImpl,
      allowedHosts: this.allowedHosts,
      serviceName: "Whisper remoto",
      urlVar: "WHISPER_REMOTE_URL",
    });
    const queueMs = Math.max(0, Date.now() - job.enqueuedAt);
    const chosenModel = job.model || this.config.model;

    const { contentType, body } = buildMultipartFormData({
      fields: {
        model: chosenModel,
        language: this.config.language,
        response_format: "verbose_json",
        beam_size: String(this.config.beamSize),
        temperature: String(this.config.temperature),
        vad_filter: this.config.vadFilter ? "true" : "false",
        condition_on_previous_text: this.config.conditionOnPreviousText ? "true" : "false",
        hotwords: this.config.hotwords || undefined,
      },
      fileField: "file",
      fileName: job.fileName,
      fileBuffer: job.fileBuffer,
      mimeType: job.mimeType,
    });

    const controller = new AbortController();
    this._activeControllers.add(controller);
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort(new Error(`STT timeout (${this.config.timeoutMs}ms)`));
    }, this.config.timeoutMs);

    const t0 = performance.now();
    try {
      const reqOpts = {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.config.apiKey}`,
          "Content-Type": contentType,
          "Content-Length": String(body.length),
          "ngrok-skip-browser-warning": "1",
        },
        body,
        signal: controller.signal,
      };
      const res = this.fetchImpl
        ? await this.fetchImpl(`${target.origin}/v1/audio/transcriptions`, { ...reqOpts, redirect: "error" })
        : await pinnedRequest(target, "/v1/audio/transcriptions", reqOpts);

      const httpTotalMs = Math.round(performance.now() - t0);

      if (res.status === 429) {
        this.metrics.status = "busy";
        // Un único reintento acotado si el fragmento sigue vigente
        if (attempt === 0 && !this._cancelled && (Date.now() - job.capturedAt + 300) < this.config.maxAgeMs) {
          await new Promise(r => setTimeout(r, 300));
          return await this._sendTranscription(job, 1);
        }
        throw Object.assign(new Error("Servidor STT ocupado (HTTP 429)"), { status: 429, busy: true });
      }
      if (!res.ok) {
        throw Object.assign(new Error(`STT HTTP ${res.status}`), { status: res.status });
      }

      const data = await res.json();
      const text = String(data?.text || "").trim();
      const duration = Number(data?.duration || job.durationSec || 0);
      const inferenceMs = Number(data?.metrics?.inference_ms || 0);
      const rtf = duration > 0 ? Number((httpTotalMs / (duration * 1000)).toFixed(2)) : Number(data?.metrics?.rtf || 0);

      this.metrics.status = "online";
      this.metrics.effectiveModel = data?.model || this.metrics.effectiveModel || chosenModel;
      this.metrics.lastTranscript = text;
      this.metrics.httpTotalMs = httpTotalMs;
      this.metrics.inferenceMs = inferenceMs;
      this.metrics.queueMs = queueMs;
      this.metrics.audioDurationSec = duration;
      this.metrics.rtf = rtf;
      this.metrics.lastError = null;

      return {
        text,
        emptySilence: text.length === 0,
        language: data?.language || this.config.language,
        duration,
        segments: Array.isArray(data?.segments) ? data.segments : [],
        model: this.metrics.effectiveModel,
        ms: httpTotalMs,
        queueMs,
        metrics: data?.metrics || { total_ms: httpTotalMs, rtf },
      };
    } catch (err) {
      if (this._cancelled || (controller.signal.aborted && !timedOut)) {
        this.metrics.cancellations += 1;
        this.metrics.lastError = "Cancelado manualmente";
        throw Object.assign(new Error("STT cancelado manualmente"), { cancelled: true });
      }
      this.metrics.failures += 1;
      if (timedOut) {
        this.metrics.lastError = `Timeout (${this.config.timeoutMs}ms)`;
        throw Object.assign(new Error(this.metrics.lastError), { timedOut: true });
      }
      this.metrics.lastError = err.message;
      throw err;
    } finally {
      clearTimeout(timer);
      this._activeControllers.delete(controller);
    }
  }
}

function clampNum(v, min, max, def) {
  const n = Number(v);
  if (!Number.isFinite(n)) return def;
  return Math.max(min, Math.min(max, n));
}

function clampInt(v, min, max, def) {
  const n = Math.round(Number(v));
  if (!Number.isFinite(n)) return def;
  return Math.max(min, Math.min(max, n));
}
