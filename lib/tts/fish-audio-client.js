import { encode } from "@msgpack/msgpack";
import { lookup as dnsLookup } from "node:dns/promises";
import { existsSync, mkdirSync, readFileSync, realpathSync, renameSync, statSync, writeFileSync } from "node:fs";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { isIP } from "node:net";
import { extname, isAbsolute, join, relative, resolve } from "node:path";

const ALLOWED_AUDIO_EXTS = new Set([".wav", ".mp3", ".ogg", ".flac"]);
const MAX_REF_BYTES = 15 * 1024 * 1024;
const MAX_WAV_BYTES = 25 * 1024 * 1024;

function ipv4ToInt(ip) {
  const parts = ip.split(".").map(Number);
  if (parts.length !== 4 || parts.some(p => !Number.isInteger(p) || p < 0 || p > 255)) return null;
  return ((parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]) >>> 0;
}

export function isPrivateOrReservedIp(ip) {
  const raw = String(ip || "").trim().toLowerCase().replace(/^\[|\]$/g, "");
  const kind = isIP(raw);
  if (!kind) return true;
  if (kind === 4) {
    const n = ipv4ToInt(raw);
    if (n === null) return true;
    const inCidr = (base, bits) => {
      const mask = bits === 0 ? 0 : (~0 << (32 - bits)) >>> 0;
      return (n & mask) === (ipv4ToInt(base) & mask);
    };
    return (
      inCidr("0.0.0.0", 8) ||
      inCidr("10.0.0.0", 8) ||
      inCidr("100.64.0.0", 10) ||
      inCidr("127.0.0.0", 8) ||
      inCidr("169.254.0.0", 16) ||
      inCidr("172.16.0.0", 12) ||
      inCidr("192.168.0.0", 16) ||
      inCidr("224.0.0.0", 4) ||
      inCidr("240.0.0.0", 4)
    );
  }
  // IPv6
  if (raw === "::" || raw === "::1") return true;
  if (raw.startsWith("::ffff:")) {
    return isPrivateOrReservedIp(raw.slice(7));
  }
  const firstHextet = parseInt(raw.split(":")[0] || "0", 16);
  if ((firstHextet & 0xfe00) === 0xfc00) return true; // fc00::/7 ULA
  if ((firstHextet & 0xffc0) === 0xfe80) return true; // fe80::/10 link-local
  if (raw.startsWith("ff")) return true; // multicast
  return false;
}

function isLoopbackHost(hostname) {
  const h = String(hostname || "").toLowerCase().replace(/^\[|\]$/g, "");
  return h === "localhost" || h === "127.0.0.1" || h === "::1";
}

export function validateFishUrl(rawUrl, { allowedHosts = null, serviceName = "Fish Audio", urlVar = "FISH_TTS_URL" } = {}) {
  const s = String(rawUrl || "").trim();
  if (!s) throw new Error(`${urlVar} no configurada`);
  let u;
  try {
    u = new URL(s);
  } catch {
    throw new Error(`URL de ${serviceName} inválida`);
  }
  if (u.protocol !== "https:" && u.protocol !== "http:") {
    throw new Error(`Protocolo de ${serviceName} no permitido (solo http/https)`);
  }
  if (u.username || u.password) {
    throw new Error(`La URL de ${serviceName} no debe incluir credenciales embebidas`);
  }
  const host = u.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (!host) throw new Error(`Host de ${serviceName} vacío`);

  if (Array.isArray(allowedHosts) && allowedHosts.length > 0) {
    const ok = allowedHosts.some(pattern => {
      const p = String(pattern || "").trim().toLowerCase();
      if (!p) return false;
      return p.startsWith(".") ? host.endsWith(p) || host === p.slice(1) : host === p;
    });
    if (!ok) throw new Error(`Host no incluido en lista permitida de ${serviceName}: ${host}`);
  }

  if (!isLoopbackHost(host)) {
    // Bloquea formatos alternativos de IPv4 (octal/hex/entero) si URL ya los normalizó a IPv4
    if (isIP(host) && isPrivateOrReservedIp(host)) {
      throw new Error(`Destino privado/reservado no autorizado para ${serviceName}: ${host}`);
    }
  }
  return u.origin;
}

export async function assertSafeFishTarget(rawUrl, { lookupImpl = dnsLookup, allowedHosts = null, serviceName = "Fish Audio", urlVar = "FISH_TTS_URL" } = {}) {
  const origin = validateFishUrl(rawUrl, { allowedHosts, serviceName, urlVar });
  const u = new URL(origin);
  const host = u.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (isLoopbackHost(host)) {
    const family = host === "::1" ? 6 : 4;
    const pinnedAddress = host === "localhost" ? "127.0.0.1" : host;
    return { origin, hostname: host, port: u.port, protocol: u.protocol, pinnedAddress, pinnedFamily: family };
  }
  if (isIP(host)) {
    if (isPrivateOrReservedIp(host)) throw new Error(`IP privada/reservada no autorizada: ${host}`);
    return { origin, hostname: host, port: u.port, protocol: u.protocol, pinnedAddress: host, pinnedFamily: isIP(host) };
  }
  const records = await lookupImpl(host, { all: true });
  if (!Array.isArray(records) || records.length === 0) {
    throw new Error(`No se pudo resolver DNS para ${host}`);
  }
  for (const rec of records) {
    if (isPrivateOrReservedIp(rec.address)) {
      throw new Error(`DNS resolvió a IP privada/reservada no autorizada (${rec.address})`);
    }
  }
  const first = records[0];
  return {
    origin,
    hostname: host,
    port: u.port,
    protocol: u.protocol,
    pinnedAddress: first.address,
    pinnedFamily: first.family || isIP(first.address) || 4,
  };
}

export function pinnedRequest(target, pathAndQuery, { method = "GET", headers = {}, body = null, signal = null } = {}) {
  return new Promise((resolveReq, rejectReq) => {
    const isHttps = target.protocol === "https:";
    const reqFn = isHttps ? httpsRequest : httpRequest;
    const req = reqFn({
      protocol: target.protocol,
      hostname: target.hostname,
      port: target.port || (isHttps ? 443 : 80),
      path: pathAndQuery,
      method,
      headers,
      signal,
      servername: isHttps && !isIP(target.hostname) ? target.hostname : undefined,
      lookup: (_hostname, opts, cb) => {
        if (typeof opts === "function") return opts(null, target.pinnedAddress, target.pinnedFamily);
        if (opts && opts.all) return cb(null, [{ address: target.pinnedAddress, family: target.pinnedFamily }]);
        return cb(null, target.pinnedAddress, target.pinnedFamily);
      },
    }, res => {
      const status = res.statusCode || 0;
      const ok = status >= 200 && status < 300;
      resolveReq({
        ok,
        status,
        arrayBuffer: () => new Promise((resAb, rejAb) => {
          const chunks = [];
          let total = 0;
          res.on("data", c => {
            total += c.length;
            if (total > MAX_WAV_BYTES) {
              res.destroy(new Error(`WAV excede el tamaño máximo permitido (${MAX_WAV_BYTES} bytes)`));
              return;
            }
            chunks.push(c);
          });
          res.on("end", () => {
            const buf = Buffer.concat(chunks);
            resAb(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
          });
          res.on("error", rejAb);
        }),
        json: () => new Promise((resJs, rejJs) => {
          const chunks = [];
          res.on("data", c => chunks.push(c));
          res.on("end", () => {
            try { resJs(JSON.parse(Buffer.concat(chunks).toString("utf8"))); }
            catch (e) { rejJs(e); }
          });
          res.on("error", rejJs);
        }),
      });
    });
    req.on("error", rejectReq);
    if (body) req.write(body);
    req.end();
  });
}

export function resolveAuthorizedVoicePath(voicesDir, filePath) {
  const raw = String(filePath || "").trim();
  if (!raw) throw new Error("Ruta de archivo de referencia vacía");
  if (/^\/content\/drive\//i.test(raw)) {
    throw new Error("Las rutas internas de Google Drive (/content/drive/...) no son accesibles desde Windows");
  }
  const baseDir = resolve(voicesDir);
  const target = isAbsolute(raw) ? resolve(raw) : resolve(baseDir, raw);
  const relLex = relative(baseDir, target);
  if (!relLex || relLex.startsWith("..") || isAbsolute(relLex)) {
    throw new Error("Archivo de referencia fuera del directorio autorizado");
  }
  const ext = extname(target).toLowerCase();
  if (!ALLOWED_AUDIO_EXTS.has(ext)) {
    throw new Error(`Formato de referencia no permitido: ${ext || "sin extensión"}`);
  }
  if (!existsSync(target)) {
    throw new Error(`Archivo de referencia no encontrado: ${relLex}`);
  }
  // Verificar realpathSync para bloquear symlinks y junctions de Windows que apunten fuera de voices/fish/
  const realBase = existsSync(baseDir) ? realpathSync(baseDir) : baseDir;
  const realTarget = realpathSync(target);
  const relReal = relative(realBase, realTarget);
  if (!relReal || relReal.startsWith("..") || isAbsolute(relReal)) {
    throw new Error("Enlace simbólico o junction fuera del directorio autorizado");
  }
  const st = statSync(realTarget);
  if (!st.isFile()) throw new Error("La ruta de referencia no es un archivo válido");
  if (st.size <= 0 || st.size > MAX_REF_BYTES) {
    throw new Error(`Tamaño de referencia fuera de rango (máx ${MAX_REF_BYTES} bytes)`);
  }
  return { path: realTarget, rel: relReal, mtimeMs: st.mtimeMs, size: st.size };
}

export function validateWavBuffer(rawBuf) {
  const buf = Buffer.isBuffer(rawBuf) ? rawBuf : Buffer.from(rawBuf);
  if (buf.length < 44) throw new Error("WAV inválido o vacío (< 44 bytes)");
  if (buf.length > MAX_WAV_BYTES) throw new Error(`WAV excede el tamaño máximo permitido (${MAX_WAV_BYTES} bytes)`);
  if (buf.toString("ascii", 0, 4) !== "RIFF" || buf.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error("Respuesta no es un archivo WAV RIFF/WAVE válido");
  }

  let offset = 12;
  let byteRate = 0;
  let dataSize = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;

  while (offset + 8 <= buf.length) {
    const id = buf.toString("ascii", offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    const dataStart = offset + 8;
    if (dataStart + size > buf.length && id !== "data") break;
    if (id === "fmt " && size >= 16 && dataStart + 16 <= buf.length) {
      channels = buf.readUInt16LE(dataStart + 2);
      sampleRate = buf.readUInt32LE(dataStart + 4);
      byteRate = buf.readUInt32LE(dataStart + 8);
      bitsPerSample = buf.readUInt16LE(dataStart + 14);
    } else if (id === "data") {
      dataSize = Math.min(size, Math.max(0, buf.length - dataStart));
      break;
    }
    offset = dataStart + size + (size % 2);
  }

  const effectiveByteRate = byteRate || (sampleRate && channels && bitsPerSample ? sampleRate * channels * (bitsPerSample / 8) : 0);
  if (!dataSize || !effectiveByteRate) {
    throw new Error("WAV corrupto: faltan chunks fmt/data válidos");
  }
  const audioDurationSec = Number((dataSize / effectiveByteRate).toFixed(3));
  return { buf, audioDurationSec, sampleRate, channels, bitsPerSample };
}

export function applyFishExpressivity(text, { emotion = null, mood = 50, enabled = true } = {}) {
  const clean = String(text || "").trim();
  if (!clean || !enabled) return clean;
  const emo = String(emotion || "").trim().toLowerCase();
  const m = Number.isFinite(Number(mood)) ? Number(mood) : 50;

  let tag = "";
  if (emo === "happy") tag = m >= 85 ? "[excited]" : "[laughing]";
  else if (emo === "sad") tag = "[sad]";
  else if (emo === "angry") tag = "[angry]";
  else if (emo === "surprised") tag = "[surprised]";
  else if (emo === "neutral" && m <= 14) tag = "[dry sarcastic tone]";

  return tag ? `${tag} ${clean}` : clean;
}

function atomicWriteJson(filePath, data) {
  const dir = resolve(filePath, "..");
  mkdirSync(dir, { recursive: true });
  const tmp = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  writeFileSync(tmp, JSON.stringify(data, null, 2));
  renameSync(tmp, filePath);
}

export class FishAudioClient {
  constructor({
    config = {},
    voicesDir,
    profilesFile,
    settingsFile,
    fetchImpl = null,
    lookupImpl = dnsLookup,
    allowedHosts = null,
  } = {}) {
    this.fetchImpl = fetchImpl;
    this.lookupImpl = lookupImpl;
    this.allowedHosts = allowedHosts;
    this.voicesDir = resolve(voicesDir || join(process.cwd(), "voices", "fish"));
    this.profilesFile = resolve(profilesFile || join(process.cwd(), "fish_voices.json"));
    this.settingsFile = resolve(settingsFile || join(process.cwd(), "tts_settings.json"));

    const saved = this._loadSavedSettings();
    const savedFish = saved.fish || {};
    const merged = { ...config };
    for (const [k, v] of Object.entries(savedFish)) {
      if (typeof v === "string") {
        if (v.trim() !== "") merged[k] = v;
      } else if (v !== undefined && v !== null) {
        merged[k] = v;
      }
    }

    this.config = {
      url: String(merged.url || "").trim(),
      apiKey: String(config.apiKey || "").trim(), // apiKey siempre solo desde .env/config de arranque, nunca desde disco
      referenceId: String(merged.referenceId || "").trim(),
      referencePath: String(merged.referencePath || "").trim(),
      referenceText: String(merged.referenceText || "").trim(),
      temperature: clampNum(merged.temperature, 0.1, 2.0, 0.7),
      topP: clampNum(merged.topP, 0.1, 1.0, 0.7),
      chunkLength: clampInt(merged.chunkLength, 50, 500, 200),
      maxNewTokens: clampInt(merged.maxNewTokens, 64, 4096, 1024),
      seed: clampInt(merged.seed, 0, 2147483647, 42),
      timeoutMs: clampInt(merged.timeoutMs, 5000, 180000, 60000),
      expressive: merged.expressive !== false,
      fallback: merged.fallback !== false,
      activeProfile: String(merged.activeProfile || "").trim(),
    };

    this.enginePrefs = {
      engine: saved.engine || null,
      shareTtsEngine: typeof saved.shareTtsEngine === "boolean" ? saved.shareTtsEngine : null,
      discordEngine: saved.discordEngine || null,
    };

    this._refCache = new Map();
    this._activeRequests = new Map();
    this.profiles = this._loadProfiles();
    this.metrics = {
      status: "unknown",
      httpTotalMs: 0,
      headersMs: 0,
      downloadMs: 0,
      audioDurationSec: 0,
      rtf: 0,
      failures: 0,
      cancellations: 0,
      fallbacks: 0,
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

  saveEnginePrefs({ engine, shareTtsEngine, discordEngine } = {}) {
    if (engine !== undefined) this.enginePrefs.engine = engine;
    if (shareTtsEngine !== undefined) this.enginePrefs.shareTtsEngine = shareTtsEngine;
    if (discordEngine !== undefined) this.enginePrefs.discordEngine = discordEngine;
    this._persistSettings();
  }

  _persistSettings() {
    const { apiKey: _omit, ...publicFish } = this.config;
    atomicWriteJson(this.settingsFile, {
      ...this.enginePrefs,
      fish: publicFish,
    });
  }

  _loadProfiles() {
    try {
      if (!existsSync(this.profilesFile)) return [];
      const parsed = JSON.parse(readFileSync(this.profilesFile, "utf8"));
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  _saveProfiles() {
    atomicWriteJson(this.profilesFile, this.profiles);
  }

  listProfiles() {
    return this.profiles.map(p => ({ ...p }));
  }

  upsertProfile(input) {
    const id = String(input?.id || input?.name || "").trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-|-$/g, "");
    const name = String(input?.name || "").trim();
    if (!id || !name) throw new Error("Nombre de perfil requerido");
    const mode = input?.mode === "reference_id" ? "reference_id" : "local_file";
    const referenceId = String(input?.referenceId || "").trim();
    const filePath = String(input?.filePath || "").trim();
    const transcript = String(input?.transcript || "").trim();
    if (mode === "reference_id" && !referenceId) {
      throw new Error("referenceId requerido para modo reference_id");
    }
    if (mode === "local_file") {
      resolveAuthorizedVoicePath(this.voicesDir, filePath);
      if (!transcript) throw new Error("Transcripción exacta requerida para referencia local");
    }
    const profile = { id, name, mode, referenceId, filePath, transcript };
    const idx = this.profiles.findIndex(p => p.id === id);
    if (idx >= 0) this.profiles[idx] = profile;
    else this.profiles.push(profile);
    this._saveProfiles();
    return { ...profile };
  }

  deleteProfile(id) {
    const before = this.profiles.length;
    this.profiles = this.profiles.filter(p => p.id !== id);
    if (this.profiles.length !== before) {
      if (this.config.activeProfile === id) {
        this.config.activeProfile = "";
        this._persistSettings();
      }
      this._saveProfiles();
      return true;
    }
    return false;
  }

  updateConfig(patch = {}) {
    if (patch.url !== undefined) {
      const u = String(patch.url || "").trim();
      this.config.url = u ? validateFishUrl(u, { allowedHosts: this.allowedHosts }) : "";
    }
    if (patch.referenceId !== undefined) this.config.referenceId = String(patch.referenceId || "").trim();
    if (patch.referencePath !== undefined) {
      const rp = String(patch.referencePath || "").trim();
      if (rp) resolveAuthorizedVoicePath(this.voicesDir, rp);
      this.config.referencePath = rp;
    }
    if (patch.referenceText !== undefined) this.config.referenceText = String(patch.referenceText || "").trim();
    if (patch.temperature !== undefined) this.config.temperature = clampNum(patch.temperature, 0.1, 2.0, this.config.temperature);
    if (patch.topP !== undefined) this.config.topP = clampNum(patch.topP, 0.1, 1.0, this.config.topP);
    if (patch.chunkLength !== undefined) this.config.chunkLength = clampInt(patch.chunkLength, 50, 500, this.config.chunkLength);
    if (patch.maxNewTokens !== undefined) this.config.maxNewTokens = clampInt(patch.maxNewTokens, 64, 4096, this.config.maxNewTokens);
    if (patch.seed !== undefined) this.config.seed = clampInt(patch.seed, 0, 2147483647, this.config.seed);
    if (patch.timeoutMs !== undefined) this.config.timeoutMs = clampInt(patch.timeoutMs, 5000, 180000, this.config.timeoutMs);
    if (patch.expressive !== undefined) this.config.expressive = Boolean(patch.expressive);
    if (patch.fallback !== undefined) this.config.fallback = Boolean(patch.fallback);
    if (patch.activeProfile !== undefined) this.config.activeProfile = String(patch.activeProfile || "").trim();
    this._refCache.clear();
    this._persistSettings();
    return this.getPublicConfig();
  }

  getPublicConfig() {
    return {
      url: this.config.url,
      apiKeyConfigured: Boolean(this.config.apiKey),
      referenceId: this.config.referenceId,
      referencePath: this.config.referencePath,
      referenceText: this.config.referenceText,
      temperature: this.config.temperature,
      topP: this.config.topP,
      chunkLength: this.config.chunkLength,
      maxNewTokens: this.config.maxNewTokens,
      seed: this.config.seed,
      timeoutMs: this.config.timeoutMs,
      expressive: this.config.expressive,
      fallback: this.config.fallback,
      activeProfile: this.config.activeProfile,
      profiles: this.listProfiles(),
      metrics: { ...this.metrics },
    };
  }

  _resolveReferencePayload() {
    if (this.config.activeProfile) {
      const profile = this.profiles.find(p => p.id === this.config.activeProfile);
      if (!profile) {
        throw new Error(`Perfil de voz "${this.config.activeProfile}" no encontrado`);
      }
      if (profile.mode === "reference_id") {
        if (!profile.referenceId) throw new Error(`Perfil "${profile.name}" sin reference_id`);
        return { reference_id: profile.referenceId };
      }
      if (profile.mode === "local_file") {
        return this._loadReferenceBytes(profile.filePath, profile.transcript);
      }
    }

    if (this.config.referenceId) {
      return { reference_id: this.config.referenceId };
    }
    if (this.config.referencePath) {
      return this._loadReferenceBytes(this.config.referencePath, this.config.referenceText);
    }
    return {};
  }

  _loadReferenceBytes(filePath, transcript) {
    const text = String(transcript || "").trim();
    if (!text) throw new Error("Transcripción de referencia requerida junto con el audio");
    const info = resolveAuthorizedVoicePath(this.voicesDir, filePath);
    const cacheKey = `${info.path}:${info.mtimeMs}:${info.size}`;
    let audio = this._refCache.get(cacheKey);
    if (!audio) {
      this._refCache.clear();
      audio = new Uint8Array(readFileSync(info.path));
      this._refCache.set(cacheKey, audio);
    }
    return { references: [{ audio, text }] };
  }

  abortSource(source) {
    const active = this._activeRequests.get(source);
    if (!active) return false;
    active.cancelled = true;
    active.controller.abort(new Error("Cancelación manual de TTS"));
    return true;
  }

  abortAll() {
    let any = false;
    for (const source of this._activeRequests.keys()) {
      if (this.abortSource(source)) any = true;
    }
    return any;
  }

  recordFallback() {
    this.metrics.fallbacks += 1;
  }

  async checkHealth() {
    if (!this.config.apiKey) {
      this.metrics.status = "unconfigured";
      this.metrics.lastError = "FISH_TTS_API_KEY no configurada";
      throw new Error(this.metrics.lastError);
    }
    const target = await assertSafeFishTarget(this.config.url, { lookupImpl: this.lookupImpl, allowedHosts: this.allowedHosts });
    const t0 = performance.now();
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(new Error("Health timeout")), Math.min(10000, this.config.timeoutMs));
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
        ? await this.fetchImpl(`${target.origin}/v1/health`, { ...reqOpts, redirect: "error" })
        : await pinnedRequest(target, "/v1/health", reqOpts);
      const headersMs = Math.round(performance.now() - t0);
      if (!res.ok) {
        this.metrics.status = res.status === 401 ? "unauthorized" : "error";
        this.metrics.lastError = `Health HTTP ${res.status}`;
        this.metrics.lastCheckedAt = Date.now();
        throw new Error(this.metrics.lastError);
      }
      let body = null;
      try { body = await res.json(); } catch { body = { status: "ok" }; }
      this.metrics.status = "online";
      this.metrics.headersMs = headersMs;
      this.metrics.lastError = null;
      this.metrics.lastCheckedAt = Date.now();
      return { ok: true, status: res.status, headersMs, body };
    } catch (err) {
      if (this.metrics.status !== "unauthorized") this.metrics.status = "offline";
      this.metrics.lastError = err.message;
      this.metrics.lastCheckedAt = Date.now();
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  async synthesizeWav(text, { emotion = null, mood = 50, source = "live", outDir = null } = {}) {
    if (!this.config.apiKey) {
      this.metrics.failures += 1;
      this.metrics.lastError = "FISH_TTS_API_KEY no configurada";
      throw Object.assign(new Error(this.metrics.lastError), { code: "UNCONFIGURED" });
    }
    const target = await assertSafeFishTarget(this.config.url, { lookupImpl: this.lookupImpl, allowedHosts: this.allowedHosts });
    const synthText = applyFishExpressivity(text, { emotion, mood, enabled: this.config.expressive });
    if (!synthText) throw new Error("Texto vacío para síntesis");

    const refPayload = this._resolveReferencePayload();
    const payload = {
      text: synthText,
      temperature: this.config.temperature,
      top_p: this.config.topP,
      chunk_length: this.config.chunkLength,
      max_new_tokens: this.config.maxNewTokens,
      seed: this.config.seed,
      format: "wav",
      ...refPayload,
    };
    const encoded = encode(payload);

    this.abortSource(source);
    const controller = new AbortController();
    const state = { controller, cancelled: false, timedOut: false };
    this._activeRequests.set(source, state);

    const timer = setTimeout(() => {
      state.timedOut = true;
      controller.abort(new Error(`Fish TTS timeout (${this.config.timeoutMs}ms)`));
    }, this.config.timeoutMs);

    const t0 = performance.now();
    try {
      let res;
      let headersMs = 0;
      const maxAttempts = 3;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        const reqOpts = {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${this.config.apiKey}`,
            "Content-Type": "application/msgpack",
            "ngrok-skip-browser-warning": "1",
          },
          body: encoded,
          signal: controller.signal,
        };
        res = this.fetchImpl
          ? await this.fetchImpl(`${target.origin}/v1/tts`, { ...reqOpts, redirect: "error" })
          : await pinnedRequest(target, "/v1/tts", reqOpts);
        headersMs = Math.round(performance.now() - t0);
        if (res.status === 429 && attempt < maxAttempts && !state.cancelled && !state.timedOut) {
          const retryAfterSec = Number(res.headers?.get?.("retry-after"));
          const waitMs = Number.isFinite(retryAfterSec) && retryAfterSec > 0
            ? Math.min(3000, Math.round(retryAfterSec * 1000))
            : attempt * 400;
          await new Promise((resolve, reject) => {
            const abortWait = () => {
              clearTimeout(wt);
              if (state.timedOut) reject(new Error(`Fish TTS timeout (${this.config.timeoutMs}ms)`));
              else reject(Object.assign(new Error("Fish TTS cancelado"), { cancelled: true }));
            };
            const wt = setTimeout(() => {
              controller.signal.removeEventListener("abort", abortWait);
              resolve();
            }, waitMs);
            controller.signal.addEventListener("abort", abortWait, { once: true });
          });
          continue;
        }
        break;
      }
      if (!res.ok) {
        if (res.status === 422) {
          this.metrics.status = "invalid_reference";
          throw Object.assign(new Error("Referencia de voz inexistente o inválida en el servidor (HTTP 422)"), { status: 422, invalidReference: true });
        }
        throw new Error(`Fish TTS HTTP ${res.status}`);
      }
      const tDl = performance.now();
      const ab = await res.arrayBuffer();
      const downloadMs = Math.round(performance.now() - tDl);
      const httpTotalMs = Math.round(performance.now() - t0);

      if (state.cancelled) {
        throw Object.assign(new Error("Cancelación manual de TTS"), { cancelled: true });
      }

      const wav = validateWavBuffer(Buffer.from(ab));
      const rtf = wav.audioDurationSec > 0 ? Number((httpTotalMs / (wav.audioDurationSec * 1000)).toFixed(2)) : 0;

      if (state.cancelled) {
        throw Object.assign(new Error("Cancelación manual de TTS"), { cancelled: true });
      }

      this.metrics.status = "online";
      this.metrics.httpTotalMs = httpTotalMs;
      this.metrics.headersMs = headersMs;
      this.metrics.downloadMs = downloadMs;
      this.metrics.audioDurationSec = wav.audioDurationSec;
      this.metrics.rtf = rtf;
      this.metrics.lastError = null;

      let filePath = null;
      if (outDir) {
        mkdirSync(outDir, { recursive: true });
        const id = `${Date.now()}${Math.floor(Math.random() * 1000)}`;
        const tmpPath = join(outDir, `${id}.wav.tmp`);
        filePath = join(outDir, `${id}.wav`);
        writeFileSync(tmpPath, wav.buf);
        if (state.cancelled) {
          try { renameSync(tmpPath, filePath); } catch { }
          throw Object.assign(new Error("Cancelación manual de TTS"), { cancelled: true });
        }
        renameSync(tmpPath, filePath);
      }

      return {
        buffer: wav.buf,
        filePath,
        synthText,
        metrics: {
          httpTotalMs,
          headersMs,
          downloadMs,
          audioDurationSec: wav.audioDurationSec,
          rtf,
        },
      };
    } catch (err) {
      if (state.cancelled || err.cancelled) {
        this.metrics.cancellations += 1;
        this.metrics.lastError = "Cancelado manualmente";
        throw Object.assign(new Error("Fish TTS cancelado"), { cancelled: true });
      }
      this.metrics.failures += 1;
      if (state.timedOut) {
        this.metrics.lastError = `Timeout (${this.config.timeoutMs}ms)`;
        throw Object.assign(new Error(this.metrics.lastError), { timedOut: true });
      }
      this.metrics.lastError = err.message;
      throw err;
    } finally {
      clearTimeout(timer);
      if (this._activeRequests.get(source) === state) {
        this._activeRequests.delete(source);
      }
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
