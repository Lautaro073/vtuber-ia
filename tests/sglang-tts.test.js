import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, unlinkSync, writeFileSync, existsSync, renameSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { SGLangTTSClient, validateSglangUrl, validateSglangWav, SGLANG_MAX_INPUT_CHARS } from "../lib/tts/sglang-tts-client.js";
import { FishAudioClient, validateWavBuffer, pinnedRequest } from "../lib/tts/fish-audio-client.js";
import { SglangStreamManager } from "../lib/tts/sglang-stream.js";

const KEY = "sk-sglang-super-secret-123";

// WAV de prueba; por defecto respeta el contrato del gateway (PCM16, mono, 44100 Hz)
function makeWav(durationSec = 0.5, sampleRate = 44100, { declaredExtra = 0, channels = 1, bits = 16, format = 1, riffDelta = 0, dataBytes = null } = {}) {
  const blockAlign = channels * (bits / 8);
  const byteRate = sampleRate * blockAlign;
  const dataSize = dataBytes ?? Math.round(durationSec * sampleRate) * blockAlign;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write("RIFF", 0, "ascii"); buf.writeUInt32LE(36 + dataSize + declaredExtra + riffDelta, 4); buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii"); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(format, 20); buf.writeUInt16LE(channels, 22);
  buf.writeUInt32LE(sampleRate, 24); buf.writeUInt32LE(byteRate, 28); buf.writeUInt16LE(blockAlign, 32); buf.writeUInt16LE(bits, 34);
  buf.write("data", 36, "ascii"); buf.writeUInt32LE(dataSize + declaredExtra, 40);
  return buf;
}

const HEALTH = { status: "ok", model: "fishaudio/s2-pro", voice: "nyra-clonada", busy: false, sample_rate: 44100, channels: 1, encoding: "pcm_s16le" };

// Mock del gateway: misma autenticación y rutas; `mode` cambia la respuesta por test
async function startMock() {
  const state = { mode: "ok", requests: [], inFlight: 0, maxInFlight: 0, speechCalls: 0, delayMs: 0, health: HEALTH };
  const server = createServer((req, res) => {
    const chunks = [];
    req.on("data", c => chunks.push(c));
    req.on("end", async () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      state.requests.push({ method: req.method, url: req.url, headers: req.headers, body: raw ? JSON.parse(raw) : null });
      if (req.headers.authorization !== `Bearer ${KEY}`) { res.writeHead(401, { "Content-Type": "application/json" }); return res.end("{}"); }
      if (req.method === "GET" && req.url === "/v1/health") {
        if (state.mode === "health503") { res.writeHead(503); return res.end(); }
        if (state.mode === "healthHtml") { res.writeHead(200, { "Content-Type": "text/html" }); return res.end("<html>"); }
        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify(state.health));
      }
      if (req.method !== "POST" || req.url !== "/v1/audio/speech") { res.writeHead(404); return res.end(); }
      state.speechCalls += 1;
      const call = state.speechCalls;
      const m = typeof state.mode === "function" ? state.mode(call) : state.mode;
      if (m === "429") { res.writeHead(429, { "Retry-After": state.retryAfter ?? "0" }); return res.end("busy"); }
      if (m === "500") { res.writeHead(500); return res.end("err"); }
      if (m === "redirect") { res.writeHead(302, { Location: state.redirectTo }); return res.end(); }
      state.inFlight += 1;
      state.maxInFlight = Math.max(state.maxInFlight, state.inFlight);
      await new Promise(r => setTimeout(r, state.delayMs));
      state.inFlight -= 1;
      if (res.destroyed) return;
      const wav = makeWav();
      if (m === "custom") { res.writeHead(200, { "Content-Type": state.customType || "audio/wav", "Cache-Control": "no-store" }); return res.end(state.customBody); }
      if (m === "html") { res.writeHead(200, { "Content-Type": "text/html" }); return res.end("<html>error</html>"); }
      if (m === "corrupt") { res.writeHead(200, { "Content-Type": "audio/wav" }); return res.end(Buffer.alloc(200, 7)); }
      if (m === "truncated") { const t = makeWav(0.5, 44100, { declaredExtra: 5000 }); res.writeHead(200, { "Content-Type": "audio/wav" }); return res.end(t); }
      if (m === "cutoff") { res.writeHead(200, { "Content-Type": "audio/wav", "Content-Length": wav.length }); res.write(wav.subarray(0, 1000)); return setTimeout(() => res.destroy(), 20); }
      if (m === "stall") { res.writeHead(200, { "Content-Type": "audio/wav", "Content-Length": wav.length }); res.write(wav.subarray(0, 1000)); return; }
      // Igual que el gateway real: Response(bytes(buf), media_type='audio/wav', headers={'Cache-Control': 'no-store'})
      res.writeHead(200, { "Content-Type": "audio/wav", "Content-Length": wav.length, "Cache-Control": "no-store" });
      res.end(wav);
    });
  });
  await new Promise(r => server.listen(0, "127.0.0.1", r));
  const url = `http://127.0.0.1:${server.address().port}`;
  return { state, url, close: () => new Promise(r => { server.closeAllConnections?.(); server.close(r); }) };
}

function makeClient(url, tmp, extra = {}) {
  const logs = [];
  const client = new SGLangTTSClient({
    config: { url, apiKey: KEY, timeoutMs: 60000, ...extra.config },
    settingsFile: join(tmp, "tts_settings.json"),
    allowLoopbackHttp: true,
    logger: { log: s => logs.push(String(s)) },
    ...extra.opts,
  });
  return { client, logs };
}

const outFiles = dir => (existsSync(dir) ? readdirSync(dir) : []);

let mock;
let tmp;
test.beforeEach(async () => { mock = await startMock(); tmp = mkdtempSync(join(tmpdir(), "nyra-sglang-")); });
test.afterEach(async () => { await mock.close(); rmSync(tmp, { recursive: true, force: true }); });

test("health: disponible, ocupado, 401, 503, respuesta inválida y sin configurar", async () => {
  const { client } = makeClient(mock.url, tmp);
  let h = await client.checkHealth();
  assert.equal(h.ok, true); assert.equal(h.status, "online");
  assert.equal(h.model, "fishaudio/s2-pro"); assert.equal(h.voice, "nyra-clonada"); assert.equal(h.sampleRate, 44100);
  assert.equal(mock.state.requests.at(-1).headers.authorization, `Bearer ${KEY}`);

  mock.state.health = { ...HEALTH, busy: true };
  h = await client.checkHealth();
  assert.equal(h.status, "busy"); assert.equal(h.ok, true, "ocupado no es desconectado");

  client.config.apiKey = "mal";
  h = await client.checkHealth();
  assert.equal(h.status, "unauthorized"); assert.equal(h.ok, false);
  client.config.apiKey = KEY;

  mock.state.mode = "health503";
  assert.equal((await client.checkHealth()).status, "unavailable");
  mock.state.mode = "healthHtml";
  assert.equal((await client.checkHealth()).status, "invalid_response");

  const noKey = makeClient(mock.url, tmp, { config: { apiKey: "" } }).client;
  assert.equal((await noKey.checkHealth()).status, "unconfigured");
  const noUrl = makeClient("", tmp).client;
  assert.equal((await noUrl.checkHealth()).status, "unconfigured");
  assert.equal(noUrl.isConfigured(), false);
  await assert.rejects(noKey.synthesizeWav("hola"), e => e.code === "UNCONFIGURED");
});

test("health: timeout y servicio caído se distinguen", async () => {
  const { client } = makeClient(mock.url, tmp);
  const hang = createServer(() => {});
  await new Promise(r => hang.listen(0, "127.0.0.1", r));
  client.config.url = `http://127.0.0.1:${hang.address().port}`;
  client.config.timeoutMs = 150;
  assert.equal((await client.checkHealth()).status, "timeout");
  hang.closeAllConnections?.(); await new Promise(r => hang.close(r));
  assert.equal((await client.checkHealth()).status, "offline");
});

test("síntesis: Bearer, JSON con solo los 5 campos permitidos, etiqueta emocional y WAV guardado", async () => {
  const { client, logs } = makeClient(mock.url, tmp);
  const out = join(tmp, "out");
  const r = await client.synthesizeWav("¡Muchísimas gracias por la donación!", { emotion: "happy", mood: 90, outDir: out });
  const req = mock.state.requests.at(-1);
  assert.equal(req.headers.authorization, `Bearer ${KEY}`);
  assert.match(req.headers["content-type"], /^application\/json/);
  assert.deepEqual(Object.keys(req.body).sort(), ["input", "model", "response_format", "stream", "voice"]);
  assert.deepEqual(req.body, { model: "fishaudio/s2-pro", voice: "nyra-clonada", input: "[excited] ¡Muchísimas gracias por la donación!", stream: false, response_format: "wav" });
  assert.ok(existsSync(r.filePath));
  assert.deepEqual(outFiles(out).filter(f => f.endsWith(".tmp")), []);
  for (const k of ["headersMs", "firstByteMs", "downloadMs", "validateMs", "writeMs", "totalMs", "audioDurationSec", "rtf", "retries", "queueMs"]) assert.ok(k in r.metrics, k);
  assert.equal(r.metrics.audioDurationSec, 0.5);
  assert.equal(r.metrics.result, "ok");
  assert.ok(logs.some(l => l.includes("📊 TTS sglang")));

  client.updateConfig({ expressive: false });
  await client.synthesizeWav("Hola", { emotion: "happy", mood: 90 });
  assert.equal(mock.state.requests.at(-1).body.input, "Hola");
});

test("síntesis: WAV inválido, truncado, cortado, Content-Type incorrecto y límite de tamaño; sin temporales", async () => {
  const out = join(tmp, "out");
  const { client } = makeClient(mock.url, tmp);
  for (const [mode, re] of [["corrupt", /WAV/], ["truncated", /truncado/], ["cutoff", /incompleta|SGLang/], ["html", /Content-Type/]]) {
    mock.state.mode = mode;
    await assert.rejects(client.synthesizeWav("hola", { outDir: out }), re, mode);
  }
  mock.state.mode = "ok";
  const small = makeClient(mock.url, tmp, { opts: { maxResponseBytes: 1000 } }).client;
  await assert.rejects(small.synthesizeWav("hola", { outDir: out }), /tamaño máximo/);
  assert.deepEqual(outFiles(out), [], "ni .wav ni .tmp tras errores");
  assert.equal(validateWavBuffer(makeWav()).truncated, false);
});

test("texto vacío o mayor a 1000 caracteres falla explícitamente sin pedir al servidor", async () => {
  const { client } = makeClient(mock.url, tmp);
  await assert.rejects(client.synthesizeWav("   "), e => e.code === "EMPTY_TEXT");
  await assert.rejects(client.synthesizeWav("a".repeat(SGLANG_MAX_INPUT_CHARS + 1)), e => e.code === "TEXT_TOO_LONG");
  assert.equal(mock.state.speechCalls, 0);
  await client.synthesizeWav("a".repeat(SGLANG_MAX_INPUT_CHARS));
  assert.equal(mock.state.speechCalls, 1);
});

test("timeout de síntesis", async () => {
  const { client } = makeClient(mock.url, tmp);
  client.config.timeoutMs = 120;
  mock.state.delayMs = 1500;
  await assert.rejects(client.synthesizeWav("hola"), e => e.timedOut === true && !e.cancelled);
});

test("HTTP 429: reintenta y sale bien, agota reintentos, respeta Retry-After y se cancela durante la espera", async () => {
  const { client } = makeClient(mock.url, tmp);
  mock.state.mode = n => (n === 1 ? "429" : "ok");
  const r = await client.synthesizeWav("hola");
  assert.equal(r.metrics.retries, 1);

  mock.state.speechCalls = 0;
  mock.state.mode = "429";
  await assert.rejects(client.synthesizeWav("hola"), e => e.code === "BUSY");
  assert.equal(mock.state.speechCalls, 3, "máximo 3 intentos");

  mock.state.speechCalls = 0;
  mock.state.retryAfter = "1";
  mock.state.mode = n => (n === 1 ? "429" : "ok");
  const t0 = Date.now();
  await client.synthesizeWav("hola");
  assert.ok(Date.now() - t0 >= 900, "esperó Retry-After: 1");

  mock.state.speechCalls = 0;
  mock.state.mode = "429";
  const t1 = Date.now();
  const p = client.synthesizeWav("hola", { source: "live" });
  setTimeout(() => client.abortSource("live"), 100);
  await assert.rejects(p, e => e.cancelled === true);
  assert.ok(Date.now() - t1 < 800, "la espera es abortable");
  assert.equal(mock.state.speechCalls, 1);
});

test("cancelación durante la descarga no deja archivos", async () => {
  const out = join(tmp, "out");
  const { client } = makeClient(mock.url, tmp);
  mock.state.mode = "stall";
  const p = client.synthesizeWav("hola", { source: "live", outDir: out });
  setTimeout(() => client.abortSource("live"), 150);
  await assert.rejects(p, e => e.cancelled === true);
  assert.deepEqual(outFiles(out), []);
});

test("seguridad: HTTPS, hosts permitidos, loopback explícito, IP privada, DNS rebinding y redirecciones", async () => {
  assert.throws(() => validateSglangUrl("http://gw.example.com", { allowedHosts: ["gw.example.com"] }), /HTTPS/);
  assert.throws(() => validateSglangUrl("https://gw.example.com"), /ALLOWED_HOSTS/);
  assert.throws(() => validateSglangUrl("http://127.0.0.1:18081"), /Loopback no autorizado/);
  assert.throws(() => validateSglangUrl("https://gw.example.com/v1/x", { allowedHosts: ["gw.example.com"] }), /sin ruta/);
  assert.equal(validateSglangUrl("https://gw.example.com", { allowedHosts: ["gw.example.com"] }).origin, "https://gw.example.com");

  const mk = (url, allowedHosts, lookupImpl) => new SGLangTTSClient({ config: { url, apiKey: KEY }, settingsFile: join(tmp, "s.json"), allowedHosts, lookupImpl, logger: null });
  const evil = mk("https://evil.example.com", ["gw.example.com"]);
  assert.equal((await evil.checkHealth()).status, "invalid_config");
  await assert.rejects(evil.synthesizeWav("hola"), /lista permitida/);
  assert.throws(() => evil.updateConfig({ url: "https://evil.example.com" }), /lista permitida/);
  await assert.rejects(mk("https://10.0.0.5", ["10.0.0.5"]).synthesizeWav("hola"), /privad/);
  const rebind = mk("https://gw.example.com", ["gw.example.com"], async () => [{ address: "127.0.0.1", family: 4 }]);
  await assert.rejects(rebind.synthesizeWav("hola"), /privada\/reservada/);

  const other = await startMock();
  try {
    const { client } = makeClient(mock.url, tmp);
    mock.state.mode = "redirect";
    mock.state.redirectTo = `${other.url}/v1/audio/speech`;
    await assert.rejects(client.synthesizeWav("hola"), /HTTP 302/);
    assert.equal(other.state.requests.length, 0, "no sigue la redirección");
  } finally { await other.close(); }
});

test("persistencia: sin API key en disco ni en la config pública; convive con Fish y con configuraciones antiguas", () => {
  const settingsFile = join(tmp, "tts_settings.json");
  // Archivo antiguo, como el real: solo motor + fish
  const legacy = { engine: "edge", shareTtsEngine: true, discordEngine: "edge", fish: { url: "https://fish.example.com", timeoutMs: 60000 } };
  writeFileSync(settingsFile, JSON.stringify(legacy));
  const { client } = makeClient("", tmp);
  assert.equal(client.config.url, "");
  assert.equal(client.isConfigured(), false, "sin configurar no rompe el arranque");
  assert.equal(client.config.timeoutMs, 60000);

  client.updateConfig({ url: mock.url, timeoutMs: 30000, fallback: false });
  let disk = JSON.parse(readFileSync(settingsFile, "utf8"));
  assert.equal(disk.engine, "edge"); assert.equal(disk.fish.url, "https://fish.example.com");
  assert.equal(disk.sglang.url, mock.url); assert.equal(disk.sglang.timeoutMs, 30000);
  assert.doesNotMatch(readFileSync(settingsFile, "utf8"), new RegExp(KEY));
  assert.doesNotMatch(JSON.stringify(client.getPublicConfig()), new RegExp(KEY));
  assert.equal(client.getPublicConfig().apiKeyConfigured, true);
  assert.equal(client.getPublicConfig().playbackMode, "wav", "sin migración: WAV por defecto");

  const reloaded = makeClient("", tmp).client;
  assert.equal(reloaded.config.url, mock.url); assert.equal(reloaded.config.timeoutMs, 30000); assert.equal(reloaded.config.fallback, false);

  // Fish guarda su config sin borrar la sección sglang
  const fish = new FishAudioClient({ config: {}, voicesDir: tmp, profilesFile: join(tmp, "p.json"), settingsFile });
  fish.saveEnginePrefs({ engine: "sglang" });
  disk = JSON.parse(readFileSync(settingsFile, "utf8"));
  assert.equal(disk.engine, "sglang"); assert.equal(disk.sglang.url, mock.url); assert.equal(disk.fish.url, "https://fish.example.com");
});

// ─── Integración con TTSEngine (clase real extraída de server.js) ─────────────
function loadTTSEngine(CONFIG, dir, logs) {
  const src = readFileSync(new URL("../server.js", import.meta.url), "utf8");
  const lines = src.split(/\r?\n/);
  const start = lines.findIndex(l => l.startsWith("class TTSEngine"));
  const end = lines.findIndex((l, i) => i > start && l === "}");
  const PatchedSglang = class extends SGLangTTSClient {
    constructor(opts) { super({ ...opts, settingsFile: join(dir, "tts_settings.json"), allowLoopbackHttp: true, logger: { log: s => logs.push(String(s)) } }); }
  };
  const FishStub = class { constructor() { this.enginePrefs = {}; this.config = {}; } abortSource() {} abortAll() {} };
  return new Function("deps", `const { CONFIG, FishAudioClient, SGLangTTSClient, SglangStreamManager, join, __dirname, mkdirSync, renameSync, rmSync, readdirSync, unlinkSync, spawn, createInterface, applyPhonetics, exec, discord } = deps;
    ${lines.slice(start, end + 1).join("\n")}
    return TTSEngine;`)({
    CONFIG, FishAudioClient: FishStub, SGLangTTSClient: PatchedSglang, SglangStreamManager, join, __dirname: dir, mkdirSync, renameSync, rmSync,
    readdirSync, unlinkSync, spawn: () => {}, createInterface: () => ({ on: () => {} }), applyPhonetics: s => s,
    exec: (_c, cb) => cb?.(), discord: { stopAudio: () => {} },
  });
}

function makeEngine(url, { share = true } = {}) {
  const logs = [];
  const CONFIG = {
    ttsEngine: "sglang", shareTtsEngine: share, discordTtsEngine: "edge", useSystemTTS: false, edgeVoice: "x", murf: { apiKey: "" },
    sglang: { url, apiKey: KEY, timeoutMs: 60000 },
  };
  const TTSEngine = loadTTSEngine(CONFIG, tmp, logs);
  const tts = new TTSEngine();
  const edge = { calls: 0 };
  tts._synthEdge = async () => { edge.calls += 1; return "/tts/edge.mp3"; };
  return { tts, edge, logs, CONFIG, ttsDir: join(tmp, "public", "tts") };
}

test("TTSEngine: SGLang seleccionable, WAV integrado a /tts, solicitudes simultáneas serializadas", async () => {
  const { tts, edge, ttsDir } = makeEngine(mock.url);
  mock.state.delayMs = 60;
  const [a, b, c] = await Promise.all([
    tts.synth("Uno", { source: "live", returnMeta: true }),
    tts.synth("Dos", { source: "discord", returnMeta: true }),
    tts.synth("Tres", { source: "live", returnMeta: true }),
  ]);
  assert.equal(mock.state.maxInFlight, 1, "una sola solicitud activa contra el backend");
  for (const r of [a, b, c]) { assert.match(r.url, /^\/tts\/\d+\.wav$/); assert.equal(r.cancelled, false); }
  assert.equal(new Set([a.url, b.url, c.url]).size, 3, "sin audios duplicados");
  assert.ok(existsSync(join(ttsDir, a.url.slice(5))));
  assert.equal(edge.calls, 0);
  // Las etiquetas que escriba el modelo no llegan como texto (clean saca corchetes); la emoción entra como etiqueta TTS
  await tts.synth("[angry] Hola", { emotion: "sad", source: "live" });
  assert.equal(mock.state.requests.at(-1).body.input, "[sad] angry Hola");
});

test("TTSEngine: fallback a Edge ante error, sin fallback ante cancelación, cancelación en cola y respuesta tardía descartada", async () => {
  const { tts, edge, ttsDir } = makeEngine(mock.url);
  mock.state.mode = "500";
  const fb = await tts.synth("Hola", { source: "live", returnMeta: true });
  assert.equal(fb.url, "/tts/edge.mp3"); assert.equal(fb.fallbackUsed, true); assert.equal(edge.calls, 1);
  assert.equal(tts.sglang.metrics.fallbacks, 1);

  mock.state.mode = "ok";
  mock.state.delayMs = 300;
  mock.state.speechCalls = 0;
  const p1 = tts.synth("Activo", { source: "live", returnMeta: true });
  const p2 = tts.synth("En cola", { source: "live", returnMeta: true });
  setTimeout(() => tts.stop("live"), 60);
  const [r1, r2] = await Promise.all([p1, p2]);
  assert.equal(r1.cancelled, true); assert.equal(r1.url, null);
  assert.equal(r2.cancelled, true); assert.equal(r2.url, null);
  assert.equal(mock.state.speechCalls, 1, "el pedido en cola nunca llegó al servidor");
  assert.equal(edge.calls, 1, "cancelar no dispara fallback");

  // Generación invalidada sin abortar el HTTP: el WAV llega tarde, no se reproduce y se borra
  mock.state.delayMs = 150;
  const before = outFiles(ttsDir).length;
  const late = tts.synth("Tardío", { source: "live", returnMeta: true });
  setTimeout(() => tts._cancelEpoch.set("live", (tts._cancelEpoch.get("live") || 0) + 1), 40);
  const lr = await late;
  assert.equal(lr.cancelled, true); assert.equal(lr.url, null);
  assert.equal(outFiles(ttsDir).length, before, "archivo tardío eliminado");

  tts.sglang.updateConfig({ fallback: false });
  mock.state.mode = "500";
  const nofb = await tts.synth("Hola", { source: "live", returnMeta: true });
  assert.equal(nofb.url, null); assert.equal(nofb.fallbackUsed, false); assert.equal(edge.calls, 1);
});

test("TTSEngine: TTL vencido antes de empezar se descarta sin pedir ni hacer fallback", async () => {
  const { tts, edge } = makeEngine(mock.url);
  mock.state.delayMs = 200;
  const first = tts.synth("Largo", { source: "live", returnMeta: true });
  const stale = tts.synth("Comentario viejo", { source: "live", returnMeta: true, expiresAt: Date.now() + 50 });
  const [r1, r2] = await Promise.all([first, stale]);
  assert.ok(r1.url);
  assert.equal(r2.cancelled, true); assert.equal(r2.url, null);
  assert.equal(mock.state.speechCalls, 1);
  assert.equal(edge.calls, 0);
});

test("TTSEngine: Discord independiente (SHARE_TTS_ENGINE=0, DISCORD_TTS_ENGINE=edge) no toca SGLang", async () => {
  const { tts, edge, CONFIG } = makeEngine(mock.url, { share: false });
  assert.equal(tts.resolveEngine("live"), "sglang");
  assert.equal(tts.resolveEngine("discord"), "edge");
  const d = await tts.synth("Hola Discord", { source: "discord", returnMeta: true });
  assert.equal(d.url, "/tts/edge.mp3"); assert.equal(edge.calls, 1);
  assert.equal(mock.state.speechCalls, 0);
  CONFIG.shareTtsEngine = true;
  assert.equal(tts.resolveEngine("discord"), "sglang");
});

test("ningún secreto en logs, errores ni config pública", async () => {
  const captured = [];
  const orig = { log: console.log, error: console.error, warn: console.warn };
  for (const k of Object.keys(orig)) console[k] = (...a) => captured.push(a.join(" "));
  try {
    const { tts, logs } = makeEngine(mock.url);
    mock.state.mode = "500";
    await tts.synth("Hola", { source: "live" });
    mock.state.mode = "ok";
    await tts.synth("Hola", { source: "live" });
    await tts.sglang.checkHealth();
    tts.sglang.updateConfig({ timeoutMs: 20000 }); // fuerza escritura del archivo de ajustes
    tts.sglang.config.apiKey = "otra-clave-mala-999";
    const h = await tts.sglang.checkHealth();
    const all = [...captured, ...logs, JSON.stringify(h), JSON.stringify(tts.sglang.getPublicConfig()), readFileSync(join(tmp, "tts_settings.json"), "utf8")].join("\n");
    assert.doesNotMatch(all, /sk-sglang-super-secret-123|otra-clave-mala-999|Bearer/);
    assert.doesNotMatch(all, /Hola/, "la telemetría no registra el texto");
  } finally { Object.assign(console, orig); }
});

// ─── Ronda de revisión (H01–H04) ──────────────────────────────────────────────
test("H01: prueba del panel pendiente detrás de una síntesis activa se cancela antes de empezar (sin HTTP ni fallback)", async () => {
  const { tts, edge } = makeEngine(mock.url);
  mock.state.delayMs = 250;
  const active = tts.synth("Síntesis del directo", { source: "live", returnMeta: true });
  const panel = tts.sglangJob("Prueba del panel", { source: "panel_test" });
  setTimeout(() => tts.stop("panel_test"), 50);
  const [a, p] = await Promise.allSettled([active, panel]);
  assert.equal(a.status, "fulfilled"); assert.match(a.value.url, /\.wav$/, "live no se ve afectado");
  assert.equal(p.status, "rejected"); assert.equal(p.reason.cancelled, true);
  assert.equal(mock.state.speechCalls, 1, "la prueba del panel nunca llegó al gateway");
  assert.equal(edge.calls, 0);

  // Stop global (botón del panel) cancela activo + pendiente
  mock.state.speechCalls = 0;
  const a2 = tts.synth("Activo", { source: "live", returnMeta: true });
  const p2 = tts.sglangJob("Panel", { source: "panel_test" });
  setTimeout(() => tts.stop(), 50);
  const [ra2, rp2] = await Promise.allSettled([a2, p2]);
  assert.equal(ra2.value.cancelled, true); assert.equal(rp2.reason.cancelled, true);
  assert.equal(mock.state.speechCalls, 1);

  // stop("live") no toca panel_test
  mock.state.speechCalls = 0;
  const a3 = tts.synth("Live a cancelar", { source: "live", returnMeta: true });
  const p3 = tts.sglangJob("Panel sigue", { source: "panel_test" });
  setTimeout(() => tts.stop("live"), 50);
  const [ra3, rp3] = await Promise.allSettled([a3, p3]);
  assert.equal(ra3.value.cancelled, true);
  assert.equal(rp3.status, "fulfilled"); assert.match(rp3.value.url, /\.wav$/);
  assert.equal(edge.calls, 0);
});

test("cancelación que llega antes de registrar el pedido (DNS/backoff) no hace la solicitud HTTP", async () => {
  const { client } = makeClient(mock.url, tmp);
  await assert.rejects(client.synthesizeWav("hola", { shouldCancel: () => true }), e => e.cancelled === true);
  assert.equal(mock.state.speechCalls, 0);
  let cancel = false;
  mock.state.mode = "429";
  mock.state.retryAfter = "0.1";
  setTimeout(() => { cancel = true; }, 30);
  await assert.rejects(client.synthesizeWav("hola", { shouldCancel: () => cancel }), e => e.cancelled === true);
  assert.equal(mock.state.speechCalls, 1, "el reintento posterior a la cancelación no sale");
});

test("TTL que vence durante el backoff de un 429 descarta el pedido sin reintentar ni fallback", async () => {
  const { client } = makeClient(mock.url, tmp);
  mock.state.mode = "429";
  mock.state.retryAfter = "1";
  const t0 = Date.now();
  await assert.rejects(client.synthesizeWav("hola", { expiresAt: Date.now() + 300 }), e => e.expired === true && !e.cancelled);
  assert.ok(Date.now() - t0 < 500, "no espera el Retry-After completo");
  assert.equal(mock.state.speechCalls, 1);
  assert.equal(client.metrics.failures, 0, "vencer no cuenta como fallo");

  const { tts, edge } = makeEngine(mock.url);
  mock.state.speechCalls = 0;
  const r = await tts.synth("viejo", { source: "live", returnMeta: true, expiresAt: Date.now() + 300 });
  assert.equal(r.cancelled, true); assert.equal(r.url, null); assert.equal(edge.calls, 0);
});

test("H02/H03: contrato audio/wav y validación estricta PCM16 mono 44100 (sin endurecer Fish)", async () => {
  assert.ok(validateSglangWav(makeWav()));
  const cases = {
    "estéreo": makeWav(0.1, 44100, { channels: 2 }),
    "frecuencia incorrecta": makeWav(0.1, 22050),
    "profundidad 24 bits": makeWav(0.1, 44100, { bits: 24 }),
    "profundidad 8 bits": makeWav(0.1, 44100, { bits: 8 }),
    "no PCM (float)": makeWav(0.1, 44100, { format: 3 }),
    "WAV vacío (sin muestras)": makeWav(0, 44100, { dataBytes: 0 }),
    "truncado": makeWav(0.1, 44100, { declaredExtra: 400 }),
    "tamaño RIFF inconsistente": makeWav(0.1, 44100, { riffDelta: 100 }),
    "muestras desalineadas": makeWav(0, 44100, { dataBytes: 101 }),
    "solo cabecera": Buffer.alloc(20),
  };
  for (const [name, buf] of Object.entries(cases)) assert.throws(() => validateSglangWav(buf), /WAV/, name);

  // Fish conserva su tolerancia: estéreo / 22050 / 24 bits siguen siendo válidos para validateWavBuffer
  for (const buf of [cases["estéreo"], cases["frecuencia incorrecta"], cases["profundidad 24 bits"]]) assert.doesNotThrow(() => validateWavBuffer(buf));

  const { client } = makeClient(mock.url, tmp);
  const out = join(tmp, "out");
  mock.state.mode = "custom";
  mock.state.customBody = cases["estéreo"];
  await assert.rejects(client.synthesizeWav("hola", { outDir: out }), e => e.code === "BAD_WAV");
  mock.state.customBody = makeWav();
  mock.state.customType = "application/octet-stream";
  await assert.rejects(client.synthesizeWav("hola", { outDir: out }), e => e.code === "BAD_CONTENT_TYPE");
  mock.state.customType = "audio/wav; charset=binary";
  assert.ok((await client.synthesizeWav("hola", { outDir: out })).filePath, "parámetros MIME se ignoran");
  assert.deepEqual(outFiles(out).filter(f => f.endsWith(".tmp")), []);
});

test("H04: helpers compartidos siguen compatibles con los llamados de Fish", async () => {
  const target = { protocol: "http:", hostname: "127.0.0.1", port: new URL(mock.url).port, pinnedAddress: "127.0.0.1", pinnedFamily: 4 };
  const auth = { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" };
  const body = JSON.stringify({ model: "m", voice: "v", input: "x", stream: false, response_format: "wav" });

  // Llamado estilo Fish (sin maxBytes/onFirstChunk): límite por defecto, mismo texto de error, sin .get() en headersRaw
  const res = await pinnedRequest(target, "/v1/audio/speech", { method: "POST", headers: auth, body });
  assert.equal(res.status, 200);
  assert.equal(typeof res.headersRaw.get, "undefined", "Fish no ve Retry-After nuevo: su backoff no cambia");
  assert.ok((await res.arrayBuffer()).byteLength > 44);
  const small = await pinnedRequest(target, "/v1/audio/speech", { method: "POST", headers: auth, body, maxBytes: 100 });
  await assert.rejects(small.arrayBuffer(), /WAV excede el tamaño máximo permitido \(100 bytes\)/);

  // Respuesta cortada antes de Content-Length: rechaza en vez de quedar colgada o resolver a medias
  mock.state.mode = "cutoff";
  const cut = await pinnedRequest(target, "/v1/audio/speech", { method: "POST", headers: auth, body });
  await assert.rejects(cut.arrayBuffer());

  // AbortController durante la descarga
  mock.state.mode = "stall";
  const ac = new AbortController();
  const st = await pinnedRequest(target, "/v1/audio/speech", { method: "POST", headers: auth, body, signal: ac.signal });
  setTimeout(() => ac.abort(new Error("stop")), 30);
  await assert.rejects(st.arrayBuffer());

  // Backoff de Fish ante 429 sigue siendo attempt*400 aunque el server mande Retry-After: 2
  const fishSrv = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      fishSrv.calls = (fishSrv.calls || 0) + 1;
      if (fishSrv.calls === 1) { res.writeHead(429, { "Retry-After": "2" }); return res.end(); }
      res.writeHead(200, { "Content-Type": "audio/wav" }); res.end(makeWav());
    });
  });
  await new Promise(r => fishSrv.listen(0, "127.0.0.1", r));
  try {
    const fish = new FishAudioClient({ config: { url: `http://127.0.0.1:${fishSrv.address().port}`, apiKey: "k", timeoutMs: 10000 }, voicesDir: tmp, profilesFile: join(tmp, "p.json"), settingsFile: join(tmp, "s.json") });
    const t0 = Date.now();
    await fish.synthesizeWav("hola");
    const waited = Date.now() - t0;
    assert.ok(waited >= 350 && waited < 1500, `Fish esperó ${waited}ms (attempt*400, no Retry-After)`);

    // _persistSettings: conserva claves ajenas y escribe sin la API key
    writeFileSync(join(tmp, "s.json"), JSON.stringify({ engine: "edge", sglang: { url: "https://gw.example.com" }, otro: 1 }));
    fish.updateConfig({ temperature: 0.8 });
    const disk = JSON.parse(readFileSync(join(tmp, "s.json"), "utf8"));
    assert.deepEqual(disk.sglang, { url: "https://gw.example.com" }); assert.equal(disk.otro, 1); assert.equal(disk.fish.temperature, 0.8);
    assert.equal("apiKey" in disk.fish, false);
  } finally { fishSrv.closeAllConnections?.(); await new Promise(r => fishSrv.close(r)); }
});
