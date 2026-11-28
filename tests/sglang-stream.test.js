// Fase 2: streaming PCM de SGLang. Mock del gateway con patrones temporales, vistas WS falsas, núcleo de reproducción
// y política de búfer con reloj simulado. Nada de esto toca servicios reales.
import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, renameSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { SGLangTTSClient } from "../lib/tts/sglang-tts-client.js";
import { SglangStreamManager, encodeFrame, decodeFrame, pcmToWav, FRAME_HEADER_BYTES } from "../lib/tts/sglang-stream.js";
import { validateWavBuffer } from "../lib/tts/fish-audio-client.js";
import { PcmPlaybackCore } from "../public/tts-stream/pcm-core.js";
import { AdaptiveStartPolicy, FixedStartPolicy, WholeFilePolicy, simulatePlayback, oracleStartSec } from "../public/tts-stream/buffer-policy.js";

const KEY = "sk-stream-secret-777";
const PCM_HEADERS = { "Content-Type": "application/octet-stream", "X-Audio-Sample-Rate": "44100", "X-Audio-Channels": "1", "X-Audio-Encoding": "pcm_s16le", "Cache-Control": "no-store", "X-Accel-Buffering": "no" };

// PCM sintético (seno), NO es audio real de la voz
function sinePcm(samples, freq = 220, amp = 0.5, phase0 = 0) {
  const b = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i++) b.writeInt16LE(Math.round(Math.sin(2 * Math.PI * freq * (i + phase0) / 44100) * amp * 32767), i * 2);
  return b;
}
function wavOf(samples) { return pcmToWav([sinePcm(samples)]); }
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Gateway mock: state.plan = { status, headers, parts:[[delayMs, Buffer]], end:"ok"|"cut"|"stall" }
async function startGateway() {
  const state = { requests: [], speechCalls: 0, plan: null, wavPlan: null, inFlight: 0, maxInFlight: 0, n429: 0 };
  const server = createServer((req, res) => {
    const chunks = [];
    req.on("data", c => chunks.push(c));
    req.on("end", async () => {
      const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : null;
      state.requests.push({ headers: req.headers, body });
      if (req.headers.authorization !== `Bearer ${KEY}`) { res.writeHead(401); return res.end(); }
      if (req.url !== "/v1/audio/speech") { res.writeHead(404); return res.end(); }
      state.speechCalls += 1;
      if (state.n429 > 0) { state.n429 -= 1; res.writeHead(429, { "Retry-After": "0" }); return res.end(); }
      state.inFlight += 1; state.maxInFlight = Math.max(state.maxInFlight, state.inFlight);
      try {
        if (!body.stream) {
          await sleep(state.wavPlan?.delayMs ?? 20);
          const w = wavOf(state.wavPlan?.samples ?? 22050);
          res.writeHead(200, { "Content-Type": "audio/wav", "Content-Length": w.length }); res.end(w); return;
        }
        const plan = state.plan;
        if (plan.headerDelayMs) await sleep(plan.headerDelayMs);
        if (res.destroyed) return;
        if (plan.status && plan.status !== 200) { res.writeHead(plan.status); return res.end(); }
        res.writeHead(200, plan.headers || PCM_HEADERS);
        for (const [delay, buf] of plan.parts) {
          await sleep(delay);
          if (res.destroyed) return;
          res.write(buf);
        }
        if (plan.end === "cut") { await sleep(10); res.destroy(); return; }
        if (plan.end === "stall") return; // nunca termina
        res.end();
      } finally { state.inFlight -= 1; }
    });
  });
  await new Promise(r => server.listen(0, "127.0.0.1", r));
  return { state, url: `http://127.0.0.1:${server.address().port}`, close: () => new Promise(r => { server.closeAllConnections?.(); server.close(r); }) };
}

let gw, tmp;
test.beforeEach(async () => { gw = await startGateway(); tmp = mkdtempSync(join(tmpdir(), "nyra-stream-")); });
test.afterEach(async () => { await gw.close(); rmSync(tmp, { recursive: true, force: true }); });

function makeClient(extra = {}) {
  const logs = [];
  const client = new SGLangTTSClient({
    config: { url: gw.url, apiKey: KEY, timeoutMs: 60000, ...extra.config },
    settingsFile: join(tmp, "tts_settings.json"), allowLoopbackHttp: true,
    logger: { log: s => logs.push(String(s)), warn: s => logs.push(String(s)) }, ...extra.opts,
  });
  return { client, logs };
}

const pcm = (n, phase = 0) => sinePcm(n, 220, 0.5, phase);

// ─── Cliente HTTP de streaming ────────────────────────────────────────────────
test("1/2: contrato de streaming (JSON stream:true/pcm) y validación de cabeceras antes de cualquier PCM", async () => {
  const { client } = makeClient();
  gw.state.plan = { parts: [[0, pcm(4410)], [120, pcm(4410, 4410)]], end: "ok" };
  const events = [];
  const m = await client.synthesizeStream("Hola chat", {
    emotion: "happy", mood: 90,
    onHeaders: () => events.push(["headers", Date.now()]),
    onPcm: (b, info) => events.push(["pcm", Date.now(), b.length, info.seq]),
  });
  const req = gw.state.requests.at(-1);
  assert.deepEqual(req.body, { model: "fishaudio/s2-pro", voice: "nyra-clonada", input: "[excited] Hola chat", stream: true, response_format: "pcm" });
  assert.equal(req.headers.authorization, `Bearer ${KEY}`);
  assert.equal(events[0][0], "headers");
  const pcmEv = events.filter(e => e[0] === "pcm");
  assert.ok(pcmEv.length >= 2, "llega progresivo, no de una vez");
  assert.ok(pcmEv.at(-1)[1] - pcmEv[0][1] >= 80, "el primer PCM se entrega antes de que termine la descarga");
  assert.equal(m.samples, 8820); assert.equal(m.result, "ok"); assert.equal(m.mode, "stream");

  for (const [headers, re] of [
    [{ ...PCM_HEADERS, "Content-Type": "audio/wav" }, /Content-Type/],
    [{ ...PCM_HEADERS, "X-Audio-Sample-Rate": "48000" }, /cabeceras de audio/],
    [{ ...PCM_HEADERS, "X-Audio-Channels": "2" }, /cabeceras de audio/],
    [{ ...PCM_HEADERS, "X-Audio-Encoding": "pcm_f32le" }, /cabeceras de audio/],
  ]) {
    let got = 0;
    gw.state.plan = { headers, parts: [[0, pcm(100)]], end: "ok" };
    await assert.rejects(client.synthesizeStream("hola", { onPcm: () => { got += 1; } }), e => re.test(e.message) && !e.partial);
    assert.equal(got, 0, "ningún PCM con cabeceras inválidas");
  }
});

test("3: chunks HTTP de tamaño impar se realinean a muestras completas sin perder ni inventar bytes", async () => {
  const { client } = makeClient();
  const all = pcm(5000);
  const cuts = [1001, 999, 3, 1, 2, 1995, 5999];
  const parts = []; let off = 0;
  for (const c of cuts) { parts.push([5, all.subarray(off, off + c)]); off += c; }
  parts.push([5, all.subarray(off)]);
  gw.state.plan = { parts, end: "ok" };
  const got = [];
  await client.synthesizeStream("hola", { onPcm: b => { assert.equal(b.length % 2, 0); got.push(b); } });
  assert.ok(Buffer.concat(got).equals(all));
});

test("4/23: conversión Int16→Float32 exacta a 44100 y remuestreo continuo a 48000 sin cambiar tono ni duración", () => {
  const core = new PcmPlaybackCore(44100, 44100, { fadeMs: 0.01 });
  const src = new Int16Array([0, 16384, -16384, 32767, -32768, 100]);
  core.push(src); core.end(); core.play();
  const out = new Float32Array(10);
  core.render(out);
  assert.deepEqual([...out.slice(1, 6)].map(v => +v.toFixed(5)), [0.5, -0.5, 0.99997, -1, 0.00305].map(v => +v.toFixed(5)));
  assert.equal(core.consumed, src.length);

  // 1 s de seno de 220 Hz en 7 chunks de tamaños raros → 48000 Hz
  const r = new PcmPlaybackCore(48000, 44100);
  const buf = pcm(44100);
  const i16 = new Int16Array(buf.buffer, buf.byteOffset, 44100);
  let o = 0;
  for (const n of [1, 999, 4410, 3, 12000, 20000, 6687]) { r.push(i16.slice(o, o + n)); o += n; }
  r.end(); r.play();
  const outs = [];
  for (let k = 0; k < 400 && r.state !== "drained"; k++) { const b = new Float32Array(128); r.render(b); outs.push(b); }
  const y = Float32Array.from(outs.flatMap(b => [...b]));
  assert.equal(r.consumed, 44100, "todas las muestras de entrada se reprodujeron");
  const nonzero = y.findLastIndex(v => v !== 0) + 1;
  assert.ok(Math.abs(nonzero - 48000) <= 130, `duración preservada (~48000 muestras, fue ${nonzero})`);
  let crossings = 0;
  for (let i = 1; i < nonzero; i++) if (y[i - 1] < 0 && y[i] >= 0) crossings++;
  assert.ok(Math.abs(crossings - 220) <= 1, `tono preservado (220 cruces, fue ${crossings})`);
  let maxJump = 0;
  for (let i = 200; i < nonzero - 200; i++) maxJump = Math.max(maxJump, Math.abs(y[i] - y[i - 1]));
  assert.ok(maxJump < 0.02, `sin discontinuidades entre chunks (salto máx ${maxJump.toFixed(4)})`);
});

test("5/6: PCM incompleto, cortes antes/después del primer PCM y límite de memoria", async () => {
  const { client } = makeClient();
  gw.state.plan = { parts: [[0, pcm(1000)], [5, Buffer.from([1, 2, 3])]], end: "ok" };
  await assert.rejects(client.synthesizeStream("hola"), e => e.code === "BAD_PCM" && e.partial === true);
  gw.state.plan = { parts: [], end: "cut" };
  await assert.rejects(client.synthesizeStream("hola"), e => !e.partial && !e.cancelled);
  gw.state.plan = { parts: [[0, pcm(1000)]], end: "cut" };
  await assert.rejects(client.synthesizeStream("hola"), e => e.partial === true && /incompleta|conexión/i.test(e.message));
  const small = makeClient({ opts: { maxResponseBytes: 3000 } }).client;
  gw.state.plan = { parts: [[0, pcm(1000)], [5, pcm(1000)]], end: "ok" };
  await assert.rejects(small.synthesizeStream("hola"), e => e.code === "TOO_LARGE" && e.partial === true);
  gw.state.plan = { parts: [], end: "ok" };
  await assert.rejects(client.synthesizeStream("hola"), e => e.code === "EMPTY_PCM");
});

test("8: timeout de cabeceras y de inactividad durante la descarga", async () => {
  const { client } = makeClient();
  client.config.timeoutMs = 100;
  gw.state.plan = { headerDelayMs: 600, parts: [[0, pcm(10)]], end: "ok" };
  await assert.rejects(client.synthesizeStream("hola"), e => e.timedOut && /cabeceras/.test(e.message));
  // inactividad: llegó un bloque y después el gateway se colgó
  const idle = makeClient({ opts: { streamIdleMs: 150 } }).client;
  gw.state.plan = { parts: [[0, pcm(1000)]], end: "stall" };
  const t0 = Date.now();
  await assert.rejects(idle.synthesizeStream("hola"), e => e.timedOut && e.partial === true && /sin datos/.test(e.message));
  assert.ok(Date.now() - t0 < 1000);
  assert.equal(idle._active.size, 0);
});

test("9/10: cancelación durante DNS (no hace HTTP) y durante la descarga", async () => {
  let resolveDns;
  const slowDns = () => new Promise(r => { resolveDns = () => r([{ address: "93.184.216.34", family: 4 }]); });
  const c = new SGLangTTSClient({ config: { url: "https://gw.example.com", apiKey: KEY }, settingsFile: join(tmp, "s.json"), allowedHosts: ["gw.example.com"], lookupImpl: slowDns, logger: null });
  let cancelled = false;
  const p = c.synthesizeStream("hola", { shouldCancel: () => cancelled });
  await sleep(20); cancelled = true; resolveDns();
  await assert.rejects(p, e => e.cancelled === true);

  const { client } = makeClient();
  gw.state.plan = { parts: [[0, pcm(2000)], [2000, pcm(2000)]], end: "ok" };
  let got = 0;
  const p2 = client.synthesizeStream("hola", { source: "live", onPcm: () => { got += 1; } });
  await sleep(150);
  client.abortSource("live");
  await assert.rejects(p2, e => e.cancelled === true);
  assert.equal(got, 1);
  assert.equal(client._active.size, 0, "sin pedidos colgados");
});

test("HTTP 429 en streaming reintenta y sigue", async () => {
  const { client } = makeClient();
  gw.state.n429 = 1;
  gw.state.plan = { parts: [[0, pcm(500)]], end: "ok" };
  const m = await client.synthesizeStream("hola");
  assert.equal(m.retries, 1); assert.equal(gw.state.speechCalls, 2);
});

// ─── Sesiones y TTSEngine ─────────────────────────────────────────────────────
function fakeWs(opts = {}) {
  return { readyState: 1, ttsStream: true, bufferedAmount: 0, sent: [], ...opts };
}
const json = ws => ws.sent.filter(m => typeof m === "string").map(m => JSON.parse(m));
const frames = ws => ws.sent.filter(m => Buffer.isBuffer(m)).map(decodeFrame);

function loadTTSEngine(CONFIG, dir, logs) {
  const src = readFileSync(new URL("../server.js", import.meta.url), "utf8");
  const lines = src.split(/\r?\n/);
  const start = lines.findIndex(l => l.startsWith("class TTSEngine"));
  const end = lines.findIndex((l, i) => i > start && l === "}");
  const PatchedSglang = class extends SGLangTTSClient {
    constructor(opts) { super({ ...opts, settingsFile: join(dir, "tts_settings.json"), allowLoopbackHttp: true, logger: { log: s => logs.push(String(s)), warn: s => logs.push(String(s)) } }); }
  };
  const PatchedStreams = class extends SglangStreamManager {
    constructor(opts) { super({ ...opts, logger: { log: s => logs.push(String(s)), warn: s => logs.push(String(s)) } }); }
  };
  const FishStub = class { constructor() { this.enginePrefs = {}; this.config = {}; } abortSource() {} abortAll() {} };
  return new Function("deps", `const { CONFIG, FishAudioClient, SGLangTTSClient, SglangStreamManager, join, __dirname, mkdirSync, renameSync, rmSync, readdirSync, unlinkSync, spawn, createInterface, applyPhonetics, exec, discord } = deps;
    ${lines.slice(start, end + 1).join("\n")}
    return TTSEngine;`)({
    CONFIG, FishAudioClient: FishStub, SGLangTTSClient: PatchedSglang, SglangStreamManager: PatchedStreams, join, __dirname: dir, mkdirSync, renameSync, rmSync,
    readdirSync, unlinkSync, spawn: () => {}, createInterface: () => ({ on: () => {} }), applyPhonetics: s => s,
    exec: (_c, cb) => cb?.(), discord: { stopAudio: () => {} },
  });
}

function makeEngine({ mode = "stream", clients = [fakeWs()], share = true } = {}) {
  const logs = [];
  const CONFIG = {
    ttsEngine: "sglang", shareTtsEngine: share, discordTtsEngine: "edge", useSystemTTS: false, edgeVoice: "x", murf: { apiKey: "" },
    sglang: { url: gw.url, apiKey: KEY, timeoutMs: 60000, playbackMode: mode },
  };
  mkdirSync(join(tmp, "public", "tts"), { recursive: true });
  const TTSEngine = loadTTSEngine(CONFIG, tmp, logs);
  const tts = new TTSEngine();
  const ctx = { tts, logs, CONFIG, clients, edge: 0, plays: [], released: 0 };
  tts.streams.hub = { clients: () => ctx.clients, send: (ws, d) => ws.sent.push(d) };
  tts._broadcast = msg => { if (msg.type === "tts_play") ctx.plays.push(msg.url); };
  tts._onDone = () => { ctx.released += 1; };
  tts._synthEdge = async () => { ctx.edge += 1; return "/tts/edge.mp3"; };
  return ctx;
}
const until = async (cond, ms = 3000) => { const t0 = Date.now(); while (!cond()) { if (Date.now() - t0 > ms) throw new Error("timeout esperando condición"); await sleep(5); } };

test("14/15/16: ids y secuencias, sin tts_play duplicado, y fin de transmisión ≠ fin de reproducción", async () => {
  const e = makeEngine();
  const ws = e.clients[0];
  gw.state.plan = { parts: [[0, pcm(4410)], [60, pcm(4410)], [60, pcm(4410)]], end: "ok" };
  const r = await e.tts.synth("Hola chat, todo bien?", { source: "live", returnMeta: true });
  assert.match(r.url, /^stream:\d+$/);
  await e.tts.play(r.url, "Hola chat, todo bien?");
  await until(() => json(ws).some(m => m.type === "tts_stream_end"));
  const msgs = json(ws);
  assert.equal(msgs[0].type, "tts_stream_start");
  assert.equal(msgs[0].sampleRate, 44100); assert.equal(msgs[0].format, "pcm_s16le"); assert.equal(msgs[0].v, 1);
  assert.ok(msgs[0].estDurationSec > 0);
  const fr = frames(ws);
  assert.deepEqual(fr.map(f => f.seq), fr.map((_, i) => i), "secuencia 0..n sin huecos");
  assert.ok(fr.every(f => f.id === msgs[0].id && f.pcm.length % 2 === 0));
  assert.equal(fr.reduce((s, f) => s + f.pcm.length / 2, 0), 13230);
  assert.equal(msgs.find(m => m.type === "tts_stream_end").totalSamples, 13230);
  assert.deepEqual(e.plays, [], "ningún tts_play del mismo audio");
  assert.equal(e.tts.speaking, true, "terminó la transmisión, no la reproducción: el turno sigue tomado");
  assert.equal(e.released, 0);
  // la vista avisa que terminó de sonar
  e.tts.currentUrl = null; e.tts.audioEnded(); // equivalente al handler tts_done con url coincidente
  assert.equal(e.released, 1);
  assert.equal(e.tts.streams.streams.size, 0, "sesión liberada");
});

test("precarga: el PCM espera en el servidor hasta el turno y llega con ageMs; chunks tardíos tras cancelar se ignoran", async () => {
  const e = makeEngine();
  const ws = e.clients[0];
  gw.state.plan = { parts: [[0, pcm(2000)], [40, pcm(2000)]], end: "ok" };
  const r = await e.tts.synth("Precargado", { source: "live", returnMeta: true });
  const st = e.tts.streams.get(r.url);
  await st.settled;
  await sleep(60);
  assert.equal(ws.sent.length, 0, "nada sale antes de play()");
  await e.tts.play(r.url, "Precargado");
  const fr = frames(ws);
  assert.equal(fr.length, 2);
  assert.ok(fr[0].ageMs >= 50, "el navegador sabe cuánto esperó cada chunk");
  // 13: un chunk que llegue después de cancelar no se reenvía
  e.tts.stop("live");
  e.tts.streams.onPcm(st, pcm(100), {});
  assert.equal(frames(ws).length, 2);
});

test("11/12: Stop durante la reproducción (transmisión ya terminada) y cancelación de pendientes sin HTTP", async () => {
  const e = makeEngine();
  const ws = e.clients[0];
  gw.state.plan = { parts: [[0, pcm(3000)]], end: "ok" };
  const r = await e.tts.synth("Uno", { source: "live", returnMeta: true });
  await e.tts.play(r.url, "Uno");
  await until(() => json(ws).some(m => m.type === "tts_stream_end"));
  e.tts.stop();
  assert.ok(json(ws).some(m => m.type === "tts_stream_cancel" && m.reason === "stop"), "Stop corta en la vista aunque ya no se genere");
  assert.equal(e.released, 1);
  assert.equal(e.tts.streams.streams.size, 0);

  // Stop DURANTE la transmisión (generando y sonando): corta la vista, aborta el HTTP y libera el turno
  gw.state.plan = { parts: [[0, pcm(2000)], [3000, pcm(2000)]], end: "ok" };
  const mid = await e.tts.synth("Larga", { source: "live", returnMeta: true });
  await e.tts.play(mid.url, "Larga");
  await until(() => frames(ws).some(f => f.id === Number(mid.url.slice(7))));
  e.tts.stop();
  assert.ok(json(ws).some(m => m.type === "tts_stream_cancel" && m.id === Number(mid.url.slice(7))));
  assert.equal(e.released, 2);
  await until(() => e.tts.sglang._active.size === 0, 1000);

  // pendiente detrás de una generación larga: Stop antes de que empiece → nunca llega al gateway
  gw.state.speechCalls = 0;
  gw.state.plan = { parts: [[0, pcm(1000)], [300, pcm(1000)]], end: "ok" };
  const a = e.tts.synth("Activo", { source: "live", returnMeta: true });
  const b = e.tts.synth("Pendiente", { source: "live", returnMeta: true });
  await sleep(60);
  e.tts.stop("live");
  const [ra, rb] = await Promise.all([a, b]);
  assert.match(ra.url, /^stream:/); // ya tenía cabeceras cuando llegó el Stop
  assert.equal(rb.cancelled, true);
  await sleep(50);
  assert.equal(gw.state.speechCalls, 1, "el pendiente nunca hizo HTTP");
  assert.equal(e.edge, 0, "cancelar no dispara fallback");
  assert.equal(e.tts.sglang._active.size, 0);
});

test("17: falla antes de sonar → fallback seguro (antes de cabeceras, durante la precarga y ya en turno sin audio enviado)", async () => {
  const e = makeEngine();
  const ws = e.clients[0];
  gw.state.plan = { status: 500, parts: [], end: "ok" };
  const r1 = await e.tts.synth("Antes de cabeceras", { source: "live", returnMeta: true });
  assert.equal(r1.url, "/tts/edge.mp3"); assert.equal(r1.fallbackUsed, true);

  gw.state.plan = { parts: [[0, pcm(500)]], end: "cut" };
  const r2 = await e.tts.synth("Falla en precarga", { source: "live", returnMeta: true });
  await e.tts.streams.get(r2.url).settled;
  await e.tts.play(r2.url, "Falla en precarga");
  await until(() => e.plays.includes("/tts/edge.mp3"));
  assert.equal(frames(ws).length, 0, "el PCM parcial nunca salió hacia la vista");
  assert.equal(e.edge, 2);

  // en turno, con tts_stream_start enviado pero sin ningún PCM todavía
  e.plays.length = 0;
  gw.state.plan = { headerDelayMs: 0, parts: [[150, Buffer.alloc(0)]], end: "cut" };
  const r3 = await e.tts.synth("Sin audio", { source: "live", returnMeta: true });
  await e.tts.play(r3.url, "Sin audio");
  await until(() => e.plays.includes("/tts/edge.mp3"));
  assert.ok(json(ws).some(m => m.type === "tts_stream_cancel" && m.reason === "error"));
  assert.equal(e.edge, 3);
});

test("Stop mientras se genera el fallback de un stream fallido: el fallback no suena", async () => {
  const e = makeEngine();
  e.tts._synthEdge = async () => { e.edge += 1; await sleep(120); return "/tts/edge-tarde.mp3"; };
  gw.state.plan = { parts: [[0, pcm(500)]], end: "cut" };
  const r = await e.tts.synth("Falla", { source: "live", returnMeta: true });
  await e.tts.streams.get(r.url).settled;
  const p = e.tts.play(r.url, "Falla");
  await sleep(30);
  e.tts.stop();
  await p; await sleep(150);
  assert.equal(e.edge, 1);
  assert.deepEqual(e.plays, [], "nada suena después del Stop");
});

test("18/19: falla después de enviar audio → fin parcial, sin fallback ni repetición, registrada", async () => {
  const e = makeEngine();
  const ws = e.clients[0];
  gw.state.plan = { parts: [[0, pcm(4410)], [80, pcm(4410)]], end: "cut" };
  const r = await e.tts.synth("Se corta a la mitad", { source: "live", returnMeta: true });
  await e.tts.play(r.url, "Se corta a la mitad");
  await until(() => json(ws).some(m => m.type === "tts_stream_end"));
  const endMsg = json(ws).find(m => m.type === "tts_stream_end");
  assert.equal(endMsg.partial, true);
  assert.equal(endMsg.totalSamples, frames(ws).reduce((s, f) => s + f.pcm.length / 2, 0));
  assert.equal(e.edge, 0); assert.deepEqual(e.plays, []);
  assert.equal(e.tts.streams.lastStats.server.partialFailure, true);
  assert.ok(e.logs.some(l => /falla parcial/.test(l)));
  assert.equal(e.tts.speaking, true, "la vista termina lo recibido y avisa tts_done");
});

test("24/25: Discord y vistas sin soporte usan WAV; sin vistas compatibles al llegar el turno se convierte el mismo PCM", async () => {
  const e = makeEngine({ share: true });
  gw.state.plan = { parts: [[0, pcm(1000)]], end: "ok" };
  const d = await e.tts.synth("Para Discord", { source: "discord", returnMeta: true });
  assert.match(d.url, /^\/tts\/\d+\.wav$/, "Discord sigue con WAV");
  assert.equal(gw.state.requests.at(-1).body.stream, false);

  const wavMode = makeEngine({ mode: "wav" });
  const w = await wavMode.tts.synth("Modo WAV", { source: "live", returnMeta: true });
  assert.match(w.url, /^\/tts\/\d+\.wav$/);

  const noViews = makeEngine({ clients: [fakeWs({ ttsStream: false })] });
  const nv = await noViews.tts.synth("Vista vieja", { source: "live", returnMeta: true });
  assert.match(nv.url, /\.wav$/, "sin vistas compatibles: WAV desde el principio");

  // compatible al sintetizar, pero se fue antes del turno: mismo PCM → WAV, sin otra síntesis
  const gone = makeEngine();
  gw.state.speechCalls = 0;
  gw.state.plan = { parts: [[0, pcm(3000)], [30, pcm(3000)]], end: "ok" };
  const g = await gone.tts.synth("Se fue la vista", { source: "live", returnMeta: true });
  gone.clients = [];
  await gone.tts.play(g.url, "Se fue la vista");
  assert.equal(gone.plays.length, 1);
  const wav = readFileSync(join(tmp, "public", gone.plays[0]));
  assert.equal(validateWavBuffer(wav).dataSize, 12000);
  assert.equal(gw.state.speechCalls, 1, "no se sintetizó de nuevo");
});

test("SHARE_TTS_ENGINE=0 + DISCORD_TTS_ENGINE=edge: Discord no toca SGLang aunque el directo use streaming", async () => {
  const e = makeEngine({ share: false });
  gw.state.speechCalls = 0;
  const r = await e.tts.synth("Hola Discord", { source: "discord", returnMeta: true });
  assert.equal(r.url, "/tts/edge.mp3"); assert.equal(gw.state.speechCalls, 0);
});

test("7/27: vista lenta se saca sin frenar a las demás; si se va la última, se libera el turno y se aborta la generación", async () => {
  const slow = fakeWs({ bufferedAmount: 10 * 1024 * 1024 });
  const ok = fakeWs();
  const e = makeEngine({ clients: [slow, ok] });
  gw.state.plan = { parts: [[0, pcm(2000)], [40, pcm(2000)], [2000, pcm(2000)]], end: "ok" };
  const r = await e.tts.synth("Dos vistas", { source: "live", returnMeta: true });
  await e.tts.play(r.url, "Dos vistas");
  await until(() => frames(ok).length >= 2);
  assert.equal(frames(slow).length, 0);
  assert.ok(json(slow).some(m => m.type === "tts_stream_cancel" && m.reason === "slow"));
  // reconexión/desconexión: se va la única vista que quedaba
  e.tts.streamClientGone(ok);
  assert.equal(e.released, 1, "turno liberado sin esperar el timeout");
  await until(() => e.tts.sglang._active.size === 0);
  assert.equal(e.tts.streams.streams.size, 0);
  // una vista nueva recibe la siguiente sesión normalmente
  const fresh = fakeWs();
  e.clients = [fresh];
  gw.state.plan = { parts: [[0, pcm(500)]], end: "ok" };
  const r2 = await e.tts.synth("Después de reconectar", { source: "live", returnMeta: true });
  await e.tts.play(r2.url, "Después de reconectar");
  await until(() => json(fresh).some(m => m.type === "tts_stream_end"));
  assert.equal(frames(fresh).length, 1);
});

test("28/29: sin secretos en mensajes, frames ni logs; recursos liberados", async () => {
  const e = makeEngine();
  const ws = e.clients[0];
  gw.state.plan = { parts: [[0, pcm(1000)]], end: "ok" };
  const r = await e.tts.synth("Hola", { source: "live", returnMeta: true });
  await e.tts.play(r.url, "Hola");
  await until(() => json(ws).some(m => m.type === "tts_stream_end"));
  e.tts.streams.recordClientStats({ id: json(ws)[0].id, stats: { audibleStartMs: 1234, underruns: 0, bufferAtStartSec: 1.2, contextRate: 48000, result: "done", extra: KEY } });
  e.tts.audioEnded();
  const everything = [...ws.sent.map(m => (Buffer.isBuffer(m) ? m.subarray(0, FRAME_HEADER_BYTES).toString("latin1") : m)), ...e.logs, JSON.stringify(e.tts.sglang.getPublicConfig())].join("\n");
  assert.doesNotMatch(everything, new RegExp(KEY));
  assert.doesNotMatch(everything, /Bearer/);
  const pub = e.tts.sglang.getPublicConfig();
  assert.equal(pub.stream.client.audibleStartMs, 1234);
  assert.equal(pub.stream.client.extra, undefined, "solo campos numéricos conocidos");
  assert.equal(e.tts.streams.streams.size, 0);
  assert.equal(e.tts.sglang._active.size, 0);
  assert.deepEqual(readdirSync(join(tmp, "public", "tts")).filter(f => f.endsWith(".tmp")), []);
});

test("30: configuraciones anteriores siguen en WAV; el modo se elige a mano y se valida", () => {
  writeFileSync(join(tmp, "tts_settings.json"), JSON.stringify({ engine: "sglang", sglang: { url: gw.url, timeoutMs: 30000 } }));
  const { client } = makeClient({ config: {} });
  assert.equal(client.config.playbackMode, "wav", "sin migración automática");
  assert.throws(() => client.updateConfig({ playbackMode: "turbo" }), /inválido/);
  client.updateConfig({ playbackMode: "stream" });
  assert.equal(JSON.parse(readFileSync(join(tmp, "tts_settings.json"), "utf8")).sglang.playbackMode, "stream");
  client.updateConfig({ playbackMode: "wav" });
  assert.equal(makeClient({ config: {} }).client.config.playbackMode, "wav", "volver a WAV es inmediato y persiste");
});

// ─── Núcleo de reproducción ───────────────────────────────────────────────────
test("20: underrun pausa con atenuación, no pierde ni duplica muestras y se recupera", () => {
  const core = new PcmPlaybackCore(44100, 44100);
  const a = pcm(2000), b = pcm(3000, 2000);
  core.push(new Int16Array(a.buffer, a.byteOffset, 2000));
  core.play();
  const outs = [];
  let level = 1;
  for (let k = 0; k < 30 && core.state === "playing"; k++) { const o = new Float32Array(128); core.render(o); outs.push(o); }
  assert.equal(core.state, "underrun");
  assert.deepEqual(core.events.splice(0), ["started", "underrun"]);
  const y = Float32Array.from(outs.flatMap(o => [...o]));
  const lastSounding = y.findLastIndex(v => v !== 0);
  assert.ok(Math.abs(y[lastSounding]) < 0.02, "se apagó suave antes del corte (sin clic)");
  const o2 = new Float32Array(128); level = core.render(o2);
  assert.equal(level, 0, "durante el corte no suena nada (la boca se cierra)");
  core.push(new Int16Array(b.buffer, b.byteOffset, 3000));
  core.end();
  core.play();
  for (let k = 0; k < 60 && core.state !== "drained"; k++) core.render(new Float32Array(128));
  assert.equal(core.state, "drained");
  assert.equal(core.consumed, 5000, "ninguna muestra perdida ni repetida");
});

// ─── Sesión del navegador (con stubs de Web Audio) ────────────────────────────
test("13/15/16 (navegador): ids desconocidos y seq repetidas se ignoran; tts_done solo al terminar de sonar; cancelar no avisa", async () => {
  const posted = [];
  globalThis.AudioWorkletNode = class { constructor() { this.port = { postMessage: m => posted.push(m), onmessage: null }; } connect(n) { return n; } disconnect() {} };
  const { StreamPlayer } = await import("../public/tts-stream/stream-player.js");
  const player = new StreamPlayer();
  player.ctx = { sampleRate: 48000, state: "running", destination: {}, outputLatency: 0.01, baseLatency: 0.005, createGain: () => ({ gain: { value: 1 }, connect: n => n, disconnect() {} }), resume: () => Promise.resolve() };
  const finished = [], audible = [];
  const s = player.createSession({ id: 7, url: "stream:7", sampleRate: 44100, estDurationSec: 1, priorRate: 0.63 }, { onAudible: x => audible.push(x.id), onFinish: (x, result) => finished.push(result) });
  const ab = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
  assert.equal(player.onBinary(ab(encodeFrame(99, 0, 0, pcm(10)))), false, "id desconocido (tardío) se ignora");
  assert.equal(player.onBinary(ab(encodeFrame(7, 0, 0, pcm(4410)))), true);
  assert.equal(player.onBinary(ab(encodeFrame(7, 0, 0, pcm(4410)))), false, "seq repetida se ignora");
  assert.equal(player.onBinary(ab(encodeFrame(7, 1, 0, pcm(4410)))), true);
  s.begin();
  assert.equal(posted.filter(m => m.type === "play").length, 0, "sin fin y con poco búfer, la política todavía espera");
  s.end(8820, false);
  assert.equal(posted.filter(m => m.type === "play").length, 1, "terminó la transmisión → arranca enseguida");
  s._onWorklet({ type: "started", consumed: 0 });
  assert.deepEqual(audible, [7]);
  assert.deepEqual(finished, [], "fin de transmisión ≠ fin de reproducción");
  s._onWorklet({ type: "drained", consumed: 8820 });
  assert.deepEqual(finished, ["done"]);
  assert.equal(s.stats.playedSamples, 8820); assert.equal(s.stats.contextRate, 48000);

  const c = player.createSession({ id: 8, url: "stream:8", sampleRate: 44100 }, { onFinish: (x, result) => finished.push(result) });
  c.begin(); c.cancel("server");
  assert.deepEqual(finished, ["done", "cancelled"]);
  assert.equal(player.sessions.size, 0, "sesiones liberadas");
  delete globalThis.AudioWorkletNode;
});

// ─── Política de búfer (reloj simulado, trazas reales) ────────────────────────
const { traces } = JSON.parse(readFileSync(new URL("./fixtures/sglang-arrival-traces.json", import.meta.url), "utf8"));
// Solo mediciones de streaming de SGLang-Omni con la configuración elegida (sin warmups, sin WAV, sin Fish original)
const selected = traces.filter(t => t.engine === "sglang-omni" && t.streaming === true && t.kind === "measure" && /^(compile|compile-auto|reference-R20)$/.test(t.profile));
const audioOf = t => t.trace.at(-1)[1];
const avg = (list, f) => list.reduce((s, t) => s + f(t), 0) / list.length;

test("21: la política adaptativa es causal (cambiar el futuro no cambia cuándo arrancó)", () => {
  for (const t of selected.filter(x => x.trace.length >= 3)) {
    const est = audioOf(t);
    const s = simulatePlayback(t.trace, new AdaptiveStartPolicy({ estDurationSec: est })).firstAudibleSec;
    // todo lo que llega DESPUÉS del arranque se demora 100 s: si la política mirara el futuro, arrancaría en otro momento
    const altered = t.trace.map(([ta, a], i) => (ta > s + 1e-9 && i < t.trace.length ? [ta + 100, a] : [ta, a]));
    const s2 = simulatePlayback(altered, new AdaptiveStartPolicy({ estDurationSec: est })).firstAudibleSec;
    assert.equal(s2, s, `${t.case}: misma decisión con otro futuro`);
  }
  // si termina antes del umbral, arranca ya
  const p = new AdaptiveStartPolicy({ estDurationSec: 30 });
  p.onArrival(3.6, 1.8);
  assert.equal(p.shouldPlay(3.7, 0), false);
  p.onEnd(3.8, 2.4);
  assert.equal(p.shouldPlay(3.8, 0), true);
});

test("22: trazas reales — adaptativa sin cortes y antes que WAV en medianas/largas; fijos cortan o esperan de más", () => {
  for (const kase of ["medium", "emotion", "long"]) {
    const list = selected.filter(t => t.case === kase);
    const wav = avg(list, t => simulatePlayback(t.trace, new WholeFilePolicy()).firstAudibleSec);
    const ad = list.map(t => simulatePlayback(t.trace, new AdaptiveStartPolicy({ estDurationSec: audioOf(t) })));
    const adStart = avg(ad, r => r.firstAudibleSec);
    assert.equal(ad.reduce((s, r) => s + r.underruns, 0), 0, `${kase}: sin cortes con estimación correcta`);
    assert.ok(adStart < wav - 0.5, `${kase}: arranca antes que WAV (${adStart.toFixed(2)} vs ${wav.toFixed(2)})`);
    assert.ok(adStart >= avg(list, t => oracleStartSec(t.trace)) - 0.05, `${kase}: nunca mejor que el óptimo retrospectivo (sería trampa)`);
    for (const r of ad) assert.ok(Math.abs(r.playedSec - r.totalSec) < 0.011, "todas las muestras se reproducen");
  }
  const longs = selected.filter(t => t.case === "long");
  assert.ok(longs.every(t => simulatePlayback(t.trace, new FixedStartPolicy({ bufferSec: 0.5 })).underruns > 0), "un búfer fijo de 0.5 s corta en respuestas largas");
  // cortas: termina casi enseguida → igual que WAV, sin pérdida
  const shorts = selected.filter(t => t.case === "short");
  assert.ok(Math.abs(avg(shorts, t => simulatePlayback(t.trace, new AdaptiveStartPolicy({ estDurationSec: audioOf(t) })).firstAudibleSec) - avg(shorts, t => simulatePlayback(t.trace, new WholeFilePolicy()).firstAudibleSec)) < 0.05);
  // con estimación 30 % corta puede cortar, pero el corte aumenta la cautela (onUnderrun)
  const p = new AdaptiveStartPolicy({ estDurationSec: 10 });
  const before = p.estUncertainty; p.onUnderrun();
  assert.ok(p.estUncertainty > before);
});

test("llegadas irregulares con jitter (traza SINTÉTICA, no benchmark): adaptativa corta menos que inmediato y arranca antes que WAV", () => {
  // generador determinista: primer bloque 1.846 s a ~3.6 s, luego ~2.09 s cada ~3.17 s con ±25 % de jitter
  let seed = 7;
  const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  const synth = [];
  for (let k = 0; k < 12; k++) {
    const tr = [[3.6 + rnd() * 0.3, 1.846]];
    const blocks = 3 + Math.floor(rnd() * 6);
    for (let b = 0; b < blocks; b++) tr.push([tr.at(-1)[0] + 3.17 * (0.75 + rnd() * 0.5), tr.at(-1)[1] + 2.09]);
    synth.push(tr);
  }
  const run = mk => synth.map(tr => simulatePlayback(tr, mk(tr)));
  const ad = run(tr => new AdaptiveStartPolicy({ estDurationSec: tr.at(-1)[1] }));
  const imm = run(() => new FixedStartPolicy({ bufferSec: 0.01 }));
  const wav = run(() => new WholeFilePolicy());
  const cuts = rs => rs.reduce((s, r) => s + r.underruns, 0);
  assert.ok(cuts(ad) < cuts(imm) / 4, `adaptativa ${cuts(ad)} cortes vs inmediato ${cuts(imm)}`);
  assert.ok(avg(ad, r => r.firstAudibleSec) < avg(wav, r => r.firstAudibleSec) - 2);
});

test("H09: fixture de trazas — deduplicado, clasificado y sin contenido privado", () => {
  const raw = readFileSync(new URL("./fixtures/sglang-arrival-traces.json", import.meta.url), "utf8");
  assert.doesNotMatch(raw, /RIFF|base64|data:audio|sk-|Bearer|api_key|[A-Z]:\\\\|\/content\//i);
  const ids = traces.map(t => t.id);
  assert.equal(new Set(ids).size, ids.length, "una entrada por síntesis");
  for (const t of traces) {
    assert.ok(["sglang-omni", "fish-speech-original"].includes(t.engine), t.id);
    assert.ok(["measure", "warmup", "split"].includes(t.kind));
    assert.equal(typeof t.streaming, "boolean");
    assert.ok(t.origins.length >= 1);
    assert.ok(t.trace.every(([a, b]) => Number.isFinite(a) && Number.isFinite(b)));
    assert.equal(Object.keys(t).some(k => /text|transcript|audio_b64|path/i.test(k)), false);
  }
  assert.ok(selected.length >= 10 && selected.every(t => t.trace.length >= 2 || t.case === "short"));
});

test("H05: evaluación causal (estimador de producción, sin futuro, calibración separada) — sin cortes y ahorro en medianas/largas", async () => {
  const { withChars, runCausal, summarize } = await import("../scripts/sim-buffer-policies.mjs");
  assert.ok(withChars.length >= 32 && withChars.every(t => t.engine === "sglang-omni" && t.streaming && t.kind === "measure"));
  for (const kase of ["medium", "emotion", "long"]) {
    const s = summarize(runCausal(withChars.filter(t => t.case === kase), "aprendido conservador ≤14 cps (PRODUCCIÓN)"));
    assert.equal(s.cuts, 0, `${kase}: sin cortes`);
    assert.ok(s.savingP50 > 0.5, `${kase}: ahorro p50 ${s.savingP50}`);
  }
  const short = summarize(runCausal(withChars.filter(t => t.case === "short"), "aprendido conservador ≤14 cps (PRODUCCIÓN)"));
  assert.equal(short.cuts, 0);
  // riesgo documentado: si la estimación queda 30 % corta, aparecen cortes en largas
  const risky = summarize(runCausal(withChars.filter(t => t.case === "long"), "aprendido, error −30 %"));
  assert.ok(risky.cuts > 0);
});

test("velocidad de generación que cambia a mitad (traza SINTÉTICA): la política se adapta mejor que un búfer fijo", () => {
  // r ≈ 0.66 al principio y ≈ 0.45 desde el 4.º bloque (p.ej. GPU compartida). No es un benchmark.
  const trace = [[3.6, 1.846]];
  for (let b = 0; b < 7; b++) trace.push([trace.at(-1)[0] + (b < 3 ? 3.17 : 4.65), trace.at(-1)[1] + 2.09]);
  const ad = simulatePlayback(trace, new AdaptiveStartPolicy({ estDurationSec: trace.at(-1)[1] }));
  const f2 = simulatePlayback(trace, new FixedStartPolicy({ bufferSec: 2 }));
  const wav = simulatePlayback(trace, new WholeFilePolicy());
  assert.ok(ad.underrunSec < f2.underrunSec, `adaptativa ${ad.underrunSec}s vs fijo 2s ${f2.underrunSec}s`);
  assert.ok(ad.firstAudibleSec < wav.firstAudibleSec);
  assert.equal(Math.round(ad.playedSec * 100), Math.round(ad.totalSec * 100));
});

test("26: frames y WAV de conversión bien formados", () => {
  const f = encodeFrame(3, 9, 120, pcm(11));
  assert.equal(f.length, FRAME_HEADER_BYTES + 22);
  assert.deepEqual({ ...decodeFrame(f), pcm: undefined }, { id: 3, seq: 9, ageMs: 120, pcm: undefined });
  assert.equal(decodeFrame(Buffer.from("XXXX0000000000000000")), null);
  const w = validateWavBuffer(pcmToWav([pcm(100), pcm(50)]));
  assert.equal(w.dataSize, 300); assert.equal(w.sampleRate, 44100); assert.equal(w.channels, 1);
});

// ─── Ronda de auditoría: consumidores, ACK, memoria y sesiones abandonadas ────
async function playToEnd(e, text, plan = { parts: [[0, pcm(2000)]], end: "ok" }) {
  gw.state.plan = plan;
  const r = await e.tts.synth(text, { source: "live", returnMeta: true });
  await e.tts.play(r.url, text);
  await until(() => e.clients.some(ws => json(ws).some(m => m.type === "tts_stream_end" && m.url === r.url)));
  return r.url;
}

test("H02/H03: dos vistas audibles — el turno espera a ambas; ACK duplicado idempotente; ajeno y tardío ignorados", async () => {
  const A = fakeWs(), B = fakeWs();
  const e = makeEngine({ clients: [A, B] });
  const url = await playToEnd(e, "Dos vistas audibles");
  assert.equal(frames(A).length, 1); assert.equal(frames(B).length, 1, "cada vista recibe el stream una sola vez");
  assert.equal(e.tts.onTtsDone(A, url), "waiting");
  assert.equal(e.tts.onTtsDone(A, url), "waiting", "duplicado: no cambia nada");
  assert.equal(e.released, 0, "B todavía suena: el turno no se libera");
  const intruso = fakeWs();
  assert.equal(e.tts.onTtsDone(intruso, url), "ignored", "una vista que no participó no puede cerrarla");
  assert.equal(e.tts.onTtsDone(B, "stream:9999"), "ignored", "url de otra sesión");
  assert.equal(e.tts.onTtsDone(B, url), "complete");
  assert.equal(e.released, 1);
  assert.equal(e.tts.onTtsDone(B, url), "ignored", "tardío después de liberar");
  // ACK tardío de la sesión anterior mientras suena la siguiente: no libera el turno nuevo
  const url2 = await playToEnd(e, "Siguiente");
  assert.equal(e.tts.onTtsDone(A, url), "ignored");
  assert.equal(e.released, 1);
  e.tts.onTtsDone(A, url2); e.tts.onTtsDone(B, url2);
  assert.equal(e.released, 2);
});

test("H02/H03: vista ?mute no libera antes que la audible; si todas son mudas manda la última; sin url no corta un stream", async () => {
  const audible = fakeWs(), muda = fakeWs({ ttsMute: true });
  const e = makeEngine({ clients: [audible, muda] });
  const url = await playToEnd(e, "Con vista muda");
  assert.equal(e.tts.onTtsDone(muda, url), "waiting", "la muda terminó primero: no alcanza");
  assert.equal(e.tts.onTtsDone(fakeWs(), undefined), "ignored", "página vieja sin url no corta la sesión PCM");
  assert.equal(e.released, 0);
  assert.equal(e.tts.onTtsDone(audible, url), "complete");
  assert.equal(e.tts.streams.lastStats.server.audibleParticipants, 1);

  const m1 = fakeWs({ ttsMute: true }), m2 = fakeWs({ ttsMute: true });
  const e2 = makeEngine({ clients: [m1, m2] });
  const u2 = await playToEnd(e2, "Todas mudas");
  assert.equal(e2.tts.onTtsDone(m1, u2), "waiting");
  assert.equal(e2.tts.onTtsDone(m2, u2), "complete");
});

test("H02: compatible + incompatible — la incompatible no recibe ni PCM ni un WAV extra; la que entra a mitad no se suma", async () => {
  const compat = fakeWs(), vieja = fakeWs({ ttsStream: false });
  const e = makeEngine({ clients: [compat, vieja] });
  gw.state.plan = { parts: [[0, pcm(2000)], [150, pcm(2000)]], end: "ok" };
  const r = await e.tts.synth("Mixto", { source: "live", returnMeta: true });
  await e.tts.play(r.url, "Mixto");
  const tarde = fakeWs();
  e.clients.push(tarde);
  await until(() => json(compat).some(m => m.type === "tts_stream_end"));
  assert.equal(vieja.sent.length, 0); assert.deepEqual(e.plays, [], "ningún tts_play que duplique el audio");
  assert.equal(tarde.sent.length, 0, "la vista conectada durante la reproducción no recibe un stream a medias");
  assert.equal(e.tts.onTtsDone(compat, r.url), "complete");
});

test("H02/H03: una vista se va (o se le suspende el audio) mientras otra sigue; si las que quedan ya terminaron, se libera", async () => {
  const A = fakeWs(), B = fakeWs();
  const e = makeEngine({ clients: [A, B] });
  const url = await playToEnd(e, "Una se va");
  e.tts.onTtsDone(B, url);
  assert.equal(e.released, 0);
  e.tts.streamClientGone(A); // equivalente a cierre de WS o tts_caps con stream:false
  assert.equal(e.released, 1, "A se fue sin terminar y B ya había terminado → turno liberado");

  const C = fakeWs(), D = fakeWs();
  const e2 = makeEngine({ clients: [C, D] });
  gw.state.plan = { parts: [[0, pcm(2000)], [200, pcm(2000)]], end: "ok" };
  const r = await e2.tts.synth("Sigue con una", { source: "live", returnMeta: true });
  await e2.tts.play(r.url, "Sigue con una");
  e2.tts.streamClientGone(C);
  const cAtLeave = C.sent.length;
  assert.equal(e2.released, 0, "D sigue escuchando");
  await until(() => json(D).some(m => m.type === "tts_stream_end"));
  assert.equal(C.sent.length, cAtLeave, "a la que se fue no se le manda nada más");
  assert.equal(frames(D).length, 2);
  assert.equal(e2.tts.onTtsDone(D, r.url), "complete");
});

test("H03: stream cancelado no revive; timeout del turno cubre la generación completa y luego audio + 15 s", async () => {
  const e = makeEngine();
  const ws = e.clients[0];
  const timeouts = [];
  const orig = e.tts._startPlaybackTimeoutMs.bind(e.tts);
  e.tts._startPlaybackTimeoutMs = ms => { timeouts.push(ms); orig(ms); };
  gw.state.plan = { parts: [[0, pcm(4410)], [100, pcm(4410)]], end: "ok" };
  const r = await e.tts.synth("x".repeat(5), { source: "live", returnMeta: true }); // estimación ridículamente corta
  await e.tts.play(r.url, "xxxxx");
  assert.equal(timeouts[0], 330000, "mientras genera no depende de la estimación por texto");
  await until(() => json(ws).some(m => m.type === "tts_stream_end"));
  assert.equal(timeouts.at(-1), 200 + 15000, "al terminar: duración real del audio + 15 s");
  e.tts.stop();
  const st = e.tts.streams.get(r.url);
  assert.equal(st, null);
  assert.equal(e.tts.onTtsDone(ws, r.url), "ignored");
  const released = e.released;
  await e.tts.play(r.url, "xxxxx"); // una referencia vieja no reactiva nada
  assert.equal(e.released, released + 1, "play de una sesión cancelada solo libera");
  assert.equal(frames(ws).filter(f => f.id === Number(r.url.slice(7))).length, 2);
});

test("H04: presupuesto global de memoria — descarta sesiones abandonadas, aborta la que no entra y la contabilidad vuelve a 0", async () => {
  const m = new SglangStreamManager({ outDir: join(tmp, "o"), logger: null, maxTotalBytes: 100_000 });
  const a = m.create({ source: "live", text: "a" }); m.onHeaders(a);
  m.onPcm(a, pcm(20_000), {}); m.onEnd(a, {});      // 40 KB, lista y sin reproducir
  const b = m.create({ source: "live", text: "b" }); m.onHeaders(b);
  m.onPcm(b, pcm(20_000), {});                       // 80 KB en total
  m.onPcm(b, pcm(15_000), {});                       // pasaría de 100 KB → descarta "a"
  assert.equal(m.get(a.url), null, "la sesión abandonada se descartó");
  assert.equal(m.totalBytes, 70_000);
  assert.throws(() => m.onPcm(b, pcm(20_000), {}), /memoria/, "si igual no entra, la descarga se aborta de forma controlada");
  assert.equal(m.totalBytes, 70_000);
  // en modo stream lo enviado se suelta enseguida
  const ws = fakeWs();
  m.hub = { clients: () => [ws], send: (w, d) => w.sent.push(d) };
  await m.play(b, {});
  assert.equal(b.bytes, 0); assert.equal(m.totalBytes, 0);
  m.onPcm(b, pcm(30_000), {});
  assert.equal(m.totalBytes, 0, "streaming largo: la memoria no crece con la duración");
  m.release(b.url);
  assert.equal(m.streams.size, 0);
});

test("H04: estrés con 30 sesiones sintéticas (sin GPU) — todo termina liberado", async () => {
  const m = new SglangStreamManager({ outDir: join(tmp, "o"), logger: null, maxTotalBytes: 2_000_000, unclaimedMs: 20 });
  const ws = fakeWs({ bufferedAmount: 0 });
  m.hub = { clients: () => [ws], send: (w, d) => w.sent.push(d) };
  const all = [];
  for (let i = 0; i < 30; i++) {
    const st = m.create({ source: i % 2 ? "live" : "panel_test", text: `sesión ${i}` });
    m.onHeaders(st);
    for (let k = 0; k < 5; k++) try { m.onPcm(st, pcm(10_000 + k), {}); } catch { m.onFail(st, Object.assign(new Error("mem"), { partial: true })); break; }
    if (st.state === "generating") m.onEnd(st, {});
    all.push(st);
    if (i % 3 === 0) m.cancel(st);
    else if (i % 3 === 1) { await m.play(st, {}); m.release(st.url); }
    // i % 3 === 2: nadie la reproduce → se descarta sola (unclaimedMs)
  }
  assert.ok(m.totalBytes <= 2_000_000);
  await sleep(60);
  assert.equal(m.streams.size, 0, "ninguna sesión quedó colgada");
  assert.equal(m.totalBytes, 0);
  assert.ok(all.filter((_, i) => i % 3 === 2).every(st => st.chunks.length === 0 && !st.participants));
  assert.ok(m.statsById.size <= 8, "historial de métricas acotado");
});

test("H04 (limitación documentada): un corte limpio en límite de muestra es indistinguible de un final normal", async () => {
  // El gateway no envía longitud total ni confirmación: si cierra bien el chunked encoding antes de tiempo, parece completo.
  const { client } = makeClient();
  gw.state.plan = { parts: [[0, pcm(1000)]], end: "ok" };
  const m = await client.synthesizeStream("Respuesta que en realidad seguía");
  assert.equal(m.result, "ok");
  // en cambio, una conexión cortada sin el chunk final sí se detecta (test 5/6)
});

test("H07: prepare-ahead — la cola remota no se libera al resolver las cabeceras; TTL vencido en cola no hace HTTP", async () => {
  const e = makeEngine();
  gw.state.maxInFlight = 0;
  gw.state.speechCalls = 0;
  gw.state.plan = { parts: [[0, pcm(1000)], [200, pcm(1000)]], end: "ok" };
  const first = await e.tts.synth("Primera", { source: "live", returnMeta: true });
  assert.match(first.url, /^stream:/);
  const second = e.tts.synth("Segunda", { source: "live", returnMeta: true });
  const stale = e.tts.synth("Vencida", { source: "live", returnMeta: true, expiresAt: Date.now() + 50 });
  await Promise.all([second, stale]);
  assert.equal(gw.state.maxInFlight, 1, "la segunda esperó a que la primera terminara de GENERAR");
  assert.equal((await stale).cancelled, true);
  assert.equal(gw.state.speechCalls, 2, "la vencida nunca hizo HTTP");
});

test("H07: sesión lista que nunca se reproduce se libera sola; cambiar a WAV no rompe la ya preparada", async () => {
  const e = makeEngine();
  e.tts.streams.unclaimedMs = 30;
  gw.state.plan = { parts: [[0, pcm(1000)]], end: "ok" };
  const abandoned = await e.tts.synth("Nadie me reproduce", { source: "live", returnMeta: true });
  await e.tts.streams.get(abandoned.url).settled;
  await sleep(60);
  assert.equal(e.tts.streams.get(abandoned.url), null);
  assert.equal(e.tts.streams.totalBytes, 0);
  assert.equal(e.tts.streams.lastStats.server.result, "unclaimed");

  const prepared = await e.tts.synth("Preparada en stream", { source: "live", returnMeta: true });
  e.tts.sglang.updateConfig({ playbackMode: "wav" });
  await e.tts.play(prepared.url, "Preparada en stream");
  await until(() => json(e.clients[0]).some(m => m.type === "tts_stream_end" && m.url === prepared.url));
  e.tts.onTtsDone(e.clients[0], prepared.url);
  const next = await e.tts.synth("Ahora en WAV", { source: "live", returnMeta: true });
  assert.match(next.url, /^\/tts\/\d+\.wav$/, "lo nuevo ya sale en WAV");
});

test("navegador: tope de memoria por sesión", async () => {
  globalThis.AudioWorkletNode = class { constructor() { this.port = { postMessage() {}, onmessage: null }; } connect(n) { return n; } disconnect() {} };
  const { StreamPlayer } = await import("../public/tts-stream/stream-player.js");
  const p = new StreamPlayer();
  p.ctx = { sampleRate: 44100, state: "running", destination: {}, createGain: () => ({ gain: { value: 1 }, connect: n => n, disconnect() {} }), resume: () => Promise.resolve() };
  const out = [];
  const s = p.createSession({ id: 1, url: "stream:1", sampleRate: 44100 }, { onFinish: (_x, r, reason) => out.push([r, reason]) });
  assert.equal(s.push(0, new Int16Array(13 * 1024 * 1024)), false);
  assert.deepEqual(out, [["error", "too_large"]]);
  delete globalThis.AudioWorkletNode;
});
