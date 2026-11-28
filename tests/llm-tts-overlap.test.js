// Fase 3: superposición LLM → TTS. Segmentador, controlador, parsers de los 4 proveedores y síntesis anticipada sobre
// TTSEngine real (extraído de server.js) con un gateway mock. LLM simulado: no se usa ningún servicio real.
import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, readdirSync, unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { findFirstSentence, prefixRemainder, OverlapController, OVERLAP_PROVIDERS, UNSAFE_SPEECH } from "../lib/tts/llm-tts-overlap.js";
import { SglangScheduler } from "../lib/tts/sglang-scheduler.js";
import { installSpeculation } from "../lib/tts/sglang-speculation.js";
import { SGLangTTSClient } from "../lib/tts/sglang-tts-client.js";
import { SglangStreamManager, decodeFrame } from "../lib/tts/sglang-stream.js";
import { ClaudeClient, AgyClient, CodexClient, OpenCodeClient } from "../lib/ai/cli-clients.js";
import { extractEmotionTag, getMoodState, resolveSpeakingEmotion, evaluateMoodPenalty } from "../lib/ai/emotion-state.js";
import { sanitizeModelResponse } from "../lib/moderation/response-sanitizer.js";
import { evaluateIgnoreAction, hasHostileEvidence } from "../lib/moderation/ignore-guard.js";
import { evaluateContextualSafety } from "../lib/moderation/safety-guard.js";
import { dropLeadingName } from "../lib/ai/rag-utils.js";

const KEY = "sk-overlap-secret-4242";
const sleep = ms => new Promise(r => setTimeout(r, ms));
const pcm = n => { const b = Buffer.alloc(n * 2); for (let i = 0; i < n; i++) b.writeInt16LE(Math.round(Math.sin(i / 9) * 8000), i * 2); return b; };
const PCM_HEADERS = { "Content-Type": "application/octet-stream", "X-Audio-Sample-Rate": "44100", "X-Audio-Channels": "1", "X-Audio-Encoding": "pcm_s16le" };

// ─── Segmentador ──────────────────────────────────────────────────────────────
test("15/16/17: segmentador conservador — límites naturales, sin cortar abreviaturas, decimales, links ni pausas", () => {
  const cases = [
    ["Che, qué buena pregunta la tuya. Te cuento que sí, lo vi ayer.", "Che, qué buena pregunta la tuya."],
    ["¡Muchísimas gracias por el regalo, sos lo más! Ya se lo dije a todos.", "¡Muchísimas gracias por el regalo, sos lo más!"],
    ["¿En serio me preguntás eso a esta hora? Bueno, te contesto igual.", "¿En serio me preguntás eso a esta hora?"],
    ["Me lo dijo el Dr. Pérez en la consulta de ayer. Y no le creí.", "Me lo dijo el Dr. Pérez en la consulta de ayer."],
    ["El juego pesa 3.5 GB y tarda bastante en bajar. Paciencia.", "El juego pesa 3.5 GB y tarda bastante en bajar."],
    ["Uhm... no sé qué decirte la verdad, che. Dejame pensarlo.", "Uhm... no sé qué decirte la verdad, che."],
  ];
  for (const [text, expected] of cases) {
    const r = findFirstSentence(text);
    assert.equal(r?.segment, expected, text);
    assert.equal(`${r.segment} ${r.rest}`, text, "concatenar segmento + resto reproduce el texto: nada perdido ni duplicado");
  }
  assert.equal(findFirstSentence("Hola. Qué tal todo por ahí, chat"), null, "primer candidato demasiado corto y no hay otro completo");
  assert.equal(findFirstSentence("Una frase sin terminar todavía, sigue llegando"), null);
  assert.equal(findFirstSentence("Esto termina acá con punto final."), null, "sin el comienzo de la oración siguiente el signo todavía puede cambiar");
  assert.equal(findFirstSentence("Me dijo «esto es así. Punto» y se fue enojado. Raro.")?.segment, "Me dijo «esto es así. Punto» y se fue enojado.", "no corta dentro de comillas: toma la oración completa");
  assert.equal(findFirstSentence("Mirá esto, una coma no es fin de oración, para nada, che"), null);
  assert.ok(UNSAFE_SPEECH.test("Mirá https://ejemplo.com ahora mismo."));
  assert.ok(UNSAFE_SPEECH.test("[IGNORAR: 5] Andate."));
  assert.ok(UNSAFE_SPEECH.test("*se da vuelta* Bueno."));
});

test("prefixRemainder: solo coincidencia exacta en límite de palabra", () => {
  assert.equal(prefixRemainder("Hola chat. Qué tal.", "Hola chat."), "Qué tal.");
  assert.equal(prefixRemainder("Hola chat.", "Hola chat."), "");
  assert.equal(prefixRemainder("Hola chatitos. Qué tal.", "Hola chat"), null, "no corta palabras");
  assert.equal(prefixRemainder("Chau chat. Qué tal.", "Hola chat."), null);
});

// ─── Controlador (TTS falso) ──────────────────────────────────────────────────
function fakeTts({ ready = null } = {}) {
  const calls = { start: [], commit: [], discard: [] };
  return {
    calls,
    overlapReady: () => (typeof ready === "function" ? ready() : ready),
    ttsTextOf: t => String(t).replace(/NYRA/g, "Nira").trim(),
    speculateStart: arg => { calls.start.push(arg); return { id: calls.start.length }; },
    commitSpeculation: async (h, remainder) => { calls.commit.push(remainder); return { url: "stream:7" }; },
    discardSpeculation: (h, reason) => { calls.discard.push(reason); return 1200; },
  };
}
const build = s => (UNSAFE_SPEECH.test(s) ? null : { ttsText: s.replace(/NYRA/g, "Nira"), emotion: "happy" });

test("2/35/1: desactivada por defecto; proveedores sin texto parcial confiable usan el camino tradicional", () => {
  const off = new OverlapController({ tts: fakeTts(), enabled: () => false });
  assert.equal(off.begin({ providerId: "agy", source: "live", build }), null);
  const on = new OverlapController({ tts: fakeTts(), enabled: () => true });
  for (const p of ["codex-cli", "opencode", undefined]) assert.equal(on.begin({ providerId: p, source: "live", build }), null);
  assert.ok(Object.keys(on.stats.skipped).some(k => /codex-cli/.test(k)));
  assert.deepEqual([...OVERLAP_PROVIDERS].sort(), ["agy", "claude-code"]);
  const notReady = new OverlapController({ tts: fakeTts({ ready: "requiere Reproducción: Streaming PCM" }), enabled: () => true });
  assert.equal(notReady.begin({ providerId: "agy", source: "live", build }), null);
  assert.equal(notReady.stats.skipped["requiere Reproducción: Streaming PCM"], 1);
});

test("4/5/6/18/21: dispara una sola vez con texto incremental; usa lo preparado solo si el final coincide", async () => {
  const tts = fakeTts();
  const c = new OverlapController({ tts, enabled: () => true, delayMs: 0, minHeadStartMs: 0 });
  const s = c.begin({ providerId: "claude-code", source: "live", mood: 70, build });
  let body = "";
  for (const d of ["Che", ", qué buena pre", "gunta la tuya, de verdad", ". Te cuento", " que sí", ". Fin."]) { body += d; c.feed(s, body); c.feed(s, body); }
  assert.equal(tts.calls.start.length, 1, "no reinicia la síntesis por cada fragmento ni por fragmentos repetidos");
  assert.equal(tts.calls.start[0].ttsText, "Che, qué buena pregunta la tuya, de verdad.");
  assert.equal(tts.calls.start[0].mood, 70);
  const used = await c.finish(s, { filtered: body, emotion: "happy" });
  assert.equal(used.url, "stream:7");
  assert.deepEqual(tts.calls.commit, ["Te cuento que sí. Fin."], "solo el resto: la 1.ª oración no se repite");
  assert.equal(c.stats.used, 1); assert.equal(c.stats.discarded, 0);
  c.abort(s, "tarde"); // ya usada: no hace nada
  assert.equal(tts.calls.discard.length, 0);
});

test("7/12/14/21: texto corregido, respuesta rechazada o emoción distinta → se descarta y sigue el camino normal", async () => {
  for (const [filtered, emotion, why] of [
    ["Che, otra cosa distinta. Nada que ver.", "happy", /no empieza/],
    ["Ups, eso no te lo puedo decir.", "happy", /no empieza/],                    // negativa del filtro: otro texto
    ["Che, qué buena pregunta la tuya, de verdad. Sí.", "sad", /emoción/],           // [EMOCIÓN] final distinta
  ]) {
    const tts = fakeTts();
    const c = new OverlapController({ tts, enabled: () => true, delayMs: 0, minHeadStartMs: 0 });
    const s = c.begin({ providerId: "agy", source: "live", build });
    c.feed(s, "Che, qué buena pregunta la tuya, de verdad. Te cuento");
    assert.equal(await c.finish(s, { filtered, emotion }), null);
    assert.equal(tts.calls.commit.length, 0);
    assert.match(Object.keys(c.stats.discardReasons)[0], why);
    assert.equal(c.stats.wastedGpuMsTotal, 1200);
  }
  // configuración cambiada entre la especulación y la validación
  let ready = null;
  const tts = fakeTts({ ready: () => ready });
  const c = new OverlapController({ tts, enabled: () => true, delayMs: 0, minHeadStartMs: 0 });
  const s = c.begin({ providerId: "agy", source: "live", build });
  c.feed(s, "Che, qué buena pregunta la tuya, de verdad. Te cuento");
  ready = "el motor del directo no es SGLang";
  assert.equal(await c.finish(s, { filtered: "Che, qué buena pregunta la tuya, de verdad. Te cuento.", emotion: "happy" }), null);
  assert.match(Object.keys(c.stats.discardReasons)[0], /configuración/);
});

test("8/9/10: oraciones con marcas o markdown no se especulan; abortar es idempotente", () => {
  const tts = fakeTts();
  const c = new OverlapController({ tts, enabled: () => true, delayMs: 0, minHeadStartMs: 0 });
  const s = c.begin({ providerId: "agy", source: "live", build });
  c.feed(s, "*se ríe un montón y mira al chat* Bueno, vamos. Dale.");
  assert.equal(tts.calls.start.length, 0);
  assert.equal(s.state, "unsafe");
  c.abort(s); c.abort(s);
  assert.equal(tts.calls.discard.length, 0, "no había nada sintetizado");
});

// ─── Parsers reales de los 4 proveedores ──────────────────────────────────────
test("8/9/11: Claude Code — solo text_delta; razonamiento y herramientas ignorados; fin por result", () => {
  const c = new ClaudeClient("chat", "m-low", { path: "x", workDir: tmpdir() });
  const got = []; let done = null;
  const events = [
    { type: "stream_event", event: { type: "content_block_delta", delta: { type: "thinking_delta", thinking: "razono en secreto" } } },
    { type: "stream_event", event: { type: "content_block_delta", delta: { type: "input_json_delta", partial_json: "{\"cmd\":\"rm" } } },
    { type: "stream_event", event: { type: "content_block_delta", delta: { type: "text_delta", text: "Hola " } } },
    { type: "assistant", message: { content: [{ type: "tool_use", name: "Bash", input: { cmd: "ls" } }] } },
    { type: "stream_event", event: { type: "content_block_delta", delta: { type: "text_delta", text: "chat." } } },
    { type: "result", subtype: "success", is_error: false, result: "Hola chat." },
  ];
  for (const e of events) c._onEvent(e, d => got.push(d), (err, v) => { done = [err, v]; });
  assert.deepEqual(got, ["Hola ", "chat."]);
  assert.deepEqual(done, [null, "Hola chat."]);
});

test("8/9: Antigravity (agy) — solo agent_response text_delta; otros pasos ignorados", () => {
  const a = new AgyClient("chat", "m", { path: "x", workDir: tmpdir() });
  const got = []; let done = null;
  for (const e of [
    { step_update: { step_type: "thinking", text_delta: "pienso" } },
    { step_update: { step_type: "tool_call", text_delta: "{\"x\":" } },
    { step_update: { step_type: "agent_response", text_delta: "Buenas" } },
    { event: "result", result: { status: "SUCCESS", response: "Buenas" } },
  ]) a._onEvent(e, d => got.push(d), (err, v) => { done = [err, v]; });
  assert.deepEqual(got, ["Buenas"]);
  assert.deepEqual(done, [null, "Buenas"]);
});

test("1/8: Codex — sin texto parcial: el mensaje llega completo; razonamiento ignorado", () => {
  const c = new CodexClient("chat", "m", { path: "x", workDir: tmpdir() });
  assert.equal(c._textFromEvent({ type: "item.completed", item: { type: "reasoning", text: "secreto" } }), "");
  assert.equal(c._textFromEvent({ type: "item.updated", item: { type: "agent_message", text: "Hol" } }), "", "no hay parciales");
  assert.equal(c._textFromEvent({ type: "item.completed", item: { type: "agent_message", text: "Hola chat." } }), "Hola chat.");
});

test("5/6/9: OpenCode — snapshots acumulativos y duplicados se normalizan; partes no-texto ignoradas", () => {
  const o = new OpenCodeClient("chat", "m", { path: "x", workDir: tmpdir() });
  const seen = new Map();
  const deltas = [
    { part: { id: "p1", type: "text", text: "Hola" } },
    { part: { id: "p1", type: "text", text: "Hola" } },          // duplicado
    { part: { id: "p1", type: "text", text: "Hola chat" } },     // snapshot acumulativo
    { part: { id: "t1", type: "tool", text: "{\"cmd\"" } },
    { part: { id: "r1", type: "reasoning", text: "pienso" } },
  ].map(e => o._textFromEvent(e, seen));
  assert.deepEqual(deltas, ["Hola", "", " chat", "", ""]);
});

// ─── Integración: TTSEngine real + especulación + gateway mock ────────────────
async function startGateway() {
  const state = { requests: [], inFlight: 0, maxInFlight: 0, plan: () => ({ parts: [[0, pcm(4410)], [40, pcm(4410)]], end: "ok" }) };
  const server = createServer((req, res) => {
    const chunks = [];
    req.on("data", c => chunks.push(c));
    req.on("end", async () => {
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
      state.requests.push({ headers: req.headers, body });
      if (req.headers.authorization !== `Bearer ${KEY}`) { res.writeHead(401); return res.end(); }
      state.inFlight += 1; state.maxInFlight = Math.max(state.maxInFlight, state.inFlight);
      try {
        const plan = state.plan(body);
        if (plan.status) { res.writeHead(plan.status); return res.end(); }
        if (!body.stream) { const w = Buffer.concat([Buffer.alloc(44), pcm(100)]); res.writeHead(500); return res.end(w); }
        res.writeHead(200, PCM_HEADERS);
        for (const [d, b] of plan.parts) { await sleep(d); if (res.destroyed) return; res.write(b); }
        res.end();
      } finally { state.inFlight -= 1; }
    });
  });
  await new Promise(r => server.listen(0, "127.0.0.1", r));
  return { state, url: `http://127.0.0.1:${server.address().port}`, close: () => new Promise(r => { server.closeAllConnections?.(); server.close(r); }) };
}

function loadTTSEngine(CONFIG, dir, logs) {
  const src = readFileSync(new URL("../server.js", import.meta.url), "utf8");
  const lines = src.split(/\r?\n/);
  const start = lines.findIndex(l => l.startsWith("class TTSEngine"));
  const end = lines.findIndex((l, i) => i > start && l === "}");
  const quiet = { log: s => logs.push(String(s)), warn: s => logs.push(String(s)) };
  const PatchedSglang = class extends SGLangTTSClient { constructor(o) { super({ ...o, settingsFile: join(dir, "tts_settings.json"), allowLoopbackHttp: true, logger: quiet }); } };
  const PatchedStreams = class extends SglangStreamManager { constructor(o) { super({ ...o, logger: quiet }); } };
  const FishStub = class { constructor() { this.enginePrefs = {}; this.config = {}; } abortSource() {} abortAll() {} };
  return new Function("deps", `const { CONFIG, FishAudioClient, SGLangTTSClient, SglangStreamManager, SglangScheduler, join, __dirname, mkdirSync, renameSync, rmSync, readdirSync, unlinkSync, spawn, createInterface, applyPhonetics, exec, discord } = deps;
    ${lines.slice(start, end + 1).join("\n")}
    return TTSEngine;`)({
    CONFIG, FishAudioClient: FishStub, SGLangTTSClient: PatchedSglang, SglangStreamManager: PatchedStreams, SglangScheduler, join, __dirname: dir, mkdirSync, renameSync, rmSync,
    readdirSync, unlinkSync, spawn: () => {}, createInterface: () => ({ on: () => {} }), applyPhonetics: s => s,
    exec: (_c, cb) => cb?.(), discord: { stopAudio: () => {} },
  });
}

let gw, tmp;
test.beforeEach(async () => { gw = await startGateway(); tmp = mkdtempSync(join(tmpdir(), "nyra-overlap-")); });
test.afterEach(async () => { await gw.close(); rmSync(tmp, { recursive: true, force: true }); });

function makeEngine({ mode = "stream", clients = null, share = true, delayMs = 0, minHeadStartMs = 0 } = {}) {
  const logs = [];
  const CONFIG = { ttsEngine: "sglang", shareTtsEngine: share, discordTtsEngine: "edge", useSystemTTS: false, edgeVoice: "x", murf: { apiKey: "" },
    sglang: { url: gw.url, apiKey: KEY, timeoutMs: 60000, playbackMode: mode } };
  mkdirSync(join(tmp, "public", "tts"), { recursive: true });
  const TTSEngine = loadTTSEngine(CONFIG, tmp, logs);
  const tts = installSpeculation(new TTSEngine());
  const ws = { readyState: 1, ttsStream: true, bufferedAmount: 0, sent: [] };
  const e = { tts, logs, ws, clients: clients ?? [ws], edge: 0, plays: [], released: 0 };
  tts.streams.hub = { clients: () => e.clients, send: (w, d) => w.sent.push(d) };
  tts._broadcast = m => { if (m.type === "tts_play") e.plays.push(m.url); };
  tts._onDone = () => { e.released += 1; };
  tts._synthEdge = async () => { e.edge += 1; return "/tts/edge.mp3"; };
  e.overlap = new OverlapController({ tts, enabled: () => true, delayMs, minHeadStartMs });
  return e;
}
const frames = ws => ws.sent.filter(Buffer.isBuffer).map(decodeFrame);
const jsonMsgs = ws => ws.sent.filter(m => typeof m === "string").map(m => JSON.parse(m));
const until = async (cond, ms = 3000) => { const t0 = Date.now(); while (!cond()) { if (Date.now() - t0 > ms) throw new Error("timeout"); await sleep(5); } };

// Simula el flujo de _prepare: LLM incremental → validación → finish (o synth tradicional)
async function respond(e, deltas, { filtered, emotion = "happy", provider = "claude-code", gapMs = 30, validate = true } = {}) {
  const spec = e.overlap.begin({ providerId: provider, source: "live", mood: 50, build: s => ({ ttsText: e.tts.ttsTextOf(s), emotion: "happy" }) });
  let body = "";
  for (const d of deltas) { await sleep(gapMs); body += d; e.overlap.feed(spec, body); }
  try {
    if (!validate) return null; // respuesta rechazada por las validaciones de siempre
    return (await e.overlap.finish(spec, { filtered: filtered ?? body, emotion })) || await e.tts.synth(filtered ?? body, { emotion, source: "live", returnMeta: true });
  } finally { e.overlap.abort(spec, "respuesta descartada antes de reproducirse"); }
}

test("24/27/34: una sola sesión PCM con los dos segmentos en orden, sin inferencias paralelas ni texto repetido", async () => {
  const e = makeEngine();
  const r = await respond(e, ["Che, qué buena pregunta", " la tuya, de verdad.", " Te cuento", " que sí, lo vi ayer."]);
  assert.match(r.url, /^stream:\d+$/); assert.equal(r.overlap, true);
  await e.tts.play(r.url, "Che, qué buena pregunta la tuya, de verdad. Te cuento que sí, lo vi ayer.");
  await until(() => jsonMsgs(e.ws).some(m => m.type === "tts_stream_end"));
  assert.equal(gw.state.maxInFlight, 1, "la cola única se respeta");
  assert.deepEqual(gw.state.requests.map(q => q.body.input), ["[laughing] Che, qué buena pregunta la tuya, de verdad.", "[laughing] Te cuento que sí, lo vi ayer."]);
  assert.ok(gw.state.requests.every(q => q.body.stream === true && Object.keys(q.body).length === 5));
  const fr = frames(e.ws);
  assert.deepEqual(fr.map(f => f.seq), fr.map((_, i) => i), "secuencia continua entre segmentos");
  assert.equal(jsonMsgs(e.ws).find(m => m.type === "tts_stream_end").totalSamples, 4 * 4410);
  assert.deepEqual(e.plays, [], "sin tts_play adicional");
  assert.equal(e.overlap.stats.used, 1);
  // la vista sabe dónde empieza el segundo pedido (para reservar búfer y no cortar en el límite)
  const start = jsonMsgs(e.ws).find(m => m.type === "tts_stream_start");
  const seg = jsonMsgs(e.ws).find(m => m.type === "tts_stream_segment");
  assert.equal(start.segmentBoundarySamples ?? seg?.boundarySamples, 2 * 4410);
  assert.equal(e.tts.onTtsDone(e.ws, r.url), "complete");
  assert.equal(e.tts.streams.streams.size, 0);
});

test("política con límite de segmento: el TTFA del segundo pedido no produce un corte (trazas SINTÉTICAS calibradas)", async () => {
  const { AdaptiveStartPolicy, simulatePlayback } = await import("../public/tts-stream/buffer-policy.js");
  const { ttsArrivals, TTS_MODEL } = await import("../scripts/sim-llm-tts-overlap.mjs");
  // 1.ª oración (3.9 s) pedida a t=2.5; el resto (4.7 s) recién cuando termina la primera y se validó la respuesta (t=7)
  const a1 = ttsArrivals(2.5, 3.9);
  const a2 = ttsArrivals(Math.max(a1.at(-1)[0], 7), 4.7).map(([t, a]) => [t, a + 3.9]);
  const gated = opts => { const p = new AdaptiveStartPolicy({ estDurationSec: 8.6, ...opts }); return { onArrival: (t, a) => p.onArrival(t, a), onEnd: (t, a) => p.onEnd(t, a), onUnderrun: () => p.onUnderrun(), shouldPlay: (t, x) => t >= 7 && p.shouldPlay(t, x) }; };
  const naive = simulatePlayback([...a1, ...a2], gated({}));
  const aware = simulatePlayback([...a1, ...a2], gated({ segmentBoundarySec: 3.9, segmentGapSec: TTS_MODEL.ttfa }));
  assert.ok(naive.underruns > 0, "sin la información del límite, corta");
  assert.equal(aware.underruns, 0, "con la información del límite, no corta");
  assert.ok(aware.firstAudibleSec > naive.firstAudibleSec, "a cambio arranca un poco después");
  // sin segmentos (Fase 2), la política no cambia
  const single = ttsArrivals(7, 8.6);
  assert.deepEqual(simulatePlayback(single, new AdaptiveStartPolicy({ estDurationSec: 8.6 })), simulatePlayback(single, new AdaptiveStartPolicy({ estDurationSec: 8.6, segmentBoundarySec: null })));
});

test("12/18/21/31: respuesta rechazada → audio anticipado descartado, nunca enviado, cola liberada", async () => {
  const e = makeEngine();
  gw.state.plan = () => ({ parts: [[0, pcm(4410)], [300, pcm(4410)]], end: "ok" });
  const r = await respond(e, ["Che, qué buena pregunta la tuya, de verdad.", " Y algo que el filtro rechaza."], { validate: false });
  assert.equal(r, null);
  assert.equal(e.overlap.stats.discarded, 1);
  assert.equal(e.ws.sent.length, 0, "nada llegó a la vista");
  await until(() => e.tts.sglang._active.size === 0);
  // la cola quedó libre: una respuesta normal sale enseguida
  const t0 = Date.now();
  const n = await e.tts.synth("Otra respuesta normal de prueba.", { emotion: "happy", source: "live", returnMeta: true });
  assert.match(n.url, /^stream:/); assert.ok(Date.now() - t0 < 1500);
});

test("7/16/17: texto final distinto → descarta y sintetiza la respuesta completa una sola vez (sin palabras duplicadas)", async () => {
  const e = makeEngine();
  const r = await respond(e, ["Che, qué buena pregunta la tuya, de verdad.", " Te cuento."], { filtered: "Juan, qué buena pregunta la tuya, de verdad. Te cuento." });
  assert.equal(r.overlap, undefined);
  assert.equal(e.overlap.stats.discarded, 1);
  const inputs = gw.state.requests.map(q => q.body.input);
  assert.equal(inputs.at(-1), "[laughing] Juan, qué buena pregunta la tuya, de verdad. Te cuento.");
  await e.tts.play(r.url, "x");
  await until(() => jsonMsgs(e.ws).some(m => m.type === "tts_stream_end"));
  assert.equal(new Set(frames(e.ws).map(f => f.id)).size, 1, "solo suena la respuesta final, una vez");
  e.tts.audioEnded();
});

test("19/20: cancelación durante el LLM y durante la síntesis anticipada — sin fallback ni audio", async () => {
  const e = makeEngine();
  gw.state.plan = () => ({ parts: [[0, pcm(4410)], [400, pcm(4410)]], end: "ok" });
  const spec = e.overlap.begin({ providerId: "agy", source: "live", build: s => ({ ttsText: s, emotion: "happy" }) });
  e.overlap.feed(spec, "Che, qué buena pregunta la tuya, de verdad. Te");
  await sleep(80);
  e.tts.stop(); e.tts.stopSpeculation(); // lo que hará TTSEngine.stop() al integrarse
  e.overlap.abort(spec, "stop");
  await until(() => e.tts.sglang._active.size === 0);
  assert.equal(e.edge, 0); assert.equal(e.ws.sent.length, 0);
  assert.equal(e.tts.streams.streams.size, 0);
});

test("22: backend ocupado rechaza especulación y TTL vencido no hace HTTP", async () => {
  const e = makeEngine();
  gw.state.plan = () => ({ parts: [[0, pcm(1000)], [250, pcm(1000)]], end: "ok" });
  await e.tts.synth("Una respuesta anterior ocupando la cola.", { emotion: "happy", source: "live", returnMeta: true });
  assert.equal(e.tts.speculateStart({ ttsText: "Che, qué buena pregunta la tuya, de verdad.", emotion: "happy" }), null, "con backend ocupado no especula");
  await until(() => !e.tts._sglang.busy);
  const h = e.tts.speculateStart({ ttsText: "Che, qué buena pregunta la tuya, de verdad.", emotion: "happy", expiresAt: Date.now() - 10 });
  await h.seg1.catch(() => {});
  assert.equal(gw.state.requests.length, 1, "el especulativo vencido nunca llegó al gateway");
  e.tts.discardSpeculation(h, "vencida");
});

test("H02 (23/24): sin reserva de cola — prioridad superior desplaza especulación activa o esperando; backend ocupado al confirmar descarta", async () => {
  // 1) Especulación activa (chat, rank=4) desplazada por donación (rank=0)
  const e1 = makeEngine();
  gw.state.plan = () => ({ parts: [[0, pcm(2000)], [300, pcm(2000)]], end: "ok" });
  const h1 = e1.tts.speculateStart({ ttsText: "Che, qué buena pregunta la tuya, de verdad.", emotion: "happy", rank: 4 });
  assert.ok(h1);
  assert.equal(e1.tts.speculateStart({ ttsText: "Otra oración especulativa larga.", emotion: "happy", rank: 4 }), null, "solo una especulación viva");
  await sleep(30);
  const tDon = Date.now();
  const don = await e1.tts.synth("¡Gracias por la donación gigante!", { emotion: "happy", source: "live", returnMeta: true, rank: 0 });
  assert.match(don.url, /^stream:/);
  assert.equal(h1.cancelled, true, "la donación abortó la especulación de menor prioridad");
  assert.equal(e1.tts._spec, null);
  assert.ok(Date.now() - tDon < 200, "la donación no esperó 120s ni el fin de la especulación");

  // 2) Especulación que ya terminó seg1 libera el backend; si llega donación esperando confirmación, se descarta
  const e2 = makeEngine();
  gw.state.plan = () => ({ parts: [[0, pcm(2000)], [20, pcm(2000)]], end: "ok" });
  const h2 = e2.tts.speculateStart({ ttsText: "Che, qué buena pregunta la tuya, de verdad.", emotion: "happy", rank: 4 });
  await h2.seg1;
  assert.equal(e2.tts._sglang.busy, false, "al terminar la 1.ª oración el backend queda libre");
  const don2 = await e2.tts.synth("¡Regalo prioritario!", { emotion: "happy", source: "live", returnMeta: true, rank: 0 });
  assert.match(don2.url, /^stream:/);
  assert.equal(h2.cancelled, true, "donación desplazó la especulación que esperaba en memoria");
  assert.equal(await e2.tts.commitSpeculation(h2, "Te cuento."), null);

  // 3) Al confirmar con resto, si el backend está ocupado con OTRO trabajo de igual prioridad, se descarta y sigue camino normal
  const e3 = makeEngine();
  gw.state.plan = () => ({ parts: [[0, pcm(2000)], [180, pcm(2000)]], end: "ok" });
  const h3 = e3.tts.speculateStart({ ttsText: "Che, qué buena pregunta la tuya, de verdad.", emotion: "happy", rank: 4 });
  await h3.seg1;
  assert.equal(e3.tts._sglang.busy, false);
  const other = await e3.tts.synth("Otro chat de igual prioridad arrancó mientras el LLM pensaba.", { emotion: "happy", source: "live", returnMeta: true, rank: 4 });
  assert.match(other.url, /^stream:/);
  assert.equal(h3.cancelled, false, "igual prioridad no la descarta anticipadamente");
  assert.equal(await e3.tts.commitSpeculation(h3, "Resto de la respuesta."), null, "al confirmar con backend ocupado, descarta para no cortar");
  e3.tts.discardSpeculation(h3, "backend ocupado");
});

test("25/26: falla del primer segmento — antes de validar: camino normal; después de confirmar: un solo fallback al sonar", async () => {
  const e = makeEngine();
  let n = 0;
  gw.state.plan = () => (++n === 1 ? { status: 500 } : { parts: [[0, pcm(2000)]], end: "ok" });
  // el LLM sigue escribiendo un rato después de la 1.ª oración: el 500 llega antes de validar
  const r = await respond(e, ["Che, qué buena pregunta la tuya, de verdad.", " Te cuento", " que sí", " de verdad."], { gapMs: 60 });
  assert.equal(r.overlap, undefined, "no se usó lo especulado");
  assert.match(r.url, /^stream:/, "la respuesta completa salió por el camino normal (SGLang)");
  assert.equal(e.edge, 0);
  assert.equal(e.overlap.stats.discarded, 1);

  // el 500 llega DESPUÉS de confirmar (todavía no se envió audio): al sonar, el fallback de Fase 2 actúa una vez
  const e2 = makeEngine();
  gw.state.plan = () => ({ status: 500 });
  const h = e2.tts.speculateStart({ ttsText: "Che, qué buena pregunta la tuya, de verdad.", emotion: "happy" });
  const c = await e2.tts.commitSpeculation(h, "Te cuento.");
  assert.match(c.url, /^stream:/);
  await e2.tts.play(c.url, "Che, qué buena pregunta la tuya, de verdad. Te cuento.");
  await until(() => e2.plays.length === 1);
  await sleep(50);
  assert.deepEqual(e2.plays, ["/tts/edge.mp3"]); assert.equal(e2.edge, 1, "un solo fallback, sin audio parcial duplicado");
  assert.equal(frames(e2.ws).length, 0);
  e2.tts.audioEnded();
});

test("28/29/35: WAV, Discord o sin vistas compatibles → no se especula (motivo registrado)", async () => {
  for (const [opts, src, reason] of [
    [{ mode: "wav" }, "live", /Streaming PCM/],
    [{}, "discord", /Discord/],
    [{ clients: [] }, "live", /vista/],
  ]) {
    const e = makeEngine(opts);
    assert.equal(e.overlap.begin({ providerId: "agy", source: src, build: s => ({ ttsText: s, emotion: "happy" }) }), null);
    assert.match(Object.keys(e.overlap.stats.skipped)[0], reason);
  }
});

test("32/33/34: sin secretos, recursos liberados y métricas de aprovechado/descartado", async () => {
  const e = makeEngine();
  const r = await respond(e, ["Che, qué buena pregunta la tuya, de verdad.", " Te cuento que sí."]);
  await e.tts.play(r.url, "x");
  await until(() => jsonMsgs(e.ws).some(m => m.type === "tts_stream_end"));
  e.tts.onTtsDone(e.ws, r.url);
  await respond(e, ["Che, qué buena pregunta la tuya, de verdad.", " Otra cosa."], { validate: false });
  await until(() => e.tts.sglang._active.size === 0);
  const st = e.overlap.publicStats();
  assert.equal(st.started, 2); assert.equal(st.used, 1); assert.equal(st.discarded, 1);
  assert.ok(st.savedMsTotal >= 0 && st.wastedGpuMsTotal >= 0);
  assert.equal(e.tts.streams.streams.size, 0); assert.equal(e.tts.streams.totalBytes, 0);
  assert.equal(e.tts._spec, null);
  const all = [...e.logs, JSON.stringify(st), JSON.stringify(e.tts.sglang.getPublicConfig())].join("\n");
  assert.doesNotMatch(all, new RegExp(KEY));
});

test("3: configuraciones anteriores sin 'overlap' quedan apagadas; se puede activar y volver atrás", () => {
  const c = new SGLangTTSClient({ config: { url: gw.url, apiKey: KEY }, settingsFile: join(tmp, "s.json"), allowLoopbackHttp: true, logger: null });
  assert.equal(c.config.overlap, false);
  c.updateConfig({ overlap: true });
  assert.equal(new SGLangTTSClient({ config: {}, settingsFile: join(tmp, "s.json"), logger: null }).config.overlap, true);
  c.updateConfig({ overlap: false });
  assert.equal(new SGLangTTSClient({ config: {}, settingsFile: join(tmp, "s.json"), logger: null }).config.overlap, false);
});

test("H04/H05/H06: retardo 250ms evita especular ráfagas; ventaja <1s descarta; cambio de expresividad descarta", async () => {
  // 1) Ráfaga completa antes de 250ms → no dispara speculateStart
  const tts1 = fakeTts();
  const c1 = new OverlapController({ tts: tts1, enabled: () => true, delayMs: 250, minHeadStartMs: 1000 });
  const s1 = c1.begin({ providerId: "claude-code", source: "live", build });
  c1.feed(s1, "Che, qué buena pregunta la tuya, de verdad. Te cuento que sí.");
  assert.equal(tts1.calls.start.length, 0, "todavía está esperando el retardo de 250ms");
  assert.equal(await c1.finish(s1, { filtered: "Che, qué buena pregunta la tuya, de verdad. Te cuento que sí.", emotion: "happy" }), null);
  assert.equal(tts1.calls.start.length, 0, "nunca especuló porque llegó todo junto");
  assert.equal(c1.stats.skipped["texto completo antes de disparar especulación"], 1);

  // 2) Ventaja < 1000ms con resto → descarta por ventaja insuficiente; sin resto → aprovecha
  let clock = 1000;
  const tts2 = fakeTts();
  const c2 = new OverlapController({ tts: tts2, enabled: () => true, now: () => clock, delayMs: 0, minHeadStartMs: 1000 });
  const s2 = c2.begin({ providerId: "claude-code", source: "live", build });
  c2.feed(s2, "Che, qué buena pregunta la tuya, de verdad. Te");
  clock += 400; // solo 400ms de ventaja (< 1000ms)
  assert.equal(await c2.finish(s2, { filtered: "Che, qué buena pregunta la tuya, de verdad. Te cuento.", emotion: "happy" }), null);
  assert.match(Object.keys(c2.stats.discardReasons)[0], /ventaja insuficiente/);

  // 3) Cambio de expresividad entre especulación y finish → descarta
  const e3 = makeEngine({ delayMs: 0, minHeadStartMs: 0 });
  const s3 = e3.overlap.begin({ providerId: "claude-code", source: "live", build: s => ({ ttsText: e3.tts.ttsTextOf(s), emotion: "happy" }) });
  e3.overlap.feed(s3, "Che, qué buena pregunta la tuya, de verdad. Te");
  e3.tts.sglang.updateConfig({ expressive: !e3.tts.sglang.config.expressive });
  assert.equal(await e3.overlap.finish(s3, { filtered: "Che, qué buena pregunta la tuya, de verdad. Te cuento.", emotion: "happy" }), null);
  assert.match(Object.keys(e3.overlap.stats.discardReasons)[0], /expresividad/);
});

test("H07: política con segmentStartedAt mide TTFA del 2.º pedido desde que arranca, no desde el fin de la 1.ª oración", async () => {
  const { AdaptiveStartPolicy } = await import("../public/tts-stream/buffer-policy.js");
  // 1.ª oración de 3.0s terminó de generarse en t=4.0; el LLM recién validó y arrancó el 2.º pedido en t=8.0
  const p = new AdaptiveStartPolicy({ estDurationSec: 7.0, segmentBoundarySec: 3.0, segmentGapSec: 3.8, segmentStartedAt: 8.0 });
  p.onArrival(2.0, 1.5);
  p.onArrival(4.0, 3.0);
  // En t=8.0, sinceLast sería 4.0s (> 3.8s), pero desde segmentStartedAt (8.0) pasaron 0s → debe pedir ~3.8s + deficit + safety (> 3.0s que hay en búfer)
  assert.equal(p.shouldPlay(8.0, 0), false, "no arranca prematuramente cuando el 2.º pedido recién empezó");
  // En t=11.0 (ya pasaron 3.0s del TTFA del 2.º pedido, faltan 0.8s + deficit + safety < 3.0s en búfer), ya puede arrancar sin cortar
  assert.ok(p.requiredBufferSec(11.0, 0) < p.requiredBufferSec(8.0, 0));
});

test("H03 (30/31): flujo real de StreamChatManager._prepare y _respond — memoria y acciones una vez, rechazo/reemplazo/vacío descartan", async () => {
  const src = readFileSync(new URL("../server.js", import.meta.url), "utf8");
  const lines = src.split(/\r?\n/);
  const start = lines.findIndex(l => l.startsWith("class StreamChatManager"));
  const end = lines.findIndex((l, i) => i > start && l === "}");
  const classCode = lines.slice(start, end + 1).join("\n");

  const e = makeEngine({ delayMs: 0, minHeadStartMs: 0 });
  const events = [];
  let nextChunks = [];
  let nextThrow = null;

  const ai = {
    llm: { name: "chat", providerId: "claude-code", modelId: "claude-code:haiku" },
    kb: { search: () => [] },
    memory: { getNick: () => null, trackViewer: (u, t) => events.push({ type: "track", u, t }) },
    _stripContextLeaks: t => t,
    remember: (role, text, who) => events.push({ type: "remember", role, text, who }),
    learnFromExchange: (u, q, a) => events.push({ type: "learn", u, q, a }),
    async *chatStream() {
      if (nextThrow) throw nextThrow;
      for (const ch of nextChunks) { await sleep(15); yield ch; }
    },
  };
  const live = { mood: 60, progress: 0, goals: [], adjustMood: () => false, broadcast: () => {}, markTold: () => {} };
  const suggestions = { add: (u, raw, summary) => events.push({ type: "suggestion", u, summary }) };
  const wishes = { add: (what) => events.push({ type: "wish", what }) };
  const filters = { data: { autoLearn: false }, filterOutput: t => t };

  const StreamChatManager = new Function(
    "deps",
    `const {
      uuidv4, live, evaluateContextualSafety, hasHostileEvidence, evaluateIgnoreAction,
      sanitizeModelResponse, orders, TRUSTED_SOURCES, STREAMER_NAME, STREAMER_ALIAS, CREATOR_RULE,
      GOAL_ASK_RE, GOAL_NUM_WORD_RE, SUGGEST_RE, NICK_RE, NICK_STOP, NO_NAME_RE, GREETING_RE, LEADING_GREETING_RE, TEA_RE,
      ANIM_RE, GESTURE_RE, SUG_RE, REFUSAL_LINES, GESTURE_HINT, PRI, LIVE, LOOK_TTL_MS,
      hasPendingSixSevenGoal, hasSixSeven, pickCreatorGesture, pickGesture, normalizeSixSevenSpeech,
      has3dAvatar, goalDistance, isModelFilterError, isModelRefusal, dropLeadingName,
      getMoodState, resolveSpeakingEmotion, extractEmotionTag, evaluateMoodPenalty,
      ai, tts, filters, wishes, suggestions, broadcast, logTranscript, followUp, discord,
      overlap, UNSAFE_SPEECH, findFirstSentence
    } = deps;
    ${classCode}
    return StreamChatManager;`
  )({
    uuidv4: () => "12345678-0000-0000-0000-000000000000",
    live, evaluateContextualSafety, hasHostileEvidence, evaluateIgnoreAction, sanitizeModelResponse,
    orders: { limits: () => "" }, TRUSTED_SOURCES: new Set(["panel", "streamer"]), STREAMER_NAME: "Lau", STREAMER_ALIAS: "Lau",
    CREATOR_RULE: "", GOAL_ASK_RE: /$a/, GOAL_NUM_WORD_RE: /$a/, SUGGEST_RE: /$a/, NICK_RE: /$a/, NICK_STOP: new Set(),
    NO_NAME_RE: /$a/, GREETING_RE: /hola/i, LEADING_GREETING_RE: /^/, TEA_RE: /\bt[eé]\b/i,
    ANIM_RE: /\[ANIMACI[OÓ]N:\s*([^\]]+)\]/gi, GESTURE_RE: /\[GESTO:\s*([^\]]+)\]/gi, SUG_RE: /\[SUGERENCIA:\s*([^\]]+)\]/i,
    REFUSAL_LINES: ["No voy a responder eso."], GESTURE_HINT: {}, PRI: { gift: 0, call: 1, look: 2, event: 3, chat: 4, idle: 5 },
    LIVE: { moodDecayMs: 60000 }, LOOK_TTL_MS: 30000,
    hasPendingSixSevenGoal: () => false, hasSixSeven: () => false, pickCreatorGesture: () => null,
    pickGesture: w => (w.toLowerCase() === "saludar" ? "saludar" : null), normalizeSixSevenSpeech: s => s,
    has3dAvatar: () => false, goalDistance: () => "", isModelFilterError: m => /blocked/i.test(String(m || "")),
    isModelRefusal: t => /^como modelo de lenguaje/i.test(String(t || "")), dropLeadingName,
    getMoodState, resolveSpeakingEmotion, extractEmotionTag, evaluateMoodPenalty,
    ai, tts: e.tts, filters, wishes, suggestions,
    broadcast: msg => events.push({ type: "broadcast", msg }), logTranscript: () => {}, followUp: { speaking: () => {}, finished: () => {} },
    discord: { on: false }, overlap: e.overlap, UNSAFE_SPEECH, findFirstSentence,
  });

  const mgr = Object.assign(Object.create(StreamChatManager.prototype), {
    _askedNick: new Set(), _viewerStrikes: new Map(), _ignoredUsers: new Map(), _deferred: new Map(), queue: [],
    _nickNote: () => "", _autoNickNote: () => "", _pump: () => {},
  });
  e.tts._onDone = () => { mgr._audioDoneResolve?.(); };

  // Caso A (30): respuesta válida de viewer con streaming + acción [SUGERENCIA: ...] — memoria y acción una sola vez, nada parcial
  nextChunks = ["[EMOCIÓN: happy] Che, qué buena idea tiraste hoy para el stream.", " Lo anoto ahora mismo. [SUGERENCIA: jugar torneo]"];
  const prepared = await mgr._prepare({ username: "ana", text: "hagamos torneo", platform: "tiktok", opts: {}, rank: 4, at: Date.now() }, ai.llm);
  assert.ok(prepared);
  assert.match(prepared.url, /^stream:/);
  assert.equal(e.overlap.stats.used, 1, "usó la especulación en _prepare real");
  assert.equal(events.filter(x => x.type === "remember").length, 0, "ninguna memoria guardada antes de _respond");
  assert.equal(events.filter(x => x.type === "suggestion").length, 1, "acción ejecutada una sola vez tras validar");
  assert.equal(typeof prepared.trace.firstDeltaMs, "number");
  assert.equal(typeof prepared.trace.firstSentenceMs, "number");
  assert.equal(typeof prepared.trace.llmDoneMs, "number");
  assert.ok(prepared.trace.overlapWindowMs >= 0);

  const playPromise = mgr._respond({ username: "ana", text: "hagamos torneo", platform: "tiktok", opts: {}, rank: 4, ready: prepared });
  await until(() => jsonMsgs(e.ws).some(m => m.type === "tts_stream_end"));
  e.tts.onTtsDone(e.ws, prepared.url);
  await playPromise;
  assert.equal(events.filter(x => x.type === "remember" && x.role === "assistant").length, 1, "memoria guardada exactamente una vez");

  // Caso B (31): respuesta bloqueada (viewer normal) → audio especulativo descartado, cero memoria, cero audio
  const wsBefore = e.ws.sent.length;
  const remBefore = events.filter(x => x.type === "remember").length;
  nextChunks = ["Che, qué buena pregunta la tuya, de verdad.", " ..."];
  nextThrow = Object.assign(new Error("blocked"), { modelBlocked: true });
  // Primero forzamos que arranque especulación con chunks y luego falle en el último chunk
  nextThrow = null;
  let chunkN = 0;
  const origStream = ai.chatStream;
  ai.chatStream = async function* () {
    yield "Che, qué buena pregunta la tuya, de verdad. ";
    await sleep(15);
    yield "Te ";
    await sleep(15);
    chunkN++;
    throw Object.assign(new Error("blocked"), { modelBlocked: true });
  };
  const blockedRes = await mgr._prepare({ username: "troll", text: "algo malo", platform: "tiktok", opts: {}, rank: 4, at: Date.now() }, ai.llm);
  assert.equal(blockedRes, null, "viewer bloqueado devuelve null");
  assert.equal(chunkN, 1);
  assert.equal(e.overlap.stats.discarded, 1, "especulación descartada ante bloqueo");
  assert.equal(e.ws.sent.length, wsBefore, "ningún audio enviado a OBS");
  assert.equal(events.filter(x => x.type === "remember").length, remBefore, "ninguna memoria guardada");

  // Caso C (31): respuesta reemplazada (donor bloqueado → REFUSAL_LINES) → especulación descartada y reemplazo sintetizado una vez
  const reqsBefore = gw.state.requests.length;
  const repRes = await mgr._prepare({ username: "donor", text: "algo", platform: "tiktok", opts: { donor: true }, rank: 0, at: Date.now() }, ai.llm);
  assert.ok(repRes && repRes.refused);
  assert.equal(e.overlap.stats.discarded, 2, "especulación descartada antes del reemplazo");
  assert.ok(gw.state.requests.length > reqsBefore);
  assert.match(gw.state.requests.at(-1).body.input, /no voy a responder eso/i);

  // Caso D (31): respuesta vacía tras limpiar marcas → descarta especulación y devuelve null
  ai.chatStream = async function* () {
    yield "Che, qué buena pregunta la tuya, de verdad. ";
    await sleep(15);
    yield "Y";
  };
  filters.filterOutput = () => ""; // simula filtro de salida que vacía todo
  // Mejor: texto solo marcas después de limpiar
  filters.filterOutput = orig => orig;
  ai.chatStream = origStream;
  nextChunks = ["[IGNORAR: 5]"];
  const emptyRes = await mgr._prepare({ username: "x", text: "hola", platform: "tiktok", opts: { rawPrompt: true }, rank: 4, at: Date.now() }, ai.llm);
  assert.equal(emptyRes, null);
});

test("H08: panel harness — /api/tts/sglang/config persiste overlap y renderSglangOverlap muestra compatibilidad real", async () => {
  const { default: express } = await import("express");
  const e = makeEngine();
  let activeProvider = "claude-code";
  e.tts.sglang.getOverlapStats = () => ({ ...e.overlap.publicStats(), activeProvider });
  const app = express();
  app.use(express.json());
  app.get("/api/tts/sglang/config", (_req, res) => res.json({ ok: true, config: e.tts.sglang.getPublicConfig() }));
  app.post("/api/tts/sglang/config", (req, res) => {
    const { url, timeoutMs, expressive, fallback, playbackMode, overlap } = req.body || {};
    res.json({ ok: true, config: e.tts.sglang.updateConfig({ url, timeoutMs, expressive, fallback, playbackMode, overlap }) });
  });
  const srv = await new Promise(r => { const s = app.listen(0, "127.0.0.1", () => r(s)); });
  try {
    const port = srv.address().port;
    const g0 = await fetch(`http://127.0.0.1:${port}/api/tts/sglang/config`).then(r => r.json());
    assert.equal(g0.config.overlap, false, "apagada por defecto");
    assert.equal(g0.config.overlapStats.activeProvider, "claude-code");
    assert.doesNotMatch(JSON.stringify(g0), new RegExp(KEY));

    const p1 = await fetch(`http://127.0.0.1:${port}/api/tts/sglang/config`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ overlap: true }),
    }).then(r => r.json());
    assert.equal(p1.config.overlap, true);

    // Extraer renderSglangOverlap de public/panel.html y verificar indicador por proveedor
    const panelHtml = readFileSync(new URL("../public/panel.html", import.meta.url), "utf8");
    const m = panelHtml.match(/function renderSglangOverlap\(cfg\) \{[\s\S]*?\n  \}/);
    assert.ok(m, "renderSglangOverlap existe en panel.html");
    const els = { sglangOverlap: { checked: false }, sglangOverlapInfo: { style: {}, innerText: "" } };
    const render = new Function("document", `${m[0]}; return renderSglangOverlap;`)({ getElementById: id => els[id] });

    render({ overlap: true, playbackMode: "wav", overlapStats: { started: 0, used: 0, discarded: 0, activeProvider: "claude-code" } });
    assert.match(els.sglangOverlapInfo.innerText, /Streaming PCM/);

    render({ overlap: true, playbackMode: "stream", overlapStats: { started: 2, used: 1, discarded: 1, savedMsTotal: 1500, wastedGpuMsTotal: 400, activeProvider: "agy" } });
    assert.match(els.sglangOverlapInfo.innerText, /Antigravity.*pendiente de comprobación/i);
    assert.match(els.sglangOverlapInfo.innerText, /Anticipadas: 2 · usadas: 1 · descartadas: 1/);

    render({ overlap: true, playbackMode: "stream", overlapStats: { started: 0, used: 0, discarded: 0, activeProvider: "codex-cli" } });
    assert.match(els.sglangOverlapInfo.innerText, /codex-cli.*no disponible/i);

    render({ overlap: true, playbackMode: "stream", overlapStats: { started: 0, used: 0, discarded: 0, activeProvider: "opencode" } });
    assert.match(els.sglangOverlapInfo.innerText, /opencode.*no disponible/i);
  } finally {
    await new Promise(r => srv.close(r));
  }
});

test("Verificación crítica: falla del 2.º segmento tras emitir PCM → partial sin Edge; cambio de URL/TTL descarta; scheduler serializa bajo carrera", async () => {
  // 1) Falla del 2.º segmento DESPUÉS de haber emitido PCM del 1.º → tts_stream_end(partial:true), cero Edge, sin duplicar
  const e1 = makeEngine();
  let reqN = 0;
  gw.state.plan = () => (++reqN === 1 ? { parts: [[0, pcm(4410)], [10, pcm(4410)]], end: "ok" } : { status: 500 });
  const h1 = e1.tts.speculateStart({ ttsText: "Primera oración lista y emitida.", emotion: "happy", rank: 4 });
  await h1.seg1;
  const c1 = await e1.tts.commitSpeculation(h1, "Segunda oración que falla.");
  assert.match(c1.url, /^stream:/);
  await e1.tts.play(c1.url, "Primera oración lista y emitida. Segunda oración que falla.");
  await until(() => jsonMsgs(e1.ws).some(m => m.type === "tts_stream_end"));
  const endMsg = jsonMsgs(e1.ws).find(m => m.type === "tts_stream_end");
  assert.equal(endMsg.partial, true, "registra fallo parcial");
  assert.equal(e1.edge, 0, "no dispara fallback a Edge tras emitir PCM");
  assert.deepEqual(e1.plays, [], "no duplica la primera oración por WAV/Edge");
  assert.ok(frames(e1.ws).length > 0);
  e1.tts.audioEnded();

  // 2) Cambio de URL o TTL vencido al confirmar → descarta
  const e2 = makeEngine();
  gw.state.plan = () => ({ parts: [[0, pcm(2000)]], end: "ok" });
  const hUrl = e2.tts.speculateStart({ ttsText: "Oración con URL original.", emotion: "happy", rank: 4 });
  await hUrl.seg1;
  e2.tts.sglang.config.url = "http://127.0.0.1:1";
  assert.equal(await e2.tts.commitSpeculation(hUrl, "Resto."), null, "cambio de URL invalida");
  e2.tts.discardSpeculation(hUrl, "url");

  const e3 = makeEngine();
  const hTtl = e3.tts.speculateStart({ ttsText: "Oración con TTL corto.", emotion: "happy", rank: 4, expiresAt: Date.now() + 30 });
  await hTtl.seg1;
  await sleep(40);
  assert.equal(await e3.tts.commitSpeculation(hTtl, "Resto."), null, "TTL vencido al confirmar invalida");
  e3.tts.discardSpeculation(hTtl, "ttl");

  // 3) Concurrencia en SglangScheduler con overlap desactivado: 5 pedidos simultáneos → maxInFlight === 1 siempre
  const e4 = makeEngine();
  gw.state.maxInFlight = 0;
  gw.state.plan = () => ({ parts: [[15, pcm(1000)]], end: "ok" });
  const jobs = await Promise.all([
    e4.tts.synth("Turno uno.", { emotion: "happy", source: "live", returnMeta: true, rank: 4 }),
    e4.tts.synth("Turno dos.", { emotion: "happy", source: "live", returnMeta: true, rank: 4 }),
    e4.tts.synth("Turno tres.", { emotion: "happy", source: "live", returnMeta: true, rank: 0 }),
    e4.tts.synth("Turno cuatro.", { emotion: "happy", source: "live", returnMeta: true, rank: 4 }),
    e4.tts.synth("Turno cinco.", { emotion: "happy", source: "live", returnMeta: true, rank: 4 }),
  ]);
  await until(() => !e4.tts._sglang.busy);
  assert.equal(gw.state.maxInFlight, 1, "nunca ejecuta dos inferencias SGLang simultáneas");
  assert.equal(jobs.filter(j => j.url).length, 5, "ningún trabajo autorizado fue cancelado");
  e4.tts.stop();
});
