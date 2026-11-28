// Fase 4A: banco de pruebas offline reproducible de 3 niveles + Perfiles A–E + impacto de telemetría.
// Uso:
//   node scripts/benchmark-latency.mjs                         → ejecuta Niveles 1, 2 y 3 (mocks) + replay histórico separado
//   node scripts/benchmark-latency.mjs --level=1|2|3|all       → selecciona nivel
//   node scripts/benchmark-latency.mjs --profiles=A,B,C,D,E    → selecciona perfiles
//   node scripts/benchmark-latency.mjs --impact                → mide impacto de telemetría detallada OFF vs ON
//   node scripts/benchmark-latency.mjs --export                → exporta JSON/JSONL/CSV/MD/HTML en logs/benchmarks/<runId>/
// Restricciones cumplidas:
//   - Todos los proveedores externos (Edge, Fish, SGLang, LLM) están mockeados localmente en puertos efímeros (nunca 3000).
//   - No modifica preferencias reales de NYRA, ni memoria real, ni JSON privados.
//   - No inicia Colab ni consume GPU remota.
//   - Separa estrictamente categorías: "mock", "simulation" y "historical_replay" jamás se mezclan en el mismo agregado.
import { createServer } from "node:http";
import { execSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { decode as decodeMsgPack } from "@msgpack/msgpack";

import { TurnTracer } from "../lib/observability/turn-tracer.js";
import { aggregateTurns, buildRunDocument, compareProfiles, exportBenchmarkRun, exportCapturedTraces, formatOverlappingTimeline, renderMarkdownReport } from "../lib/observability/benchmark-report.js";
import { FishAudioClient } from "../lib/tts/fish-audio-client.js";
import { SGLangTTSClient } from "../lib/tts/sglang-tts-client.js";
import { SglangStreamManager, decodeFrame, pcmToWav } from "../lib/tts/sglang-stream.js";
import { SglangScheduler } from "../lib/tts/sglang-scheduler.js";
import { installSpeculation } from "../lib/tts/sglang-speculation.js";
import { OverlapController, UNSAFE_SPEECH, findFirstSentence } from "../lib/tts/llm-tts-overlap.js";
import { AdaptiveStartPolicy, WholeFilePolicy, simulatePlayback, oracleStartSec } from "../public/tts-stream/buffer-policy.js";
import { extractEmotionTag, getMoodState, resolveSpeakingEmotion, evaluateMoodPenalty } from "../lib/ai/emotion-state.js";
import { sanitizeModelResponse } from "../lib/moderation/response-sanitizer.js";
import { evaluateIgnoreAction, hasHostileEvidence } from "../lib/moderation/ignore-guard.js";
import { evaluateContextualSafety } from "../lib/moderation/safety-guard.js";
import { dropLeadingName } from "../lib/ai/rag-utils.js";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
export const CORPUS = JSON.parse(readFileSync(new URL("../tests/fixtures/latency-corpus-es-ar.json", import.meta.url), "utf8"));
const ARRIVAL_FIXTURE = JSON.parse(readFileSync(new URL("../tests/fixtures/sglang-arrival-traces.json", import.meta.url), "utf8"));

const MOCK_KEY = "sk-bench-local-mock-0000";
const PCM_HEADERS = { "Content-Type": "application/octet-stream", "X-Audio-Sample-Rate": "44100", "X-Audio-Channels": "1", "X-Audio-Encoding": "pcm_s16le" };
const sleep = ms => new Promise(r => setTimeout(r, ms));

export const PROFILES = {
  A: { id: "A", name: "Referencia actual (Edge mock)", engine: "edge", playbackMode: "wav", overlap: false, provider: "claude-code", voiceNote: "Voz Edge distinta (referencia operativa)" },
  B: { id: "B", name: "Fish Speech original (WAV mock)", engine: "fish", playbackMode: "wav", overlap: false, provider: "claude-code", voiceNote: "Fish S2 Pro R20 (mock calibrado)" },
  C: { id: "C", name: "SGLang WAV (mock)", engine: "sglang", playbackMode: "wav", overlap: false, provider: "claude-code", voiceNote: "Fish S2 Pro R20 vía SGLang WAV" },
  D: { id: "D", name: "SGLang PCM adaptativo (mock)", engine: "sglang", playbackMode: "stream", overlap: false, provider: "claude-code", voiceNote: "Fish S2 Pro R20 vía SGLang PCM" },
  E: { id: "E", name: "SGLang PCM + overlap (mock)", engine: "sglang", playbackMode: "stream", overlap: true, provider: "claude-code", voiceNote: "Fish S2 Pro R20 vía SGLang PCM + overlap" },
};

function makePcmBuffer(samples) {
  const b = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i++) b.writeInt16LE(Math.round(Math.sin(i / 11) * 7000), i * 2);
  return b;
}

// H02: duración y cantidad de muestras idénticas para B, C y D sobre el mismo texto (quita tags [emocion] antes de medir).
// Edge (Perfil A) usa su propia velocidad operativa documentada.
export function syntheticAudioForText(rawText, { cps = 16, sampleRate = 44100 } = {}) {
  const clean = String(rawText || "").replace(/^\[[^\]]+\]\s*/, "").trim();
  const audioSec = Number(Math.max(0.5, clean.length / cps).toFixed(3));
  const totalSamples = Math.round(audioSec * sampleRate);
  return { cleanChars: clean.length, audioSec, totalSamples, sampleRate };
}

export const ACCELERATED_PARAMS = {
  timeScaleMode: "accelerated_mock",
  note: "Escala acelerada para pruebas offline rápidas (~50× más rápida que L4). NO representa tiempos reales de producción ni de GPU L4.",
  overlapDelayMs: 15,
  overlapMinHeadStartMs: 20,
  llmFirstDeltaMs: 12,
  llmDeltaGapMs: 22,
  llmFullOnlyTotalMs: 140,
  sglangFirstChunkMs: 35,
  sglangChunkGapMs: 18,
  sglangWavTotalMs: 75,
  fishWavTotalMs: 95,
  edgeDelayMs: 28,
  edgeCps: 17.5,
  fishSglangCps: 16,
  playbackScaleMsPerAudioSec: 18,
  wavDecodeDelayMs: 8,
  lookTtlMs: 8000,
  sglangTimeoutMs: 5000,
  backoffNote: "Retry-After=0 en mock 429; backoff real de producción es Retry-After o attempt*500ms",
  adaptivePolicyNote: "AdaptiveStartPolicy mantiene su fórmula real en segundos de audio sobre tiempos de llegada acelerados del mock",
};

export const REPRESENTATIVE_PARAMS = {
  timeScaleMode: "representative_1x_mock",
  note: "Escenario representativo en escala temporal 1:1 con umbrales EXACTOS de producción de Fase 3 (delayMs=250, minHeadStartMs=1000) y reproducción simulada a 1000 ms/s. Sus intervalos PCM sintéticos (1600 ms) NO reproducen la distribución histórica de SGLang L4 (~1.846s 1.er bloque, ~2.09s posteriores, ~3.2-3.3s entre bloques, RTF ~1.54-1.63), y el mock de navegador simula duración por temporizador sin consumo continuo real de muestras (underruns simulados, no medidos).",
  overlapDelayMs: 250,
  overlapMinHeadStartMs: 1000,
  llmFirstDeltaMs: 1200,
  llmDeltaGapMs: 380,
  sglangFirstChunkMs: 3780,
  sglangChunkGapMs: 1600,
  sglangWavTotalMs: 6500,
  fishSglangCps: 16,
  playbackScaleMsPerAudioSec: 1000,
  playbackSimulationMode: "timer_duration_simulated_not_real_pcm_drain",
  wavDecodeDelayMs: 35,
  lookTtlMs: 8000,
  sglangTimeoutMs: 60000,
};

function getCommitShort() {
  try { return execSync("git rev-parse --short HEAD", { cwd: ROOT, stdio: ["ignore", "pipe", "ignore"] }).toString("utf8").trim(); }
  catch { return "unknown"; }
}

// ─── Harness aislado de clases reales de server.js ────────────────────────────
let _cachedClasses = null;
function extractServerClass(src, className) {
  const lines = src.split(/\r?\n/);
  const start = lines.findIndex(l => l.startsWith(`class ${className} {`));
  let depth = 0, end = start;
  for (let i = start; i < lines.length; i++) {
    for (const ch of lines[i]) { if (ch === "{") depth++; else if (ch === "}") depth--; }
    if (i > start && depth === 0) { end = i; break; }
  }
  return lines.slice(start, end + 1).join("\n");
}

function loadServerClasses() {
  if (_cachedClasses) return _cachedClasses;
  const src = readFileSync(join(ROOT, "server.js"), "utf8");
  _cachedClasses = {
    ttsCode: extractServerClass(src, "TTSEngine"),
    chatCode: extractServerClass(src, "StreamChatManager"),
  };
  return _cachedClasses;
}

// Servidor HTTP mock efímero (puerto 0, jamás 3000) que sirve Fish MsgPack WAV + SGLang WAV/PCM
export async function startMockTtsServer({
  sglangFirstChunkMs = 35,
  sglangChunkGapMs = 18,
  sglangWavTotalMs = 70,
  fishWavTotalMs = 95,
  jitterMs = 0,
  slowFactor = 1,
  fail429Once = false,
  failSecondSegment = false,
  timeoutNext = false,
} = {}) {
  let reqCount = 0;
  let sglang429Left = fail429Once ? 1 : 0;
  let timeoutLeft = timeoutNext ? 1 : 0;
  let segCalls = 0;
  const delay = ms => {
    const j = jitterMs ? ((reqCount * 7) % (jitterMs + 1)) : 0; // jitter determinista reproducible
    return Math.max(1, Math.round((ms + j) * slowFactor));
  };
  const srv = createServer(async (req, res) => {
    reqCount += 1;
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const bodyBuf = Buffer.concat(chunks);

    if (req.url === "/v1/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok", model: "fishaudio/s2-pro", voice: "nyra-clonada", busy: false, sample_rate: 44100, channels: 1, encoding: "pcm_s16le" }));
      return;
    }

    if (req.url === "/v1/tts") {
      // Fish Speech original mock (WAV 44.1kHz): decodifica MsgPack real para generar la MISMA duración/muestras que SGLang
      let decoded = {};
      try { decoded = decodeMsgPack(bodyBuf) || {}; } catch { }
      const { totalSamples } = syntheticAudioForText(decoded.text || "", { cps: 16, sampleRate: 44100 });
      const wav = pcmToWav([makePcmBuffer(totalSamples)], 44100);
      await sleep(delay(fishWavTotalMs));
      if (res.destroyed) return;
      res.writeHead(200, { "Content-Type": "audio/wav", "Content-Length": wav.length });
      res.end(wav);
      return;
    }

    if (req.url === "/v1/audio/speech") {
      if (timeoutLeft > 0) { timeoutLeft -= 1; return; } // deja colgada para probar timeout
      if (sglang429Left > 0) {
        sglang429Left -= 1;
        res.writeHead(429, { "Content-Type": "application/json", "Retry-After": "0" });
        res.end(JSON.stringify({ error: "busy" }));
        return;
      }
      let payload = {};
      try { payload = JSON.parse(bodyBuf.toString("utf8")); } catch { }
      const { audioSec, totalSamples } = syntheticAudioForText(payload.input || "", { cps: 16, sampleRate: 44100 });

      if (payload.stream === true) {
        segCalls += 1;
        res.writeHead(200, PCM_HEADERS);
        const numChunks = Math.max(2, Math.min(6, Math.ceil(audioSec / 2.0)));
        const perChunk = Math.floor(totalSamples / numChunks);
        await sleep(delay(sglangFirstChunkMs));
        for (let i = 0; i < numChunks; i++) {
          if (res.destroyed) return;
          if (failSecondSegment && segCalls >= 2 && i === 1) { res.destroy(); return; }
          const n = i === numChunks - 1 ? totalSamples - perChunk * (numChunks - 1) : perChunk;
          res.write(makePcmBuffer(n));
          if (i < numChunks - 1) await sleep(delay(sglangChunkGapMs));
        }
        res.end();
        return;
      }

      const wav = pcmToWav([makePcmBuffer(totalSamples)], 44100);
      await sleep(delay(sglangWavTotalMs));
      if (res.destroyed) return;
      res.writeHead(200, { "Content-Type": "audio/wav", "Content-Length": wav.length });
      res.end(wav);
      return;
    }

    res.writeHead(404);
    res.end();
  });
  await new Promise(r => srv.listen(0, "127.0.0.1", r));
  const { port } = srv.address();
  return {
    url: `http://127.0.0.1:${port}`,
    getRequests: () => reqCount,
    close: () => new Promise(r => srv.close(r)),
  };
}

// Vista OBS/navegador mockeada con política adaptativa real y soporte WAV + PCM
function createMockBrowserHub(tts, {
  playbackScaleMsPerAudioSec = 18, // acelerado para tests/benchmarks offline rápidos y reproducibles
  wavDecodeDelayMs = 8,
  dropAfterFirstFrame = false,
  slowBufferedAmount = 0,
  extraViews = 0,
} = {}) {
  const clients = [];
  const makeView = (idx, { mute = false } = {}) => {
    const ws = {
      readyState: 1,
      ttsStream: true,
      ttsMute: mute,
      bufferedAmount: slowBufferedAmount,
      _viewId: `mock_obs_${idx}`,
      sent: [],
    };
    clients.push(ws);
    return ws;
  };
  const primary = makeView(1, { mute: false });
  for (let i = 0; i < extraViews; i++) makeView(i + 2, { mute: i % 2 === 1 });

  const sessions = new Map(); // id -> { policy, received, started, startMs, t0, chunks, ended }
  const hub = {
    clients: () => clients.filter(c => c.readyState === 1),
    send(ws, data) {
      ws.sent.push(data);
      if (ws !== primary) {
        // Vistas secundarias solo acusan recibo al finalizar la primaria (salvo que sean audibles)
        if (typeof data === "string") {
          const msg = JSON.parse(data);
          if (msg.type === "tts_stream_end" && !ws.ttsMute) {
            setTimeout(() => tts.onTtsDone(ws, msg.url), 12);
          }
        }
        return;
      }
      if (Buffer.isBuffer(data)) {
        const f = decodeFrame(data);
        if (!f) return;
        const s = sessions.get(f.id);
        if (!s || s.closed) return;
        s.chunks += 1;
        s.received += f.pcm.length / 2;
        const nowSec = (performance.now() - s.t0) / 1000;
        s.policy.onArrival(nowSec, s.received / 44100);
        if (!s.started && s.chunks >= 1) {
          s.started = true;
          s.playStartedAt = performance.now();
          const clientMs = Math.round(s.playStartedAt - s.t0);
          tts.onPlaybackStarted(ws, {
            id: f.id, url: s.url, mode: "stream", startMethod: "worklet_first_render", mute: ws.ttsMute,
            clientRenderStartMs: clientMs, bufferAtStartSec: Number((s.received / 44100).toFixed(3)),
            contextRate: 44100, contextState: "running", outputLatencyMs: 12,
          });
        }
        if (dropAfterFirstFrame && s.chunks === 1) {
          ws.readyState = 3;
          tts.streamClientGone(ws);
        }
        return;
      }
      const msg = JSON.parse(data);
      if (msg.type === "tts_stream_start") {
        sessions.set(msg.id, {
          id: msg.id, url: msg.url, t0: performance.now(), playStartedAt: null, received: 0, chunks: 0, started: false, closed: false,
          policy: new AdaptiveStartPolicy({ estDurationSec: msg.estDurationSec || 2 }),
        });
      } else if (msg.type === "tts_stream_end") {
        const s = sessions.get(msg.id);
        if (!s || s.closed) return;
        s.closed = true;
        const audioSec = (msg.totalSamples || s.received) / 44100;
        const totalPlayMs = Math.max(5, Math.round(audioSec * playbackScaleMsPerAudioSec));
        const alreadyPlayedMs = s.playStartedAt ? Math.max(0, performance.now() - s.playStartedAt) : 0;
        const playWait = Math.max(4, Math.round(totalPlayMs - alreadyPlayedMs));
        setTimeout(() => {
          const stats = {
            firstChunkMs: 5, bufferingStartMs: 1, audibleStartMs: 10, effectiveRenderStartMs: 10,
            bufferAtStartSec: Number((s.received / 44100).toFixed(3)),
            underruns: 0, underrunMs: 0, underrunRecoveries: 0,
            receivedSamples: msg.totalSamples || s.received, playedSamples: msg.totalSamples || s.received,
            contextRate: 44100, contextState: "running", outputLatencyMs: 12,
            streamEndMs: Math.round(performance.now() - s.t0),
            playbackEndMs: Math.round(performance.now() - s.t0) + playWait,
            result: msg.partial ? "error" : "done",
          };
          tts.streams.recordClientStats({ id: msg.id, stats });
          tts.onPlaybackStats(ws, { id: msg.id, url: msg.url, stats });
          tts.onTtsDone(ws, msg.url);
        }, playWait);
      } else if (msg.type === "tts_stream_cancel") {
        const s = sessions.get(msg.id);
        if (s) s.closed = true;
      }
    },
  };

  const broadcast = msg => {
    if (msg.type === "tts_play") {
      const t0 = performance.now();
      const tr = tts.tracer?.resolveTurnByUrl?.(msg.url);
      const audioSec = tr?.tts?.audioDurationSec || 1.5;
      setTimeout(() => {
        tts.onPlaybackStarted(primary, {
          url: msg.url, mode: "wav", startMethod: "html_audio_onplaying", mute: Boolean(msg.mute),
          clientRenderStartMs: wavDecodeDelayMs, contextRate: 44100, contextState: "running", outputLatencyMs: 12,
        });
        const playDur = Math.max(10, Math.round(audioSec * playbackScaleMsPerAudioSec));
        setTimeout(() => {
          tts.onPlaybackStats(primary, {
            url: msg.url, mode: "wav",
            stats: {
              effectiveRenderStartMs: wavDecodeDelayMs, playbackEndMs: Math.round(performance.now() - t0),
              decodedDurationSec: audioSec, contextRate: 44100, contextState: "running", outputLatencyMs: 12,
              underruns: 0, underrunMs: 0, underrunRecoveries: 0, result: "done",
            },
          });
          tts.onTtsDone(primary, msg.url);
        }, playDur);
      }, wavDecodeDelayMs);
    }
  };
  return { hub, broadcast, primary, clients };
}

export function createIsolatedHarness({
  profileId = "C",
  serverUrl = "",
  edgeDelayMs = ACCELERATED_PARAMS.edgeDelayMs,
  edgeCps = ACCELERATED_PARAMS.edgeCps,
  tracer = null,
  discordOn = false,
  shareTtsEngine = false,
  discordTtsEngine = "edge",
  llmProviderId = null,
  llmDelays = { firstDeltaMs: ACCELERATED_PARAMS.llmFirstDeltaMs, deltaGapMs: ACCELERATED_PARAMS.llmDeltaGapMs, totalMs: ACCELERATED_PARAMS.llmFullOnlyTotalMs },
  overlapOpts = { delayMs: ACCELERATED_PARAMS.overlapDelayMs, minHeadStartMs: ACCELERATED_PARAMS.overlapMinHeadStartMs },
  hubOpts = {},
  sglangTimeoutMs = ACCELERATED_PARAMS.sglangTimeoutMs,
} = {}) {
  const dir = mkdtempSync(join(tmpdir(), "nyra-bench-"));
  const prof = PROFILES[profileId] || PROFILES.C;
  const { ttsCode, chatCode } = loadServerClasses();
  const CONFIG = {
    ttsEngine: prof.engine,
    shareTtsEngine,
    discordTtsEngine,
    useSystemTTS: false,
    edgeVoice: "es-AR-ElenaNeural",
    murf: { apiKey: "" },
    fish: { url: serverUrl, apiKey: MOCK_KEY, referenceId: "r20-mock", timeoutMs: sglangTimeoutMs, fallback: true },
    sglang: { url: serverUrl, apiKey: MOCK_KEY, playbackMode: prof.playbackMode, overlap: prof.overlap, timeoutMs: sglangTimeoutMs, fallback: true },
  };
  const silentLogger = { log() {}, warn() {}, error() {} };
  const PatchedSglang = class extends SGLangTTSClient {
    constructor(opts) { super({ ...opts, settingsFile: join(dir, "tts_settings.json"), allowLoopbackHttp: true, logger: silentLogger }); }
  };
  const PatchedFish = class extends FishAudioClient {
    constructor(opts) { super({ ...opts, settingsFile: join(dir, "tts_settings.json"), profilesFile: join(dir, "fish_voices.json"), voicesDir: join(dir, "voices"), allowedHosts: ["127.0.0.1"] }); }
  };
  const PatchedStreams = class extends SglangStreamManager {
    constructor(opts) { super({ ...opts, logger: silentLogger }); }
  };
  const discordStub = {
    on: discordOn,
    stopAudio() {},
    play: async () => true,
  };
  const TTSEngine = new Function(
    "deps",
    `const { CONFIG, FishAudioClient, SGLangTTSClient, SglangStreamManager, SglangScheduler, join, __dirname, mkdirSync, renameSync, rmSync, readdirSync, unlinkSync, spawn, createInterface, applyPhonetics, exec, discord } = deps;
     ${ttsCode}
     return TTSEngine;`
  )({
    CONFIG, FishAudioClient: PatchedFish, SGLangTTSClient: PatchedSglang, SglangStreamManager: PatchedStreams, SglangScheduler,
    join, __dirname: dir, mkdirSync, renameSync, rmSync, readdirSync, unlinkSync,
    spawn: () => {}, createInterface: () => ({ on: () => {} }), applyPhonetics: s => s,
    exec: (_c, cb) => cb?.(), discord: discordStub,
  });

  const tts = installSpeculation(new TTSEngine(), { applyPhonetics: s => s });
  // Mock Edge para jamás salir a red en Fase 4A
  tts._synthEdge = async function (text, { turnId = null, requestedEngine = "edge", fallback = false } = {}) {
    const jobIds = this.tracer?.startTtsJob?.(turnId, { requestedEngine, effectiveEngine: "edge", playbackMode: "wav", fallback });
    const t0 = performance.now();
    if (jobIds) this.tracer?.updateTtsJob?.(turnId, jobIds.ttsJobId, { requestStart: true, queueWaitMs: 0 });
    await sleep(edgeDelayMs);
    const outDir = join(dir, "public", "tts");
    mkdirSync(outDir, { recursive: true });
    const id = `${Date.now()}${Math.floor(Math.random() * 1000)}`;
    const { audioSec, totalSamples } = syntheticAudioForText(text, { cps: edgeCps, sampleRate: 44100 });
    const wav = pcmToWav([makePcmBuffer(totalSamples)], 44100);
    writeFileSync(join(outDir, `${id}.mp3`), wav);
    const url = `/tts/${id}.mp3`;
    if (jobIds) {
      this.tracer?.updateTtsJob?.(turnId, jobIds.ttsJobId, {
        url, headers: true, firstByte: true, httpBodyEnd: true,
        httpTotalMs: Math.round(performance.now() - t0), bytesReceived: wav.length, samplesReceived: totalSamples,
        audioDurationSec: audioSec, result: "ok", fallbackUsed: fallback,
      });
    }
    return url;
  };

  const activeTracer = tracer || new TurnTracer({ enabled: true, category: "mock" });
  tts.tracer = activeTracer;
  const browser = createMockBrowserHub(tts, hubOpts);
  tts.streams.hub = browser.hub;
  tts._broadcast = browser.broadcast;

  const overlap = new OverlapController({
    tts,
    enabled: () => tts.sglang.config.overlap === true,
    delayMs: overlapOpts.delayMs,
    minHeadStartMs: overlapOpts.minHeadStartMs,
    tracer: activeTracer,
  });

  const PRI = { gift: 0, call: 1, look: 2, event: 3, chat: 4, idle: 5 };
  const effectiveProvider = llmProviderId || prof.provider;
  const replyQueue = [];
  const ai = {
    llm: { name: "chat", providerId: effectiveProvider, modelId: `${effectiveProvider}:mock`, proc: true },
    llmAhead: { name: "chat", providerId: effectiveProvider, modelId: `${effectiveProvider}:mock`, proc: true },
    companion: { name: "companion", providerId: effectiveProvider, modelId: `${effectiveProvider}:mock`, proc: true },
    memory: { getNick: () => null, trackViewer() {}, data: { viewers: {} }, save() {} },
    kb: { search: () => [] },
    chatLearning: { addExchange() {} },
    _stripContextLeaks: s => s,
    remember() {},
    learnFromExchange() {},
    async *chatStream(_prompt, _user, client) {
      const reply = replyQueue.shift() || "Buenas noches, todo bien por acá. Arrancamos con todo.";
      const isIncremental = client?.providerId === "claude-code" || client?.providerId === "agy";
      if (!isIncremental) {
        await sleep(llmDelays.totalMs || 50);
        yield reply;
        return;
      }
      await sleep(llmDelays.firstDeltaMs || 10);
      // Emitir deltas incrementales realistas (~4 palabras por chunk) para que la 1.ª oración + inicio de la 2.ª lleguen a mitad del LLM
      const words = reply.split(/(\s+)/);
      const step = 6; // ~3 palabras + espacios por chunk
      for (let i = 0; i < words.length; i += step) {
        yield words.slice(i, i + step).join("");
        if (i + step < words.length) await sleep(llmDelays.deltaGapMs || 14);
      }
    },
  };

  const live = { mood: 50, goals: [], progress: 0, adjustMood: () => false, broadcast() {}, markTold() {} };
  const nextUp = (q, now) => {
    const ready = q.filter(i => i.ready && !(i.holdUntil > now)).sort((a, b) => a.rank - b.rank || a.at - b.at);
    return ready[0] || null;
  };
  let counter = 0;
  const StreamChatManager = new Function(
    "deps",
    `const {
      uuidv4, live, evaluateContextualSafety, PRI, LOOK_TTL_MS, SLOW_MS, READY_AHEAD, IDLE_FACT_MS, PROMO_MS, TIKTOK_REFRESH_MS,
      STREAMER_ALIAS, STREAMER_NAME, CREATOR_RULE, SUGGEST_RE, TRUSTED_SOURCES, GESTURE_HINT,
      REFUSAL_LINES, ANIM_RE, GESTURE_RE, SUG_RE, GREETING_RE, LEADING_GREETING_RE, TEA_RE,
      NICK_RE, NICK_STOP, NO_NAME_RE, AUTO_NICK_AFTER, GOAL_ASK_RE, GOAL_NUM_WORD_RE,
      hasHostileEvidence, sanitizeModelResponse, evaluateIgnoreAction, nextUp, discord, audioSource,
      hasPendingSixSevenGoal, hasSixSeven, pickCreatorGesture, pickGesture, normalizeSixSevenSpeech,
      has3dAvatar, goalDistance, isModelFilterError, isModelRefusal, dropLeadingName,
      getMoodState, resolveSpeakingEmotion, extractEmotionTag, evaluateMoodPenalty,
      ai, tts, filters, wishes, suggestions, broadcast, logTranscript, followUp, orders, evolution,
      overlap, UNSAFE_SPEECH, findFirstSentence
    } = deps;
    ${chatCode}
    return StreamChatManager;`
  )({
    uuidv4: () => `bench${String(++counter).padStart(4, "0")}-0000-0000-0000-000000000000`,
    live, evaluateContextualSafety, PRI, LOOK_TTL_MS: 8000, SLOW_MS: 4000, READY_AHEAD: 2,
    IDLE_FACT_MS: 0, PROMO_MS: 0, TIKTOK_REFRESH_MS: 9999999,
    STREAMER_ALIAS: "Yisus", STREAMER_NAME: "Yisus", CREATOR_RULE: "", SUGGEST_RE: /$^/,
    TRUSTED_SOURCES: new Set(["streamer", "panel"]), GESTURE_HINT: {}, REFUSAL_LINES: ["Paso de esa."],
    ANIM_RE: /$^/g, GESTURE_RE: /$^/g, SUG_RE: /$^/, GREETING_RE: /hola|buenas/i,
    LEADING_GREETING_RE: /^$/, TEA_RE: /$^/,
    NICK_RE: /$^/, NICK_STOP: new Set(), NO_NAME_RE: /$^/, AUTO_NICK_AFTER: 999,
    GOAL_ASK_RE: /$^/, GOAL_NUM_WORD_RE: /$^/,
    hasHostileEvidence, sanitizeModelResponse, evaluateIgnoreAction, nextUp, discord: discordStub,
    audioSource: u => u, hasPendingSixSevenGoal: () => false, hasSixSeven: () => false,
    pickCreatorGesture: () => null, pickGesture: () => null, normalizeSixSevenSpeech: s => s,
    has3dAvatar: () => false, goalDistance: () => "", isModelFilterError: () => false, isModelRefusal: () => false,
    dropLeadingName, getMoodState, resolveSpeakingEmotion, extractEmotionTag, evaluateMoodPenalty,
    ai, tts, filters: { filterOutput: s => s, data: { autoLearn: false } },
    wishes: { add() {} }, suggestions: { add() {} }, broadcast: browser.broadcast,
    logTranscript() {}, followUp: { speaking() {}, finished() {} }, orders: { limits: () => "" }, evolution: { note() {} },
    overlap, UNSAFE_SPEECH, findFirstSentence,
  });

  const chat = new StreamChatManager();
  chat.tracer = activeTracer;
  tts._onDone = () => chat.onAudioDone();

  const cleanup = () => {
    try { tts.stop(); } catch { }
    try { rmSync(dir, { recursive: true, force: true }); } catch { }
  };

  return { dir, tts, chat, overlap, tracer: activeTracer, browser, ai, replyQueue, PRI, cleanup };
}

// ─── Nivel 1: TTS aislado (textos fijos en A, B, C, D) + Replay histórico separado ──
export async function runLevel1({ profiles = ["A", "B", "C", "D"], warmups = 1 } = {}) {
  const srv = await startMockTtsServer();
  const byProfile = {};
  const allTraces = [];
  try {
    for (const pid of profiles.filter(p => p !== "E")) {
      const tracer = new TurnTracer({ enabled: true, category: "mock", maxHistory: 200 });
      const h = createIsolatedHarness({ profileId: pid, serverUrl: srv.url, tracer });
      try {
        const items = [
          ...Array.from({ length: warmups }, () => ({ ...CORPUS.ttsIsolatedTexts[0], warmup: true })),
          ...CORPUS.ttsIsolatedTexts.map(x => ({ ...x, warmup: false })),
        ];
        for (const item of items) {
          const turnId = tracer.startTurn({
            source: "live", platform: "benchmark_l1", priority: 4,
            warmup: item.warmup, requestedEngine: PROFILES[pid].engine, playbackMode: PROFILES[pid].playbackMode,
          });
          tracer.markEnqueued(turnId, { priority: 4 });
          tracer.markPrepareStart(turnId);
          const synth = await h.tts.synth(item.text, { emotion: item.emotion, source: "live", returnMeta: true, turnId });
          tracer.markPrepareEnd(turnId);
          tracer.markPlaybackTurnStart(turnId);
          const doneP = new Promise(r => { h.tts._onDone = r; });
          await h.tts.play(synth.url, item.text, false, { turnId });
          await doneP;
          tracer.markPlaybackTurnEnd(turnId);
          const closed = tracer.finishTurn(turnId, { state: "completed" });
          closed.profile = pid;
          closed.corpusId = item.id;
          allTraces.push(closed);
        }
        byProfile[`L1_${pid}_${PROFILES[pid].engine}_${PROFILES[pid].playbackMode}`] = aggregateTurns(tracer.getHistory());
      } finally {
        h.cleanup();
      }
    }
  } finally {
    await srv.close();
  }

  // H02: tabla explícita de equivalencia de audio (duración, muestras y sample rate por perfil y caso)
  const audioEquivalenceTable = allTraces.filter(t => !t.warmup).map(t => {
    const job = t.tts.jobs[0] || {};
    const samples = job.samplesReceived || Math.round((t.metrics.audioDurationSec || 0) * 44100);
    return {
      profile: t.profile,
      caseId: t.corpusId,
      engine: t.effectiveEngine,
      mode: t.playbackMode,
      audioDurationSec: t.metrics.audioDurationSec,
      samples,
      sampleRate: 44100,
      startMs: t.metrics.estimatedPlaybackStartRelMs,
      endMs: t.metrics.playbackEndRelMs,
    };
  });

  // H05: Replay histórico SGLang-Omni L4 en tres vistas separadas (categoría: "historical_replay").
  // Excluye Fish Speech original, warmups y trazas WAV de un solo punto; nunca fabrica largos faltantes.
  const allSglangStreamingMeasure = ARRIVAL_FIXTURE.traces.filter(t => t.engine === "sglang-omni" && t.streaming === true && t.kind === "measure");
  const histPool = allSglangStreamingMeasure.filter(t => Number.isFinite(t.chars) && t.chars > 0);
  const withoutCharsCount = allSglangStreamingMeasure.length - histPool.length;
  const profilesIncluded = [...new Set(histPool.map(t => t.profile).filter(Boolean))];
  const PRIOR_CPS = 14;
  const learnCps = (cps, chars, audioSec) => Math.max(8, Math.min(25, 0.7 * cps + 0.3 * (chars / audioSec)));

  const buildHistTraces = (list, { calibCount = 0 } = {}) => {
    let cps = PRIOR_CPS;
    const wavTraces = [], pcmTraces = [], oracleTraces = [];
    const trW = new TurnTracer({ enabled: true, category: "historical_replay", now: () => 0, maxHistory: 200 });
    const trP = new TurnTracer({ enabled: true, category: "historical_replay", now: () => 0, maxHistory: 200 });
    const trO = new TurnTracer({ enabled: true, category: "historical_replay", now: () => 0, maxHistory: 200 });
    list.forEach((t, idx) => {
      const audioSec = t.trace.at(-1)[1];
      // Estimador conservador de producción (<= 14 cps), aprendiendo SOLO de respuestas anteriores
      const estSec = t.chars / Math.min(cps, PRIOR_CPS);
      const wavSim = simulatePlayback(t.trace, new WholeFilePolicy());
      const pcmSim = simulatePlayback(t.trace, new AdaptiveStartPolicy({ estDurationSec: estSec }));
      const oraStart = oracleStartSec(t.trace);
      if (idx >= calibCount) {
        const mk = (tr, mode, startS, endS, underruns = 0, underrunSec = 0) => {
          const tid = tr.startTurn({ source: "live", platform: "historical_l4", requestedEngine: "sglang", playbackMode: mode, receivedAtMonoMs: 0 });
          tr.markPlaybackDispatched(tid, { mode, participants: 1, audibleParticipants: 1, monoMs: startS * 1000 });
          tr.recordPlaybackStartAck(tid, { mode, startMethod: mode === "wav" ? "html_audio_onplaying" : "worklet_first_render", monoMs: startS * 1000 });
          tr.recordClientPlaybackStats(tid, { stats: { underruns, underrunMs: Math.round(underrunSec * 1000), result: "done" }, monoMs: endS * 1000 });
          tr.markPlaybackEnd(tid, { result: "done", monoMs: endS * 1000 });
          const c = tr.finishTurn(tid, { state: "completed", monoMs: endS * 1000 });
          c.metrics.audioDurationSec = Number(audioSec.toFixed(3));
          c.corpusId = t.id;
          return c;
        };
        wavTraces.push(mk(trW, "wav", wavSim.firstAudibleSec, wavSim.endSec));
        pcmTraces.push(mk(trP, "stream", pcmSim.firstAudibleSec, pcmSim.endSec, pcmSim.underruns, pcmSim.underrunSec));
        oracleTraces.push(mk(trO, "stream", oraStart, oraStart + audioSec));
      }
      // Recién al terminar la sesión el estimador causal aprende de ella
      cps = learnCps(cps, t.chars, audioSec);
    });
    return {
      wav: aggregateTurns(wavTraces),
      pcm: aggregateTurns(pcmTraces),
      oracle: aggregateTurns(oracleTraces),
      pairedComparison: compareProfiles(wavTraces, pcmTraces, x => x.metrics.estimatedPlaybackStartRelMs, { pairKey: x => x.corpusId }),
    };
  };

  const calibCount = Math.floor(histPool.length * 0.25); // 10 calibración / 30 evaluación (igual que Fase 2)
  const descriptiveAll40 = buildHistTraces(histPool, { calibCount: 0 });
  const causalPhase2 = buildHistTraces(histPool, { calibCount });

  const historicalReplay = {
    category: "historical_replay",
    metadata: {
      totalSglangStreamingMeasure: allSglangStreamingMeasure.length,
      withCharsCount: histPool.length,
      withoutCharsCount,
      withoutCharsPolicy: "Excluidas del replay con estimación por texto porque estDurationSec requiere longitud de texto (no se fabrican largos).",
      profilesIncluded,
      voiceReference: "R20",
      estimatorPolicy: "aprendido conservador <=14 cps (producción Fase 2, aprende únicamente de sesiones anteriores)",
      calibrationSplit: { calibrationCount: calibCount, evaluationCount: histPool.length - calibCount },
    },
    descriptiveAll40: {
      note: "Replay descriptivo sobre las 40 trazas SGLang-Omni con chars (sin separación de calibración).",
      aggregates: { SGLang_WAV_N40: descriptiveAll40.wav, SGLang_PCM_adaptive_N40: descriptiveAll40.pcm },
      pairedComparison: descriptiveAll40.pairedComparison,
    },
    causalPhase2: {
      note: "Reproducción exacta de la evaluación causal de Fase 2: 10 trazas solo para calibración + 30 evaluadas.",
      aggregates: { SGLang_WAV_causal30: causalPhase2.wav, SGLang_PCM_adaptive_causal30: causalPhase2.pcm },
      pairedComparison: causalPhase2.pairedComparison,
    },
    theoreticalOracle: {
      note: "Referencia retrospectiva teórica NO causal (oracleStartSec) sobre las mismas 30 trazas evaluadas.",
      aggregate: causalPhase2.oracle,
    },
  };

  return { aggregates: byProfile, traces: allTraces, audioEquivalenceTable, historicalReplay };
}

// ─── Nivel 2: LLM + TTS determinista (C vs D vs E, rápido/lento/ráfaga/full-only) ──
export async function runLevel2({ profiles = ["C", "D", "E"] } = {}) {
  const srv = await startMockTtsServer({ sglangFirstChunkMs: 35, sglangChunkGapMs: 18, sglangWavTotalMs: 75 });
  const byProfile = {};
  const allTraces = [];
  const subset = CORPUS.turns.filter(t => t.source === "live" && !t.displacesTurnId).slice(0, 6);
  try {
    for (const pid of profiles.filter(p => ["C", "D", "E"].includes(p))) {
      const tracer = new TurnTracer({ enabled: true, category: "mock", maxHistory: 200 });
      const h = createIsolatedHarness({
        profileId: pid, serverUrl: srv.url, tracer,
        llmDelays: { firstDeltaMs: 12, deltaGapMs: 22, totalMs: 140 },
      });
      try {
        for (const turn of subset) {
          h.replyQueue.push(turn.reply);
          const turnId = tracer.startTurn({
            source: "live", platform: turn.platform, priority: turn.priority,
            requestedProvider: PROFILES[pid].provider, requestedEngine: "sglang", playbackMode: PROFILES[pid].playbackMode,
          });
          tracer.markEnqueued(turnId, { priority: turn.priority });
          const prep = await h.chat._prepare({
            username: "viewer_bench", text: turn.input, platform: turn.platform,
            opts: { donor: Boolean(turn.donor) }, rank: turn.priority, at: Date.now(), turnId,
          }, h.ai.llm);
          tracer.markPlaybackTurnStart(turnId);
          await h.chat._respond({ username: "viewer_bench", text: turn.input, platform: turn.platform, opts: {}, rank: turn.priority, ready: prep, turnId });
          const rec = tracer.getHistory().at(-1);
          if (rec) { rec.profile = pid; rec.corpusId = turn.id; allTraces.push(rec); }
        }
        byProfile[`L2_${pid}_${PROFILES[pid].playbackMode}${PROFILES[pid].overlap ? "_overlap" : ""}`] = aggregateTurns(tracer.getHistory());
      } finally {
        h.cleanup();
      }
    }
  } finally {
    await srv.close();
  }
  const cTraces = allTraces.filter(t => t.profile === "C");
  const dTraces = allTraces.filter(t => t.profile === "D");
  const eTraces = allTraces.filter(t => t.profile === "E");
  const comparisons = [];
  if (cTraces.length && dTraces.length) comparisons.push({ label: "L2 C (WAV) vs D (PCM)", ...compareProfiles(cTraces, dTraces, t => t.metrics.estimatedPlaybackStartRelMs, { pairKey: t => t.corpusId }) });
  if (dTraces.length && eTraces.length) comparisons.push({ label: "L2 D (PCM) vs E (PCM+overlap)", ...compareProfiles(dTraces, eTraces, t => t.metrics.estimatedPlaybackStartRelMs, { pairKey: t => t.corpusId }) });
  return { aggregates: byProfile, traces: allTraces, comparisons };
}

// ─── Nivel 3: Turno completo con colas, prioridades, Discord y fallos controlados ──
export async function runLevel3({ profiles = ["A", "B", "C", "D", "E"] } = {}) {
  const srv = await startMockTtsServer({ sglangFirstChunkMs: 30, sglangChunkGapMs: 15, sglangWavTotalMs: 65, fishWavTotalMs: 85 });
  const byProfile = {};
  const allTraces = [];
  const sampleTurns = CORPUS.turns.slice(0, 8);
  try {
    for (const pid of profiles) {
      const tracer = new TurnTracer({ enabled: true, category: "mock", maxHistory: 200 });
      const h = createIsolatedHarness({
        profileId: pid, serverUrl: srv.url, tracer,
        discordOn: true, shareTtsEngine: false, discordTtsEngine: "edge",
        llmDelays: { firstDeltaMs: 12, deltaGapMs: 18, totalMs: 110 },
      });
      try {
        for (const t of sampleTurns) {
          h.replyQueue.push(t.reply);
          h.chat._enqueue(t.source === "discord" ? "streamer" : "viewer_synth", t.input, t.platform, { donor: Boolean(t.donor) }, t.priority, 0);
        }
        // Esperar a que drene toda la cola real de StreamChatManager
        const deadline = Date.now() + 15000;
        while ((h.chat.queue.length > 0 || h.chat._isResponding) && Date.now() < deadline) {
          await sleep(15);
        }
        await sleep(30);
        const hist = tracer.getHistory();
        for (const rec of hist) { rec.profile = pid; allTraces.push(rec); }
        byProfile[`L3_${pid}_${PROFILES[pid].engine}_${PROFILES[pid].playbackMode}${PROFILES[pid].overlap ? "_overlap" : ""}`] = aggregateTurns(hist);
      } finally {
        h.cleanup();
      }
    }
  } finally {
    await srv.close();
  }
  return { aggregates: byProfile, traces: allTraces };
}

// ─── H01: Escenario representativo 1:1 con umbrales EXACTOS de producción (solo mocks) ──
// Usa delayMs=250, minHeadStartMs=1000, TTFA=3780ms, bloques 1600ms, LLM ~6.5s y reproducción 1:1 (1000ms/s).
// Corre C, D y E en paralelo en 3 harnesses aislados para completar en ~15s de reloj real.
export async function runRepresentativeScaleScenario() {
  const turn = CORPUS.turns.find(t => t.id === "c03_pregunta_tecnica") || CORPUS.turns[2];
  const runOne = async pid => {
    const srv = await startMockTtsServer({
      sglangFirstChunkMs: REPRESENTATIVE_PARAMS.sglangFirstChunkMs,
      sglangChunkGapMs: REPRESENTATIVE_PARAMS.sglangChunkGapMs,
      sglangWavTotalMs: REPRESENTATIVE_PARAMS.sglangWavTotalMs,
    });
    const tracer = new TurnTracer({ enabled: true, category: "mock", maxHistory: 20 });
    const h = createIsolatedHarness({
      profileId: pid,
      serverUrl: srv.url,
      tracer,
      llmDelays: { firstDeltaMs: REPRESENTATIVE_PARAMS.llmFirstDeltaMs, deltaGapMs: REPRESENTATIVE_PARAMS.llmDeltaGapMs, totalMs: 6500 },
      overlapOpts: { delayMs: REPRESENTATIVE_PARAMS.overlapDelayMs, minHeadStartMs: REPRESENTATIVE_PARAMS.overlapMinHeadStartMs },
      hubOpts: { playbackScaleMsPerAudioSec: REPRESENTATIVE_PARAMS.playbackScaleMsPerAudioSec, wavDecodeDelayMs: REPRESENTATIVE_PARAMS.wavDecodeDelayMs },
      sglangTimeoutMs: REPRESENTATIVE_PARAMS.sglangTimeoutMs,
    });
    try {
      h.replyQueue.push(turn.reply);
      const turnId = tracer.startTurn({
        source: "live", platform: turn.platform, priority: turn.priority,
        requestedProvider: "claude-code", requestedEngine: "sglang", playbackMode: PROFILES[pid].playbackMode,
      });
      tracer.markEnqueued(turnId, { priority: turn.priority });
      const prep = await h.chat._prepare({ username: "viewer_rep", text: turn.input, platform: turn.platform, opts: {}, rank: turn.priority, at: Date.now(), turnId }, h.ai.llm);
      tracer.markPlaybackTurnStart(turnId);
      await h.chat._respond({ username: "viewer_rep", text: turn.input, platform: turn.platform, opts: {}, rank: turn.priority, ready: prep, turnId });
      const rec = tracer.getHistory().at(-1);
      if (rec) { rec.profile = pid; rec.corpusId = turn.id; }
      return rec;
    } finally {
      h.cleanup();
      await srv.close();
    }
  };

  const [traceC, traceD, traceE] = await Promise.all(["C", "D", "E"].map(runOne));
  return {
    params: REPRESENTATIVE_PARAMS,
    corpusId: turn.id,
    traces: { C: traceC, D: traceD, E: traceE },
  };
}

// ─── Benchmark de impacto de instrumentación (microbenchmark TurnTracer OFF vs ON) ──
export function runImpactBenchmark({ iterations = 300, longRunIterations = 1000 } = {}) {
  const runBatch = (enabled, count) => {
    let clk = 0;
    const tracer = new TurnTracer({ enabled, category: "mock", now: () => (clk += 1), maxHistory: 200, maxEventsPerTurn: 128 });
    const memBefore = process.memoryUsage();
    const t0 = performance.now();
    for (let i = 0; i < count; i++) {
      const tid = tracer.startTurn({ source: "live", platform: "tiktok", priority: 4, requestedProvider: "claude-code", requestedEngine: "sglang", playbackMode: "stream" });
      tracer.markEnqueued(tid, { priority: 4 });
      tracer.markPrepareStart(tid);
      const lid = tracer.startLlmAttempt(tid, { providerId: "claude-code", streamingCapability: "incremental" });
      tracer.markLlmEvent(tid, lid, "first_event");
      tracer.markLlmEvent(tid, lid, "first_text_delta");
      tracer.markLlmEvent(tid, lid, "first_sentence");
      tracer.markLlmEvent(tid, lid, "complete");
      tracer.markLlmEvent(tid, lid, "end");
      tracer.recordValidations(tid, { sanitizeMs: 0.2, filtersMs: 0.3, emotionExtractionMs: 0.1, status: "ok" });
      const job = tracer.startTtsJob(tid, { requestedEngine: "sglang", effectiveEngine: "sglang", playbackMode: "stream" });
      if (job) tracer.updateTtsJob(tid, job.ttsJobId, { requestStart: true, headers: true, firstValidPcm: true, httpBodyEnd: true, bytesReceived: 176400, samplesReceived: 88200, chunkCount: 4, audioDurationSec: 2.0 });
      tracer.markPrepareEnd(tid);
      tracer.markPlaybackTurnStart(tid);
      tracer.markPlaybackDispatched(tid, { url: `stream:${i + 1}`, mode: "stream" });
      tracer.recordPlaybackStartAck(tid, { clientRenderStartMs: 25, contextRate: 44100, contextState: "running" });
      tracer.markPlaybackEnd(tid, { result: "done" });
      tracer.markPlaybackTurnEnd(tid);
      tracer.finishTurn(tid, { state: "completed" });
    }
    const elapsedMs = Number((performance.now() - t0).toFixed(2));
    const memAfter = process.memoryUsage();
    const history = tracer.getHistory();
    const jsonBytes = Buffer.byteLength(JSON.stringify(history));
    const totalEvents = history.reduce((s, t) => s + (t.events?.length || 0), 0);
    return {
      enabled,
      iterations: count,
      elapsedMs,
      perTurnUs: Number(((elapsedMs * 1000) / count).toFixed(2)),
      heapUsedDeltaKb: Math.round((memAfter.heapUsed - memBefore.heapUsed) / 1024),
      rssDeltaKb: Math.round((memAfter.rss - memBefore.rss) / 1024),
      storedHistory: history.length,
      storedEvents: totalEvents,
      jsonExportBytes: jsonBytes,
      summary: tracer.getSummary(),
    };
  };

  const off = runBatch(false, iterations);
  const on = runBatch(true, iterations);
  const longRun = runBatch(true, longRunIterations);
  return {
    off,
    on,
    longRunBounded: {
      iterations: longRunIterations,
      storedHistory: longRun.storedHistory,
      maxHistory: longRun.summary.maxHistory,
      boundedOk: longRun.storedHistory <= longRun.summary.maxHistory,
      jsonExportBytes: longRun.jsonExportBytes,
    },
    note: "Microbenchmark del módulo TurnTracer (creación de IDs, registro de eventos, sanitizado, timeline y serialización JSON; NO mide el impacto completo de NYRA). El historial de 1000 turnos prueba acotamiento del número de entradas (<= maxHistory), no ausencia matemática de cualquier otra retención de memoria.",
  };
}

// ─── CLI principal ────────────────────────────────────────────────────────────
const isMain = process.argv[1]?.replace(/\\/g, "/").endsWith("scripts/benchmark-latency.mjs");
if (isMain) {
  const args = Object.fromEntries(process.argv.slice(2).map(a => {
    const [k, v] = a.replace(/^--/, "").split("=");
    return [k, v === undefined ? true : v];
  }));
  const commit = getCommitShort();

  // H06: exportar trazas reales ya capturadas (JSON de GET /api/diagnostics/traces o array de trazas)
  if (typeof args["export-traces"] === "string") {
    const raw = JSON.parse(readFileSync(args["export-traces"], "utf8"));
    const traces = Array.isArray(raw) ? raw : (Array.isArray(raw.traces) ? raw.traces : []);
    const runId = `capture_${new Date().toISOString().replace(/[:.]/g, "-")}`;
    const outDir = typeof args.out === "string" ? args.out : join(ROOT, "logs", "benchmarks", runId);
    const isSynthetic = Boolean(args.synthetic === true || args.synthetic === "true" || args.synthetic === "1" || raw?.synthetic === true);
    const { doc, files } = exportCapturedTraces(traces, {
      outDir, runId, commit,
      category: typeof args.category === "string" ? args.category : undefined,
      synthetic: isSynthetic,
      dataOrigin: isSynthetic ? "synthetic_test" : "runtime_capture",
      config: { sourceFile: "captured_traces", exportedVia: "benchmark-latency --export-traces", synthetic: isSynthetic },
    });
    console.log(renderMarkdownReport(doc));
    console.log("Exportado en:", outDir);
    for (const [k, v] of Object.entries(files)) console.log(`  - ${k}: ${v}`);
    process.exit(0);
  }

  const level = String(args.level || "all");
  const profiles = String(args.profiles || "A,B,C,D,E").split(",").map(s => s.trim().toUpperCase()).filter(p => PROFILES[p]);
  const runId = `bench_${new Date().toISOString().replace(/[:.]/g, "-")}`;

  const levelsOut = {};
  const aggregates = {};
  const traces = [];
  const comparisons = [];

  if (level === "all" || level === "1") {
    const l1 = await runLevel1({ profiles });
    levelsOut.level1 = { aggregates: l1.aggregates, audioEquivalenceTable: l1.audioEquivalenceTable, historicalReplay: l1.historicalReplay };
    Object.assign(aggregates, l1.aggregates);
    traces.push(...l1.traces);
  }
  if (level === "all" || level === "2") {
    const l2 = await runLevel2({ profiles });
    levelsOut.level2 = { aggregates: l2.aggregates, comparisons: l2.comparisons };
    Object.assign(aggregates, l2.aggregates);
    traces.push(...l2.traces);
    comparisons.push(...l2.comparisons);
  }
  if (level === "all" || level === "3") {
    const l3 = await runLevel3({ profiles });
    levelsOut.level3 = { aggregates: l3.aggregates };
    Object.assign(aggregates, l3.aggregates);
    traces.push(...l3.traces);
  }
  if (args.representative) {
    levelsOut.representative1x = await runRepresentativeScaleScenario();
  }
  if (args.impact) {
    levelsOut.impact = runImpactBenchmark();
  }

  const doc = buildRunDocument({
    runId,
    commit,
    profile: profiles.join(","),
    config: {
      level,
      profiles: profiles.join(","),
      mockOnly: true,
      overlapDefaultInProd: false,
      streamingDefaultInProd: false,
      ...ACCELERATED_PARAMS,
    },
    category: "mock",
    levels: levelsOut,
    aggregates,
    comparisons,
    traces,
    limitations: [
      "Ejecución 100% offline con mocks locales en escala acelerada (NO representa tiempos reales de GPU L4, gateway SGLang ni OBS).",
      "Edge TTS (Perfil A) usa velocidad operativa distinta (17.5 cps mock) y voz distinta; Perfiles B, C y D usan exactamente la misma duración y cantidad de muestras a 44100 Hz para el mismo texto.",
      "La cancelación verificada es HTTP local (AbortController); la cancelación efectiva en GPU remota queda para Fase 4B.",
      "El replay histórico de SGLang L4 se mantiene separado en levels.level1.historicalReplay (categoría historical_replay), distinguiendo N=40 descriptivo, evaluación causal Fase 2 (10 calib / 30 eval) y referencia teórica no causal.",
    ],
  });

  console.log(renderMarkdownReport(doc));
  if (levelsOut.level1?.audioEquivalenceTable?.length) {
    console.log("## Nivel 1 — Equivalencia de audio (duración y muestras por perfil y caso)");
    console.log("| Perfil | Caso | Motor | Modo | Duración (s) | Muestras | Sample Rate | Inicio (ms) | Fin (ms) |");
    console.log("|---|---|---|---|---:|---:|---:|---:|---:|");
    for (const r of levelsOut.level1.audioEquivalenceTable) {
      console.log(`| ${r.profile} | ${r.caseId} | ${r.engine} | ${r.mode} | ${r.audioDurationSec} | ${r.samples} | ${r.sampleRate} | ${r.startMs} | ${r.endMs} |`);
    }
    console.log("");
  }
  if (levelsOut.level1?.historicalReplay) {
    const hr = levelsOut.level1.historicalReplay;
    const c30 = hr.causalPhase2;
    const d40 = hr.descriptiveAll40;
    console.log("## Replay histórico SGLang L4 (categoría separada: historical_replay)");
    console.log(`- Universo: ${hr.metadata.totalSglangStreamingMeasure} trazas SGLang-Omni streaming measure (${hr.metadata.withCharsCount} con chars, ${hr.metadata.withoutCharsCount} sin chars excluidas; perfiles: ${hr.metadata.profilesIncluded.join(", ")}; ref: ${hr.metadata.voiceReference})`);
    console.log(`- Causal Fase 2 (10 calib / 30 eval): WAV p50/p95=${c30.aggregates.SGLang_WAV_causal30.metrics.estimatedPlaybackStartRelMs.median}/${c30.aggregates.SGLang_WAV_causal30.metrics.estimatedPlaybackStartRelMs.p95} ms | PCM p50/p95=${c30.aggregates.SGLang_PCM_adaptive_causal30.metrics.estimatedPlaybackStartRelMs.median}/${c30.aggregates.SGLang_PCM_adaptive_causal30.metrics.estimatedPlaybackStartRelMs.p95} ms (underruns=${c30.aggregates.SGLang_PCM_adaptive_causal30.counts.underrunCount}) | diffMed=${c30.pairedComparison.diffOfMedians} ms | medDiffPareada=${c30.pairedComparison.paired?.medianOfPairedDiffs} ms`);
    console.log(`- Descriptivo N=40 (sin holdout): WAV p50=${d40.aggregates.SGLang_WAV_N40.metrics.estimatedPlaybackStartRelMs.median} ms | PCM p50=${d40.aggregates.SGLang_PCM_adaptive_N40.metrics.estimatedPlaybackStartRelMs.median} ms (underruns=${d40.aggregates.SGLang_PCM_adaptive_N40.counts.underrunCount}) | medDiffPareada=${d40.pairedComparison.paired?.medianOfPairedDiffs} ms`);
    console.log(`- Referencia retrospectiva teórica (oracle, 30 eval): p50=${hr.theoreticalOracle.aggregate.metrics.estimatedPlaybackStartRelMs.median} ms / p95=${hr.theoreticalOracle.aggregate.metrics.estimatedPlaybackStartRelMs.p95} ms\n`);
  }
  if (levelsOut.representative1x) {
    const rep = levelsOut.representative1x;
    console.log("## Escenario representativo 1:1 (mocks con umbrales EXACTOS de producción: delayMs=250, minHeadStartMs=1000)");
    for (const pid of ["C", "D", "E"]) {
      const t = rep.traces[pid];
      console.log(`- Perfil ${pid}: inicio=${t.metrics.estimatedPlaybackStartRelMs} ms | fin=${t.metrics.playbackEndRelMs} ms | LLM=${t.metrics.llmCompleteMs} ms | pedidosTTS=${t.metrics.ttsRequestsPerTurn} | overlapUsed=${t.metrics.speculationUsed} | headStart=${t.overlap?.headStartMs ?? "-"} ms`);
    }
    console.log("\n" + formatOverlappingTimeline(rep.traces.E) + "\n");
  }
  if (levelsOut.impact) {
    const imp = levelsOut.impact;
    console.log("## Microbenchmark de TurnTracer (detallada OFF vs ON; no es overhead total de NYRA)");
    console.log(`- OFF (${imp.off.iterations} turnos): ${imp.off.elapsedMs} ms (${imp.off.perTurnUs} µs/turno), heapΔ=${imp.off.heapUsedDeltaKb} KB`);
    console.log(`- ON  (${imp.on.iterations} turnos): ${imp.on.elapsedMs} ms (${imp.on.perTurnUs} µs/turno), heapΔ=${imp.on.heapUsedDeltaKb} KB, historial=${imp.on.storedHistory}, eventos=${imp.on.storedEvents}, JSON=${imp.on.jsonExportBytes} B`);
    console.log(`- Largo plazo (${imp.longRunBounded.iterations} turnos): historial=${imp.longRunBounded.storedHistory}/${imp.longRunBounded.maxHistory} (acotado=${imp.longRunBounded.boundedOk})\n`);
  }

  if (args.export || args.out) {
    const outDir = typeof args.out === "string" ? args.out : (typeof args.export === "string" ? args.export : join(ROOT, "logs", "benchmarks", runId));
    const files = exportBenchmarkRun(outDir, doc);
    console.log("Exportado en:", outDir);
    for (const [k, v] of Object.entries(files)) console.log(`  - ${k}: ${v}`);
  }
}
