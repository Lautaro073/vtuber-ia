// Fase 4A: tests obligatorios (1–40) para instrumentación integral, trazas, relojes, exportación,
// privacidad, límites de memoria, aislamiento de fallos y banco de pruebas offline reproducible.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { TurnTracer, TRACE_SCHEMA_VERSION, sanitizeValue } from "../lib/observability/turn-tracer.js";
import {
  PERCENTILE_METHOD,
  percentile,
  summarizeSeries,
  aggregateTurns,
  compareProfiles,
  formatOverlappingTimeline,
  toCsv,
  buildRunDocument,
  exportBenchmarkRun,
  exportCapturedTraces,
  renderHtmlReport,
  checkDiagnosticsAccess,
  resolveAuthorizedExportDir,
} from "../lib/observability/benchmark-report.js";
import { SGLangTTSClient } from "../lib/tts/sglang-tts-client.js";
import { SglangScheduler } from "../lib/tts/sglang-scheduler.js";
import { StreamPlayer } from "../public/tts-stream/stream-player.js";
import {
  CORPUS,
  PROFILES,
  ACCELERATED_PARAMS,
  syntheticAudioForText,
  startMockTtsServer,
  createIsolatedHarness,
  runLevel1,
  runLevel2,
  runLevel3,
  runImpactBenchmark,
} from "../scripts/benchmark-latency.mjs";

const sleep = ms => new Promise(r => setTimeout(r, ms));

function makeSimClock(start = 1000) {
  let t = start;
  return {
    now: () => t,
    advance: ms => { t += ms; return t; },
    set: ms => { t = ms; return t; },
  };
}

// ─── 1, 2, 3, 4, 5: Trazas, IDs únicos y correlación completa ─────────────────
test("1/2/3/4/5: creación de trazas, IDs únicos no textuales, correlación mensaje→turno→múltiples TTS y segmentos", () => {
  const clk = makeSimClock(100);
  const tr = new TurnTracer({ enabled: true, category: "simulation", now: clk.now });
  const t1 = tr.startTurn({ source: "live", platform: "tiktok", priority: 4, cancelEpoch: 2, requestedProvider: "claude-code", requestedEngine: "sglang", playbackMode: "stream" });
  const t2 = tr.startTurn({ source: "live", platform: "tiktok", priority: 4, cancelEpoch: 2, requestedProvider: "claude-code", requestedEngine: "sglang", playbackMode: "stream" });
  assert.ok(t1 && t2 && t1 !== t2);

  const active1 = tr.getActive(t1);
  const active2 = tr.getActive(t2);
  assert.notEqual(active1.ids.messageId, active2.ids.messageId);
  assert.match(active1.ids.messageId, /^msg_/);
  assert.match(active1.ids.turnId, /^trn_/);
  assert.equal(active1.cancelEpoch, 2);

  // Dos segmentos TTS (especulativo + resto) correlacionados bajo UN SOLO turno
  const seg1 = tr.startTtsJob(t1, { requestedEngine: "sglang", effectiveEngine: "sglang", playbackMode: "stream", segmentIndex: 0, speculative: true, streamId: 7 });
  clk.advance(50);
  tr.updateTtsJob(t1, seg1.ttsJobId, { requestStart: true, headers: true, firstValidPcm: true, httpBodyEnd: true, audioDurationSec: 1.5, bytesReceived: 132300, samplesReceived: 66150, chunkCount: 2 });
  const seg2 = tr.startTtsJob(t1, { requestedEngine: "sglang", effectiveEngine: "sglang", playbackMode: "stream", segmentIndex: 1, speculative: false, streamId: 7 });
  clk.advance(70);
  tr.updateTtsJob(t1, seg2.ttsJobId, { requestStart: true, headers: true, firstValidPcm: true, httpBodyEnd: true, audioDurationSec: 2.0, bytesReceived: 176400, samplesReceived: 88200, chunkCount: 3 });

  const done = tr.finishTurn(t1, { state: "completed" });
  tr.finishTurn(t2, { state: "completed" });

  assert.equal(done.schemaVersion, TRACE_SCHEMA_VERSION);
  assert.equal(done.ids.ttsJobIds.length, 2);
  assert.equal(done.ids.httpRequestIds.length, 2);
  assert.equal(done.ids.segmentIds.length, 2);
  assert.notEqual(seg1.segmentId, seg2.segmentId);
  assert.deepEqual(done.ids.streamIds, [7]);
  assert.equal(done.tts.totalRequests, 2);
  assert.equal(done.metrics.ttsRequestsPerTurn, 2);
  assert.equal(done.metrics.audioDurationSec, 3.5);
  // 2 segmentos siguen contando como 1 turno conversacional
  assert.equal(tr.getHistory().length, 2);
});

// ─── 6, 7, 8: Cola, LLM (incremental vs full-only sin inventar TTFT) y validaciones ──
test("6/7/8: medición de cola, LLM incremental vs full-only (sin fabricar TTFT) y validaciones", () => {
  const clk = makeSimClock(500);
  const tr = new TurnTracer({ enabled: true, category: "simulation", now: clk.now });

  // Proveedor incremental (claude-code)
  const tidInc = tr.startTurn({ source: "live", platform: "tiktok", priority: 4, requestedProvider: "claude-code" });
  clk.advance(5);
  tr.markEnqueued(tidInc, { priority: 4, holdMs: 2500 });
  clk.advance(45);
  tr.markPrepareStart(tidInc, { overlappedWithPlayback: false });
  const llmInc = tr.startLlmAttempt(tidInc, { providerId: "claude-code", streamingCapability: "incremental" });
  clk.advance(20);
  tr.markLlmEvent(tidInc, llmInc, "first_event");
  clk.advance(10);
  tr.markLlmEvent(tidInc, llmInc, "first_text_delta");
  clk.advance(30);
  tr.markLlmEvent(tidInc, llmInc, "first_sentence");
  clk.advance(40);
  tr.markLlmEvent(tidInc, llmInc, "complete");
  tr.markLlmEvent(tidInc, llmInc, "end");
  tr.recordValidations(tidInc, { sanitizeMs: 0.4, filtersMs: 0.5, responseValidationMs: 0.6, emotionExtractionMs: 0.2, phoneticsMs: 0.3, ttsTransformMs: 0.1, status: "ok" });
  const doneInc = tr.finishTurn(tidInc, { state: "completed" });

  assert.equal(doneInc.queue.conversationalQueueWaitMs, 45);
  assert.equal(doneInc.llm.firstEventLatencyMs, 20);
  assert.equal(doneInc.llm.firstTextLatencyMs, 30);
  assert.equal(doneInc.llm.firstSentenceLatencyMs, 60);
  assert.equal(doneInc.llm.totalMs, 100);
  assert.equal(doneInc.validations.totalMs, 2.1);
  assert.equal(doneInc.validations.status, "ok");
  assert.match(doneInc.clocks.receptionClockNote, /TikTok|Discord/i);

  // Proveedor de respuesta completa (codex-cli): NO inventa TTFT incremental
  const tidFull = tr.startTurn({ source: "live", platform: "tiktok", priority: 4, requestedProvider: "codex-cli" });
  const llmFull = tr.startLlmAttempt(tidFull, { providerId: "codex-cli", streamingCapability: "full_only" });
  clk.advance(150);
  tr.markLlmEvent(tidFull, llmFull, "first_event");
  tr.markLlmEvent(tidFull, llmFull, "first_text_delta"); // debe ignorarse para full_only
  tr.markLlmEvent(tidFull, llmFull, "complete");
  tr.markLlmEvent(tidFull, llmFull, "end");
  const doneFull = tr.finishTurn(tidFull, { state: "completed" });

  assert.equal(doneFull.llm.firstEventLatencyMs, 150);
  assert.equal(doneFull.llm.firstTextLatencyMs, null, "proveedores full_only no deben inventar TTFT incremental");
  assert.equal(doneFull.llm.totalMs, 150);
});

// ─── 9, 10, 11, 12, 13, 14: WAV vs PCM, inicio efectivo, fin HTTP vs fin audible, relojes ──
test("9/10/11/12/13/14: WAV vs PCM progresivo, inicio efectivo unificado, fin HTTP vs audible y separación de relojes", () => {
  const clk = makeSimClock(1000);
  const tr = new TurnTracer({ enabled: true, category: "simulation", now: clk.now });

  // WAV: registra firstByteRelMs (sin llamarlo firstValidPcmRelMs) y remoteGenEndRelMs=null
  const tidWav = tr.startTurn({ source: "live", platform: "tiktok", requestedEngine: "sglang", playbackMode: "wav" });
  const jWav = tr.startTtsJob(tidWav, { requestedEngine: "sglang", effectiveEngine: "sglang", playbackMode: "wav" });
  clk.advance(10); tr.updateTtsJob(tidWav, jWav.ttsJobId, { requestStart: true });
  clk.advance(40); tr.updateTtsJob(tidWav, jWav.ttsJobId, { headers: true, firstByte: true });
  clk.advance(20); tr.updateTtsJob(tidWav, jWav.ttsJobId, { httpBodyEnd: true, bytesReceived: 88244, audioDurationSec: 1.0 });
  tr.markPlaybackDispatched(tidWav, { url: "/tts/1.wav", mode: "wav" });
  clk.advance(15);
  // Reloj del navegador tiene su propio origen (ej. 520000ms desde apertura de pestaña): nunca se resta al reloj de Node
  tr.recordPlaybackStartAck(tidWav, { viewId: "obs_1", mode: "wav", clientRenderStartMs: 12, contextRate: 44100, contextState: "running", outputLatencyMs: 18, estimatedRttMs: 3 });
  clk.advance(1000);
  tr.markPlaybackEnd(tidWav, { result: "done" });
  const doneWav = tr.finishTurn(tidWav, { state: "completed" });

  assert.equal(doneWav.metrics.firstByteRelMs, 50);
  assert.equal(doneWav.metrics.firstPcmRelMs, null, "WAV no debe rellenar firstValidPcmRelMs");
  assert.equal(doneWav.tts.jobs[0].remoteGenEndRelMs, null);
  assert.equal(doneWav.metrics.estimatedPlaybackStartRelMs, 85);
  assert.equal(doneWav.playback.serverToPlaybackStartAckMs, 15);
  assert.equal(doneWav.playback.views[0].clientClock.effectiveRenderStartMs, 12);
  assert.equal(doneWav.metrics.playbackEndRelMs, 1085);
  assert.equal(doneWav.metrics.audibleEndDiffFromHttpEndMs, 1015, "fin audible ocurre después del fin HTTP");

  // PCM progresivo: registra firstValidPcmRelMs y arranca reproducción ANTES de httpBodyEnd
  const tidPcm = tr.startTurn({ source: "live", platform: "tiktok", requestedEngine: "sglang", playbackMode: "stream" });
  const jPcm = tr.startTtsJob(tidPcm, { requestedEngine: "sglang", effectiveEngine: "sglang", playbackMode: "stream", streamId: 99 });
  clk.advance(5); tr.updateTtsJob(tidPcm, jPcm.ttsJobId, { requestStart: true });
  clk.advance(30); tr.updateTtsJob(tidPcm, jPcm.ttsJobId, { headers: true });
  clk.advance(5); tr.updateTtsJob(tidPcm, jPcm.ttsJobId, { firstValidPcm: true });
  tr.markPlaybackDispatched(tidPcm, { url: "stream:99", mode: "stream" });
  clk.advance(10); tr.recordPlaybackStartAck(tidPcm, { viewId: "obs_1", mode: "stream", clientRenderStartMs: 8, bufferAtStartSec: 1.85, contextRate: 44100, contextState: "running" });
  clk.advance(200); tr.updateTtsJob(tidPcm, jPcm.ttsJobId, { httpBodyEnd: true, bytesReceived: 352800, samplesReceived: 176400, chunkCount: 4, audioDurationSec: 4.0 });
  clk.advance(3800); tr.markPlaybackEnd(tidPcm, { result: "done" });
  const donePcm = tr.finishTurn(tidPcm, { state: "completed" });

  assert.equal(donePcm.metrics.firstPcmRelMs, 40);
  assert.ok(donePcm.metrics.estimatedPlaybackStartRelMs < donePcm.metrics.httpBodyEndRelMs, "PCM progresivo arranca antes del fin HTTP");
  assert.ok(donePcm.metrics.audibleEndDiffFromHttpEndMs > 0);
});

// ─── 15, 16, 17: Cancelaciones durante LLM, TTS y reproducción ──────────────────
test("15/16/17: cancelación durante LLM, durante TTS (HTTP local sin fallback) y durante reproducción", async () => {
  const srv = await startMockTtsServer({ sglangWavTotalMs: 250, sglangFirstChunkMs: 120 });
  const tracer = new TurnTracer({ enabled: true, category: "mock" });
  const h = createIsolatedHarness({ profileId: "C", serverUrl: srv.url, tracer });
  try {
    // 15. Cancelación durante LLM
    const tLlm = tracer.startTurn({ source: "live", platform: "tiktok", requestedProvider: "claude-code" });
    const lid = tracer.startLlmAttempt(tLlm, { providerId: "claude-code", streamingCapability: "incremental" });
    tracer.markLlmEvent(tLlm, lid, "cancelled");
    const cLlm = tracer.finishTurn(tLlm, { state: "cancelled", reason: "llm_cancelled" });
    assert.equal(cLlm.state, "cancelled");
    assert.equal(cLlm.llm.cancelled, true);

    // 16. Cancelación durante TTS (sin fallback)
    const tTts = tracer.startTurn({ source: "live", platform: "tiktok", requestedEngine: "sglang", playbackMode: "wav" });
    const synthP = h.tts.synth("Texto largo para cancelar en vuelo durante la descarga.", { source: "live", returnMeta: true, turnId: tTts });
    await sleep(30);
    h.tts.stop("live");
    const resTts = await synthP;
    const cTts = tracer.finishTurn(tTts, { state: "cancelled", reason: "tts_stop" });
    assert.equal(resTts.cancelled, true);
    assert.equal(resTts.fallbackUsed, false);
    assert.equal(cTts.tts.cancelled, true);
    assert.equal(cTts.tts.fallbackUsed, false);

    // 17. Cancelación durante reproducción
    const tPlay = tracer.startTurn({ source: "live", platform: "tiktok", requestedEngine: "sglang", playbackMode: "wav" });
    const resOk = await h.tts.synth("Listo para reproducir y cortar.", { source: "live", returnMeta: true, turnId: tPlay });
    assert.ok(resOk.url);
    await h.tts.play(resOk.url, "Listo para reproducir y cortar.", false, { turnId: tPlay });
    h.tts.stop("live");
    const cPlay = tracer.finishTurn(tPlay, { state: "cancelled", reason: "playback_stop" });
    assert.equal(cPlay.playback.cancelled, true);
    assert.equal(cPlay.state, "cancelled");
  } finally {
    h.cleanup();
    await srv.close();
  }
});

// ─── 18, 19, 20: Fallback, fallo parcial y reintentos HTTP 429 ────────────────
test("18/19/20: fallback a Edge, fallo parcial del 2.º segmento sin duplicar y reintentos HTTP 429 con backoff", async () => {
  // 20. Reintento HTTP 429
  const srv429 = await startMockTtsServer({ fail429Once: true, sglangWavTotalMs: 25 });
  const tr429 = new TurnTracer({ enabled: true, category: "mock" });
  const h429 = createIsolatedHarness({ profileId: "C", serverUrl: srv429.url, tracer: tr429 });
  try {
    const tid = tr429.startTurn({ source: "live", platform: "tiktok", requestedEngine: "sglang", playbackMode: "wav" });
    const res = await h429.tts.synth("Prueba de reintento ante ocupado.", { source: "live", returnMeta: true, turnId: tid });
    const done = tr429.finishTurn(tid, { state: "completed" });
    assert.ok(res.url && !res.fallbackUsed);
    assert.equal(done.metrics.retries429, 1);
  } finally {
    h429.cleanup();
    await srv429.close();
  }

  // 18. Fallback a Edge ante error antes de sonar
  const srvTimeout = await startMockTtsServer({ timeoutNext: true });
  const trFb = new TurnTracer({ enabled: true, category: "mock" });
  const hFb = createIsolatedHarness({ profileId: "C", serverUrl: srvTimeout.url, tracer: trFb, sglangTimeoutMs: 80 });
  hFb.tts.sglang.config.timeoutMs = 80; // fuerza timeout rápido en mock
  try {
    const tid = trFb.startTurn({ source: "live", platform: "tiktok", requestedEngine: "sglang", playbackMode: "wav" });
    const res = await hFb.tts.synth("Debe caer a Edge una sola vez.", { source: "live", returnMeta: true, turnId: tid });
    const done = trFb.finishTurn(tid, { state: "completed" });
    assert.equal(res.fallbackUsed, true);
    assert.ok(res.url.endsWith(".mp3"));
    assert.equal(done.metrics.fallbackUsed, true);
    assert.equal(done.tts.jobs.length, 2, "solicitud SGLang fallida + fallback Edge correlacionados al mismo turno");
    assert.equal(done.tts.jobs[1].effectiveEngine, "edge");
  } finally {
    hFb.cleanup();
    await srvTimeout.close();
  }

  // 19. Fallo parcial del 2.º segmento tras emitir PCM → sin Edge duplicado
  const srvPartial = await startMockTtsServer({ failSecondSegment: true, sglangFirstChunkMs: 15, sglangChunkGapMs: 15 });
  const trPart = new TurnTracer({ enabled: true, category: "mock" });
  const hPart = createIsolatedHarness({ profileId: "E", serverUrl: srvPartial.url, tracer: trPart, llmDelays: { firstDeltaMs: 10, deltaGapMs: 25 } });
  let edgeFallbacks = 0;
  const origEdge = hPart.tts._synthEdge.bind(hPart.tts);
  hPart.tts._synthEdge = async (...a) => { edgeFallbacks += 1; return origEdge(...a); };
  try {
    hPart.replyQueue.push("Esta primera oración tiene largo suficiente para disparar especulación real. Y esta segunda oración se corta a mitad del stream.");
    const tid = trPart.startTurn({ source: "live", platform: "tiktok", requestedProvider: "claude-code", requestedEngine: "sglang", playbackMode: "stream" });
    const prep = await hPart.chat._prepare({ username: "ana", text: "contame algo", platform: "tiktok", opts: {}, rank: 4, at: Date.now(), turnId: tid }, hPart.ai.llm);
    await hPart.chat._respond({ username: "ana", text: "contame algo", platform: "tiktok", opts: {}, rank: 4, ready: prep, turnId: tid });
    const hist = trPart.getHistory();
    assert.equal(edgeFallbacks, 0, "fallo parcial tras emitir PCM no debe disparar Edge");
    assert.equal(hist.at(-1).metrics.partialFailure, true);
    assert.equal(hist.at(-1).state, "partial_failure");
  } finally {
    hPart.cleanup();
    await srvPartial.close();
  }
});

// ─── 21, 22, 23: Overlap aprovechado, overlap descartado y donación prioritaria ──
test("21/22/23: overlap aprovechado (2 segmentos en 1 turno), overlap descartado y donación prioritaria desplazando especulación", async () => {
  const srv = await startMockTtsServer({ sglangFirstChunkMs: 25, sglangChunkGapMs: 15 });
  const tr = new TurnTracer({ enabled: true, category: "mock" });
  const h = createIsolatedHarness({ profileId: "E", serverUrl: srv.url, tracer: tr, llmDelays: { firstDeltaMs: 10, deltaGapMs: 25 } });
  try {
    // 21. Overlap aprovechado
    h.replyQueue.push("Che, qué buena pregunta la tuya sobre el directo de hoy. Te cuento todos los detalles paso a paso sin apuro.");
    const tidUsed = tr.startTurn({ source: "live", platform: "tiktok", requestedProvider: "claude-code", requestedEngine: "sglang", playbackMode: "stream" });
    const prepUsed = await h.chat._prepare({ username: "leo", text: "pregunta", platform: "tiktok", opts: {}, rank: 4, at: Date.now(), turnId: tidUsed }, h.ai.llm);
    await h.chat._respond({ username: "leo", text: "pregunta", platform: "tiktok", opts: {}, rank: 4, ready: prepUsed, turnId: tidUsed });
    const recUsed = tr.getHistory().find(x => x.ids.turnId === tidUsed);
    assert.equal(recUsed.overlap.speculationStarted, true);
    assert.equal(recUsed.overlap.speculationUsed, true);
    assert.equal(recUsed.overlap.measuredGpuMs, null, "no declara GPU medida sin telemetría del backend");
    assert.equal(recUsed.tts.jobs.length, 2, "dos segmentos bajo el mismo turnId");

    // 22. Overlap descartado (cambio de emoción / texto no coincide)
    const tidDisc = tr.startTurn({ source: "live", platform: "tiktok", requestedProvider: "claude-code", requestedEngine: "sglang", playbackMode: "stream" });
    const spec = h.overlap.begin({
      providerId: "claude-code", source: "live", rank: 4, turnId: tidDisc,
      build: s => ({ ttsText: s, emotion: "happy" }),
    });
    h.overlap.feed(spec, "Esta primera oración es candidata para especular rápido. Arranca otra.");
    await sleep(30);
    const fin = await h.overlap.finish(spec, { filtered: "Texto completamente distinto que no coincide.", emotion: "happy" });
    assert.equal(fin, null);
    await spec.handle?.seg1?.catch(() => {});
    const recDisc = tr.finishTurn(tidDisc, { state: "completed" });
    assert.equal(recDisc.overlap.speculationDiscarded, true);
    assert.ok(recDisc.overlap.discardReason);
    assert.equal(recDisc.overlap.measuredGpuMs, null);

    // 23. Donación prioritaria (rank 0) desplaza especulación de chat (rank 4)
    const tidLow = tr.startTurn({ source: "live", platform: "tiktok", priority: 4, requestedEngine: "sglang", playbackMode: "stream" });
    const specLow = h.tts.speculateStart({ ttsText: "Oración de chat común que va a ser desplazada.", emotion: "neutral", rank: 4, turnId: tidLow });
    assert.ok(specLow);
    const tidGift = tr.startTurn({ source: "live", platform: "tiktok", priority: 0, requestedEngine: "sglang", playbackMode: "stream" });
    const giftRes = await h.tts.synth("¡Gracias por la donación gigante!", { source: "live", rank: 0, returnMeta: true, turnId: tidGift });
    assert.ok(giftRes.url);
    const recLow = tr.finishTurn(tidLow, { state: "discarded", reason: "displaced" });
    tr.finishTurn(tidGift, { state: "completed" });
    assert.equal(recLow.overlap.displacedByPriority, true);
  } finally {
    h.cleanup();
    await srv.close();
  }
});

// ─── 24, 25, 26: Prepare-ahead, Discord independiente, consumidores múltiples ─
test("24/25/26: prepare-ahead sin sumar solapes, Discord independiente intacto y múltiples consumidores", async () => {
  const srv = await startMockTtsServer({ sglangFirstChunkMs: 20, sglangChunkGapMs: 10 });
  const tr = new TurnTracer({ enabled: true, category: "mock" });
  const h = createIsolatedHarness({
    profileId: "D", serverUrl: srv.url, tracer: tr,
    discordOn: true, shareTtsEngine: false, discordTtsEngine: "edge",
    hubOpts: { extraViews: 2 },
  });
  try {
    // 25. Discord independiente no toca SGLang
    const reqBefore = srv.getRequests();
    const tidDisc = tr.startTurn({ source: "discord", platform: "discord", priority: 1, requestedEngine: "edge", playbackMode: "wav" });
    const discRes = await h.tts.synth("Respuesta exclusiva de Discord.", { source: "discord", returnMeta: true, turnId: tidDisc });
    const recDisc = tr.finishTurn(tidDisc, { state: "completed" });
    assert.equal(srv.getRequests(), reqBefore, "Discord independiente (edge) no hace peticiones a SGLang");
    assert.ok(discRes.url.endsWith(".mp3"));
    assert.equal(recDisc.effectiveEngine, "edge");

    // 24 + 26. Prepare-ahead + múltiples consumidores (audible + muted)
    h.replyQueue.push("Primer turno que ocupa reproducción en las vistas conectadas.");
    h.replyQueue.push("Segundo turno preparado en paralelo mientras suena el primero.");
    h.chat._enqueue("u1", "hola uno", "tiktok", {}, 4, 0);
    h.chat._enqueue("u2", "hola dos", "tiktok", {}, 4, 0);
    const deadline = Date.now() + 8000;
    while ((h.chat.queue.length > 0 || h.chat._isResponding) && Date.now() < deadline) await sleep(15);
    await sleep(30);
    const liveTurns = tr.getHistory().filter(t => t.source === "live");
    assert.equal(liveTurns.length, 2);
    assert.ok(liveTurns[0].playback.views.length >= 1);
    const tl = formatOverlappingTimeline(liveTurns[0]);
    assert.match(tl, /etapas superpuestas/i);
  } finally {
    h.cleanup();
    await srv.close();
  }
});

// ─── 27, 28: Trazas abandonadas y límites de memoria ──────────────────────────
test("27/28: expiración de trazas abandonadas y límites estrictos de memoria (maxActive, maxHistory, maxEventsPerTurn)", () => {
  const clk = makeSimClock(0);
  const tr = new TurnTracer({
    enabled: true, category: "simulation", now: clk.now,
    maxHistory: 10, maxActive: 5, maxEventsPerTurn: 12, abandonTimeoutMs: 500,
  });
  const orphan = tr.startTurn({ source: "live", platform: "tiktok" });
  for (let i = 0; i < 30; i++) tr.markEnqueued(orphan, { priority: 4 });
  assert.equal(tr.getActive(orphan).events.length, 12, "eventos por turno acotados");
  assert.ok(tr.droppedEvents > 0);

  clk.advance(600);
  const swept = tr.sweepAbandoned();
  assert.equal(swept, 1);
  assert.equal(tr.getHistory()[0].state, "abandoned");

  // Saturar maxActive y maxHistory
  for (let i = 0; i < 40; i++) {
    const tid = tr.startTurn({ source: "live", platform: "tiktok" });
    if (i % 2 === 0) tr.finishTurn(tid, { state: "completed" });
  }
  assert.ok(tr.active.size <= 5, "activas acotadas por maxActive");
  assert.ok(tr.history.length <= 10, "historial acotado por maxHistory");
  assert.ok(tr.evictedActive > 0);
});

// ─── 29, 30, 31, 32, 33: Exportación JSON/CSV, percentiles, warmups y categorías ──
test("29/30/31/32/33: exportación JSON/JSONL/CSV/MD/HTML, percentiles R7, exclusión de warmups y separación de categorías", () => {
  // 31. Percentiles lineales R7 e inestabilidad p95 con N < 20
  assert.equal(PERCENTILE_METHOD, "linear_interpolation_R7");
  assert.equal(percentile([10, 20, 30, 40, 50], 0.5), 30);
  assert.equal(percentile([10, 20, 30, 40], 0.5), 25);
  const small = summarizeSeries([10, 20, 30, 40, 50]);
  assert.equal(small.p95Unstable, true);
  const big = summarizeSeries(Array.from({ length: 25 }, (_, i) => (i + 1) * 10));
  assert.equal(big.p95Unstable, false);

  // 32. Exclusión de warmups + 31 diferencia de medianas vs mediana de diferencias pareadas
  const mk = (id, cat, startMs, warmup = false) => ({
    ids: { turnId: id, messageId: `m_${id}` },
    corpusId: id,
    category: cat,
    warmup,
    state: "completed",
    source: "live",
    platform: "tiktok",
    priority: 4,
    metrics: { estimatedPlaybackStartRelMs: startMs, playbackEndRelMs: startMs + 500, turnTotalMs: startMs + 500, ttsRequestsPerTurn: 1, underruns: 0, underrunTotalMs: 0 },
    timeline: [{ stage: "tts", startRelMs: 0, endRelMs: startMs }, { stage: "playback", startRelMs: startMs, endRelMs: startMs + 500 }],
  });
  const base = [mk("w0", "mock", 9999, true), mk("a", "mock", 100), mk("b", "mock", 200), mk("c", "mock", 400)];
  const cand = [mk("w0", "mock", 8888, true), mk("a", "mock", 90), mk("b", "mock", 120), mk("c", "mock", 250)];
  const agg = aggregateTurns(base);
  assert.equal(agg.counts.warmupsExcluded, 1);
  assert.equal(agg.counts.measured, 3);
  assert.equal(agg.metrics.estimatedPlaybackStartRelMs.median, 200);
  assert.equal(agg.warmupSummary.n, 1);

  const cmp = compareProfiles(base, cand, t => t.metrics.estimatedPlaybackStartRelMs, { pairKey: t => t.corpusId });
  assert.equal(cmp.diffOfMedians, 80); // 200 - 120
  assert.equal(cmp.paired.medianOfPairedDiffs, 80); // median([10, 80, 150]) = 80

  // 33. Mezclar mock y real_gateway debe lanzar explícitamente
  assert.throws(() => aggregateTurns([mk("a", "mock", 100), mk("b", "real_gateway", 100)]), /mezclar categorías/i);
  assert.throws(() => compareProfiles([mk("a", "mock", 100)], [mk("a", "real_gateway", 90)]), /categorías distintas/i);

  // 29 + 30. Exportación JSON, JSONL, CSV, Markdown y HTML
  const tmp = mkdtempSync(join(tmpdir(), "nyra-export-"));
  try {
    const doc = buildRunDocument({
      runId: "test_run_1", commit: "86dbf3f", category: "mock",
      aggregates: { base: agg }, traces: base, limitations: ["solo mock"],
    });
    const out = exportBenchmarkRun(tmp, doc);
    const parsed = JSON.parse(readFileSync(out.jsonPath, "utf8"));
    assert.equal(parsed.schemaVersion, TRACE_SCHEMA_VERSION);
    assert.equal(parsed.category, "mock");
    const csvText = readFileSync(out.csvPath, "utf8");
    assert.match(csvText, /^turnId,messageId,profile,category/);
    assert.equal(csvText.trim().split(/\r?\n/).length, 5); // header + 4 rows
    assert.match(readFileSync(out.mdPath, "utf8"), /linear_interpolation_R7/);
    assert.match(readFileSync(out.htmlPath, "utf8"), /<!doctype html>/i);
    assert.equal(toCsv([{ a: 'con,coma', b: 'con"comilla' }]), 'a,b\n"con,coma","con""comilla"\n');
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

// ─── 34, 35: Ausencia de secretos y contenido privado ─────────────────────────
test("34/35: ausencia de secretos, prompts, memorias, texto privado y datos personales en trazas y exportaciones", () => {
  const dirty = sanitizeValue({
    apiKey: "sk-live-secret-99999999",
    authorization: "Bearer sk-token-12345678",
    prompt: "Prompt secreto del sistema",
    memory: "Memoria privada del viewer",
    text: "Mensaje privado del chat",
    username: "persona_real",
    referencePath: "C:\\voices\\privada.wav",
    safeCode: "ok",
    nestedNote: "Bearer sk-leak-99999999",
  });
  assert.equal(dirty.apiKey, undefined);
  assert.equal(dirty.authorization, undefined);
  assert.equal(dirty.prompt, undefined);
  assert.equal(dirty.memory, undefined);
  assert.equal(dirty.text, undefined);
  assert.equal(dirty.username, undefined);
  assert.equal(dirty.referencePath, undefined);
  assert.equal(dirty.safeCode, "ok");
  assert.equal(dirty.nestedNote, "[REDACTED]");

  const tr = new TurnTracer({ enabled: true, category: "mock" });
  const tid = tr.startTurn({ source: "live", platform: "tiktok" });
  tr.recordValidations(tid, { status: "ok", reasonCode: "Bearer sk-topsecret-12345" });
  const done = tr.finishTurn(tid, { state: "completed" });
  const serialized = JSON.stringify(done);
  assert.doesNotMatch(serialized, /sk-topsecret|sk-live-secret|Prompt secreto|Mensaje privado/);
});

// ─── 36, 37, 38, 39: Compatibilidad, default OFF, tolerancia a fallos y preservación ──
test("36/37/38/39: compatibilidad de configuraciones viejas, telemetría OFF por defecto, fallos aislados y preservación de Fases 1–3", async () => {
  // 36 + 39. Configuración antigua sin playbackMode ni overlap → wav y overlap=false
  const tmp = mkdtempSync(join(tmpdir(), "nyra-compat-"));
  try {
    const sf = join(tmp, "tts_settings.json");
    writeFileSync(sf, JSON.stringify({ engine: "edge", sglang: { url: "https://example.com" } }));
    const client = new SGLangTTSClient({ settingsFile: sf, allowedHosts: ["example.com"] });
    assert.equal(client.config.playbackMode, "wav");
    assert.equal(client.config.overlap, false);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }

  // 37. Telemetría detallada desactivada por defecto
  const defTracer = new TurnTracer();
  assert.equal(defTracer.isEnabled(), false);
  assert.equal(defTracer.startTurn({ source: "live" }), null);
  assert.equal(defTracer.getHistory().length, 0);

  // 38. Fallos internos de telemetría nunca afectan a NYRA
  let caughtErrors = 0;
  const brokenTracer = new TurnTracer({
    enabled: true,
    now: () => { throw new Error("clock exploded"); },
    onError: () => { caughtErrors += 1; },
  });
  assert.equal(brokenTracer.startTurn({ source: "live" }), null);
  assert.ok(caughtErrors >= 1);
  assert.ok(brokenTracer.internalErrors >= 1);

  const srv = await startMockTtsServer({ sglangWavTotalMs: 20 });
  const h = createIsolatedHarness({ profileId: "C", serverUrl: srv.url, tracer: brokenTracer });
  try {
    const res = await h.tts.synth("NYRA sigue funcionando aunque la telemetría falle.", { source: "live", returnMeta: true });
    assert.ok(res.url, "TTSEngine no se rompe aunque el tracer lance errores");
  } finally {
    h.cleanup();
    await srv.close();
  }

  // 39. SglangScheduler mantiene serialización estricta
  const sched = new SglangScheduler();
  let inFlight = 0, maxInFlight = 0;
  await Promise.all([1, 2, 3].map(() => sched.enqueue(async () => {
    inFlight += 1; maxInFlight = Math.max(maxInFlight, inFlight);
    await sleep(10);
    inFlight -= 1;
  })));
  assert.equal(maxInFlight, 1);
});

// ─── 40: Corpus completo y ejecución reproducible del benchmark de 3 niveles + impacto ──
test("40: corpus sintético es-AR completo (13 tipos) y benchmark reproducible (Niveles 1–3, replay histórico e impacto)", async () => {
  const requiredTypes = [
    "saludo_corto", "respuesta_casual", "pregunta_tecnica", "chiste", "respuesta_sarcastica",
    "donacion", "reaccion_emocional", "respuesta_extensa", "mensaje_con_numeros",
    "mensaje_con_abreviaturas", "nombre_propio", "conversacion_discord", "prioritario_durante_especulacion",
  ];
  const types = new Set(CORPUS.turns.map(t => t.type));
  for (const rt of requiredTypes) assert.ok(types.has(rt), `falta tipo de corpus: ${rt}`);
  assert.equal(CORPUS.synthetic, true);
  assert.equal(Object.keys(PROFILES).sort().join(","), "A,B,C,D,E");

  const l1 = await runLevel1({ profiles: ["A", "B", "C", "D"], warmups: 1 });
  assert.equal(l1.aggregates.L1_A_edge_wav.counts.warmupsExcluded, 1);
  assert.equal(l1.aggregates.L1_B_fish_wav.counts.measured, 4);
  assert.equal(l1.aggregates.L1_C_sglang_wav.counts.measured, 4);
  assert.equal(l1.aggregates.L1_D_sglang_stream.counts.measured, 4);
  // H02: B, C y D producen exactamente la misma duración y cantidad de muestras para cada texto fijo
  for (const item of CORPUS.ttsIsolatedTexts) {
    const rB = l1.audioEquivalenceTable.find(r => r.profile === "B" && r.caseId === item.id);
    const rC = l1.audioEquivalenceTable.find(r => r.profile === "C" && r.caseId === item.id);
    const rD = l1.audioEquivalenceTable.find(r => r.profile === "D" && r.caseId === item.id);
    assert.equal(rB.audioDurationSec, rC.audioDurationSec, `B vs C duración en ${item.id}`);
    assert.equal(rC.audioDurationSec, rD.audioDurationSec, `C vs D duración en ${item.id}`);
    assert.equal(rB.samples, rC.samples, `B vs C muestras en ${item.id}`);
    assert.equal(rC.samples, rD.samples, `C vs D muestras en ${item.id}`);
  }
  // H05: replay histórico en tres vistas separadas reproduciendo exactamente Fase 2 causal (10 calib / 30 eval)
  assert.equal(l1.historicalReplay.category, "historical_replay");
  assert.equal(l1.historicalReplay.metadata.withCharsCount, 40);
  assert.equal(l1.historicalReplay.metadata.withoutCharsCount, 30);
  assert.equal(l1.historicalReplay.causalPhase2.aggregates.SGLang_PCM_adaptive_causal30.counts.measured, 30);
  assert.equal(l1.historicalReplay.causalPhase2.aggregates.SGLang_PCM_adaptive_causal30.counts.underrunCount, 0);
  // R7 (interpolación lineal en N=30 par): WAV p50=8315 ms / PCM p50=6845 ms (equivalente al método histórico de índice redondeado de sim-buffer-policies.mjs: 8470 / 6950 ms)
  assert.equal(l1.historicalReplay.causalPhase2.aggregates.SGLang_WAV_causal30.metrics.estimatedPlaybackStartRelMs.median, 8315);
  assert.equal(l1.historicalReplay.causalPhase2.aggregates.SGLang_PCM_adaptive_causal30.metrics.estimatedPlaybackStartRelMs.median, 6845);
  assert.equal(l1.historicalReplay.descriptiveAll40.aggregates.SGLang_PCM_adaptive_N40.counts.measured, 40);
  assert.ok(l1.historicalReplay.theoreticalOracle.aggregate.metrics.estimatedPlaybackStartRelMs.median > 0);

  const l2 = await runLevel2({ profiles: ["C", "D", "E"] });
  assert.equal(l2.aggregates.L2_C_wav.counts.measured, 6);
  assert.equal(l2.aggregates.L2_D_stream.counts.measured, 6);
  assert.equal(l2.aggregates.L2_E_stream_overlap.counts.measured, 6);
  assert.ok(l2.aggregates.L2_E_stream_overlap.counts.speculationsUsed > 0, "Perfil E debe aprovechar especulación en Nivel 2");

  const l3 = await runLevel3({ profiles: ["A", "D", "E"] });
  assert.equal(l3.aggregates.L3_A_edge_wav.counts.measured, 8);
  assert.equal(l3.aggregates.L3_D_sglang_stream.counts.measured, 8);
  assert.equal(l3.aggregates.L3_E_sglang_stream_overlap.counts.measured, 8);

  const impact = runImpactBenchmark({ iterations: 100, longRunIterations: 400 });
  assert.equal(impact.off.storedHistory, 0);
  assert.equal(impact.on.storedHistory, 100);
  assert.equal(impact.longRunBounded.boundedOk, true);
});

// ─── H03: Fiabilidad del ACK de reproducción en TTSEngine real ────────────────
test("H03: validación estricta de tts_play_started y tts_playback_stats (participantes, silenciadas, duplicados, tardíos, malformados)", async () => {
  const srv = await startMockTtsServer({ sglangFirstChunkMs: 15, sglangChunkGapMs: 10 });
  const tr = new TurnTracer({ enabled: true, category: "mock" });
  const h = createIsolatedHarness({ profileId: "D", serverUrl: srv.url, tracer: tr });
  try {
    // 1) Sesión PCM con 2 vistas audibles + 1 silenciada + 1 ajena no participante + 1 panel
    const vMuted = { readyState: 1, ttsStream: true, ttsMute: true, _viewId: "obs_muted" };
    const vAud1 = { readyState: 1, ttsStream: true, ttsMute: false, _viewId: "obs_aud_1" };
    const vAud2 = { readyState: 1, ttsStream: true, ttsMute: false, _viewId: "obs_aud_2" };
    const vRogue = { readyState: 1, ttsStream: true, ttsMute: false, _viewId: "obs_rogue" };
    const vPanel = { readyState: 1, ttsStream: false, isPanel: true, _viewId: "panel_1" };
    h.tts.streams.hub = { clients: () => [vMuted, vAud1, vAud2], send() {} };

    const tid = tr.startTurn({ source: "live", platform: "tiktok", requestedEngine: "sglang", playbackMode: "stream" });
    const synth = await h.tts.synth("Prueba de múltiples vistas y ACK.", { source: "live", returnMeta: true, turnId: tid });
    await h.tts.play(synth.url, "Prueba de múltiples vistas y ACK.", false, { turnId: tid });
    const st = h.tts.streams.get(synth.url);

    // Ajena, panel y payload malformado son rechazados
    assert.equal(h.tts.onPlaybackStarted(vRogue, { id: st.id, url: synth.url, mode: "stream", startMethod: "worklet_first_render" }), false);
    assert.equal(h.tts.onPlaybackStarted(vPanel, { id: st.id, url: synth.url, mode: "stream" }), false);
    assert.equal(h.tts.onPlaybackStarted(vAud1, null), false);
    assert.equal(h.tts.onPlaybackStarted(vAud1, { id: 99999, url: "stream:99999", mode: "stream" }), false);

    // Vista silenciada llega primero: NO debe fijar la métrica principal audible cuando hay vistas audibles
    assert.equal(h.tts.onPlaybackStarted(vMuted, { id: st.id, url: synth.url, mode: "stream", startMethod: "worklet_first_render", mute: true, clientRenderStartMs: 5 }), true);
    assert.equal(tr.getActive(tid).playback.estimatedPlaybackStartRelMs, null, "vista silenciada no debe falsear inicio audible");
    assert.notEqual(tr.getActive(tid).playback.mutedPlaybackStartRelMs, null);

    // Primera vista audible fija la métrica principal; segunda audible se registra individualmente; duplicado se ignora
    assert.equal(h.tts.onPlaybackStarted(vAud1, { id: st.id, url: synth.url, mode: "stream", startMethod: "worklet_first_render", clientRenderStartMs: 14, contextRate: 48000, contextState: "running" }), true);
    const firstStart = tr.getActive(tid).playback.estimatedPlaybackStartRelMs;
    assert.ok(firstStart !== null);
    assert.equal(tr.getActive(tid).playback.startSource, "first_audible_view");
    assert.equal(tr.getActive(tid).playback.startMethod, "worklet_first_render");
    assert.equal(h.tts.onPlaybackStarted(vAud1, { id: st.id, url: synth.url, mode: "stream", clientRenderStartMs: 999 }), false, "duplicado rechazado");
    assert.equal(tr.getActive(tid).playback.duplicateStartAcks, 1);

    await sleep(10);
    assert.equal(h.tts.onPlaybackStarted(vAud2, { id: st.id, url: synth.url, mode: "stream", startMethod: "worklet_first_render", clientRenderStartMs: 24, contextRate: 48000 }), true);
    assert.equal(tr.getActive(tid).playback.estimatedPlaybackStartRelMs, firstStart, "segunda vista no pisa el inicio de la primera audible");
    assert.ok(tr.getActive(tid).playback.lastAudibleStartAckRelMs >= firstStart);
    assert.equal(tr.getActive(tid).playback.views.length, 3);

    // Campos arbitrarios desde navegador se descartan (allowlist estricta)
    assert.equal(h.tts.onPlaybackStats(vAud1, { id: st.id, url: synth.url, stats: { underruns: 0, contextRate: 48000, evilField: "hack", prompt: "privado" } }), true);
    const v1Clock = tr.getActive(tid).playback.views.find(v => v.viewId === "obs_aud_1").clientClock;
    assert.equal(v1Clock.evilField, undefined);
    assert.equal(v1Clock.prompt, undefined);
    assert.equal(v1Clock.contextRate, 48000);

    // Tras cancelar/finalizar, un ACK tardío es rechazado
    h.tts.stop("live");
    assert.equal(h.tts.onPlaybackStarted(vAud2, { id: st.id, url: synth.url, mode: "stream" }), false);
    tr.finishTurn(tid, { state: "cancelled" });

    // 2) Caso donde TODAS las vistas están silenciadas: se etiqueta explícitamente muted_only
    h.tts.streams.hub = { clients: () => [vMuted], send() {} };
    const tidM = tr.startTurn({ source: "live", platform: "tiktok", requestedEngine: "sglang", playbackMode: "stream" });
    const synthM = await h.tts.synth("Solo vistas silenciadas.", { source: "live", returnMeta: true, turnId: tidM });
    await h.tts.play(synthM.url, "Solo vistas silenciadas.", false, { turnId: tidM });
    const stM = h.tts.streams.get(synthM.url);
    assert.equal(h.tts.onPlaybackStarted(vMuted, { id: stM.id, url: synthM.url, mode: "stream", startMethod: "worklet_first_render", mute: true, clientRenderStartMs: 9 }), true);
    assert.equal(tr.getActive(tidM).playback.allViewsMuted, true);
    assert.equal(tr.getActive(tidM).playback.startSource, "muted_only");
    assert.notEqual(tr.getActive(tidM).playback.estimatedPlaybackStartRelMs, null);
    h.tts.stop("live");
    tr.finishTurn(tidM, { state: "cancelled" });
  } finally {
    h.cleanup();
    await srv.close();
  }
});

// ─── H04: Flujo real integrado (StreamChatManager + StreamPlayer navegador) ───
test("H04: flujo real de StreamChatManager (expiración PRI.look, reemplazo idle, fallback LLM) y StreamPlayer navegador", async () => {
  const srv = await startMockTtsServer({ sglangWavTotalMs: 20 });
  const tr = new TurnTracer({ enabled: true, category: "mock" });
  const h = createIsolatedHarness({ profileId: "C", serverUrl: srv.url, tracer: tr });
  try {
    // 1) Expiración real de PRI.look en StreamChatManager._pump()
    h.chat._isResponding = true;
    h.chat._enqueue("NYRA", "comentario viejo de visión", "vision", { rawPrompt: true }, h.PRI.look, 0);
    h.chat.queue[0].at = Date.now() - 20000; // supera LOOK_TTL_MS=8000
    h.chat._pump();
    const expired = tr.getHistory().find(t => t.platform === "vision");
    assert.equal(expired?.state, "expired");
    assert.equal(expired?.queue.expired, true);

    // 2) Reemplazo real de idle al llegar chat prioritario
    h.chat._enqueue("NYRA", "dato curioso", "idle", { rawPrompt: true }, h.PRI.idle, 9999);
    h.chat._enqueue("ana", "hola chat", "tiktok", {}, h.PRI.chat, 9999);
    const replaced = tr.getHistory().find(t => t.platform === "idle");
    assert.equal(replaced?.state, "discarded");
    assert.equal(replaced?.queue.replaced, true);
    h.chat.queue = [];
    h.chat._isResponding = false;

    // 3) Propagación real de fallback LLM desde chatStream hacia tracer en _prepare
    const origChatStream = h.ai.chatStream.bind(h.ai);
    h.ai.chatStream = async function* (prompt, user, client, img, q, trace, auth) {
      if (trace) { trace.fallbackUsed = true; trace.effectiveProvider = "agy"; }
      yield* origChatStream(prompt, user, client, img, q, trace, auth);
    };
    h.replyQueue.push("Respuesta servida tras fallback de proveedor.");
    const tidFb = tr.startTurn({ source: "live", platform: "tiktok", requestedProvider: "claude-code", requestedEngine: "sglang", playbackMode: "wav" });
    const prepFb = await h.chat._prepare({ username: "ana", text: "hola", platform: "tiktok", opts: {}, rank: 4, at: Date.now(), turnId: tidFb }, h.ai.llm);
    await h.chat._respond({ username: "ana", text: "hola", platform: "tiktok", opts: {}, rank: 4, ready: prepFb, turnId: tidFb });
    const recFb = tr.getHistory().find(t => t.ids.turnId === tidFb);
    assert.equal(recFb.llm.fallbackUsed, true);
    assert.equal(recFb.effectiveProvider, "agy");
  } finally {
    h.cleanup();
    await srv.close();
  }

  // 4) StreamPlayer real (navegador) con stubs de AudioContext / AudioWorkletNode (48000 Hz)
  const origWorklet = globalThis.AudioWorkletNode;
  const posted = [];
  let workletInstance = null;
  globalThis.AudioWorkletNode = class {
    constructor() {
      this.port = { postMessage: m => posted.push(m), onmessage: null };
      workletInstance = this;
    }
    connect(x) { return x; }
    disconnect() {}
  };
  try {
    const sp = new StreamPlayer();
    sp.ctx = {
      state: "running",
      sampleRate: 48000, // H08: remuestreo soportado (no exige 44100)
      baseLatency: 0.005,
      outputLatency: 0.01,
      destination: {},
      createGain: () => ({ gain: { value: 1 }, connect() {}, disconnect() {} }),
      resume: () => Promise.resolve(),
    };
    let finishedStats = null;
    const sess = sp.createSession({ id: 5, url: "stream:5", sampleRate: 44100, estDurationSec: 1.0 }, {
      onFinish: s => { finishedStats = { ...s.stats }; },
    });
    sess.begin();
    sess.push(0, new Int16Array(44100), 0);
    workletInstance.port.onmessage({ data: { type: "started", consumed: 128 } });
    workletInstance.port.onmessage({ data: { type: "underrun", consumed: 22050 } });
    sess.push(1, new Int16Array(44100), 0);
    sess.end(88200, false);
    workletInstance.port.onmessage({ data: { type: "drained", consumed: 88200 } });
    assert.ok(finishedStats);
    assert.equal(finishedStats.contextRate, 48000);
    assert.equal(finishedStats.contextState, "running");
    assert.ok(finishedStats.bufferingStartMs !== null);
    assert.ok(finishedStats.effectiveRenderStartMs !== null);
    assert.equal(finishedStats.underruns, 1);
    assert.equal(finishedStats.underrunRecoveries, 1);
    assert.ok(finishedStats.streamEndMs !== null);
    assert.ok(finishedStats.playbackEndMs !== null);
  } finally {
    globalThis.AudioWorkletNode = origWorklet;
  }
});

// ─── H06: Exportación de trazas reales capturadas (con datos sintéticos), XSS y auth ─
test("H06: exportCapturedTraces (categoría real_gateway con datos sintéticos), escape HTML XSS, redacción URLs/rutas y auth /api", () => {
  const tr = new TurnTracer({ enabled: true, category: "real_gateway" });
  const tid = tr.startTurn({ source: "live", platform: "tiktok", requestedProvider: "claude-code", requestedEngine: "sglang", playbackMode: "stream" });
  tr.markEnqueued(tid, { priority: 4 });
  tr.markPrepareStart(tid);
  const job = tr.startTtsJob(tid, { requestedEngine: "sglang", effectiveEngine: "sglang", playbackMode: "stream" });
  tr.updateTtsJob(tid, job.ttsJobId, { requestStart: true, headers: true, firstValidPcm: true, httpBodyEnd: true, audioDurationSec: 2.5, bytesReceived: 220500, samplesReceived: 110250 });
  tr.markPrepareEnd(tid);
  tr.markPlaybackDispatched(tid, { url: "stream:1", mode: "stream", participants: 1, audibleParticipants: 1 });
  tr.recordPlaybackStartAck(tid, { viewId: "obs_1", audible: true, mode: "stream", startMethod: "worklet_first_render", clientRenderStartMs: 20, contextRate: 48000 });
  tr.markPlaybackEnd(tid, { result: "done" });
  // Intentar colar URL privada / ruta Windows / Bearer en reasonCode
  tr.finishTurn(tid, { state: "completed", reason: "https://privado.ngrok.app/v1 C:\\Users\\priv\\key.txt" });

  const captured = tr.getHistory({ category: "real_gateway" });
  const tmp = mkdtempSync(join(tmpdir(), "nyra-real-export-"));
  try {
    const { doc, files } = exportCapturedTraces(captured, {
      outDir: tmp,
      runId: "capture_<script>alert(1)</script>",
      commit: "86dbf3f",
      category: "real_gateway",
      synthetic: true,
      dataOrigin: "synthetic_test",
      config: { ttsEngine: "sglang", url: "https://secreto.ngrok.app", apiKey: "sk-secret-99999999" },
    });
    assert.equal(doc.category, "real_gateway");
    assert.equal(doc.synthetic, true, "prueba sintética etiquetada real_gateway queda marcada como sintética");
    assert.equal(doc.dataOrigin, "synthetic_test");
    assert.equal(doc.traces.length, 1);
    assert.equal(doc.config.url, undefined, "URL privada excluida de config exportada");
    assert.equal(doc.config.apiKey, undefined, "apiKey excluida de config exportada");
    const jsonRaw = readFileSync(files.jsonPath, "utf8");
    assert.doesNotMatch(jsonRaw, /privado\.ngrok\.app|secreto\.ngrok\.app|sk-secret|C:\\\\Users/);
    const htmlRaw = readFileSync(files.htmlPath, "utf8");
    assert.doesNotMatch(htmlRaw, /<script>alert\(1\)<\/script>/, "HTML debe escapar runId/campos dinámicos");
    assert.match(htmlRaw, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }

  // Verificar que /api/diagnostics/traces y /export quedan protegidos detrás del middleware app.use("/api", ...) en server.js
  const serverSrc = readFileSync(new URL("../server.js", import.meta.url), "utf8");
  const authIdx = serverSrc.indexOf('app.use("/api"');
  const getDiagIdx = serverSrc.indexOf('app.get("/api/diagnostics/traces"');
  const postExportIdx = serverSrc.indexOf('app.post("/api/diagnostics/traces/export"');
  assert.ok(authIdx > 0 && getDiagIdx > authIdx && postExportIdx > authIdx, "endpoints de diagnóstico deben ir detrás del middleware de autenticación /api");
});

// ─── H07: Equivalencia funcional integrada con telemetría OFF vs ON vs fallando ──
test("H07: escenario integrado equivalente con telemetría OFF, ON y con fallo interno produce idéntico comportamiento", async () => {
  const srv = await startMockTtsServer({ sglangFirstChunkMs: 20, sglangChunkGapMs: 12 });
  const runScenario = async tracer => {
    const h = createIsolatedHarness({ profileId: "E", serverUrl: srv.url, tracer, llmDelays: { firstDeltaMs: 10, deltaGapMs: 20 } });
    const reqBefore = srv.getRequests();
    const spoken = [];
    const origRemember = h.ai.remember;
    h.ai.remember = (role, txt) => { if (role === "assistant") spoken.push(txt); return origRemember(role, txt); };
    try {
      const inputs = [
        { user: "u1", text: "pregunta normal", reply: "Che, qué buena pregunta la tuya sobre el stream. Te cuento cómo funciona paso a paso.", rank: 4 },
        { user: "u2", text: "donación prioritaria", reply: "¡Muchas gracias por la donación gigante! Sos un genio total.", rank: 0 },
      ];
      for (const it of inputs) {
        h.replyQueue.push(it.reply);
        h.chat._enqueue(it.user, it.text, "tiktok", { donor: it.rank === 0 }, it.rank, 0);
      }
      const deadline = Date.now() + 8000;
      while ((h.chat.queue.length > 0 || h.chat._isResponding) && Date.now() < deadline) await sleep(15);
      await sleep(30);
      return { spoken, httpReqs: srv.getRequests() - reqBefore };
    } finally {
      h.cleanup();
    }
  };
  try {
    const resOff = await runScenario(new TurnTracer({ enabled: false, category: "mock" }));
    const resOn = await runScenario(new TurnTracer({ enabled: true, category: "mock" }));
    const resBroken = await runScenario(new TurnTracer({ enabled: true, category: "mock", now: () => { throw new Error("boom"); } }));

    assert.deepEqual(resOn.spoken, resOff.spoken, "telemetría ON no altera respuestas ni prioridades");
    assert.deepEqual(resBroken.spoken, resOff.spoken, "fallo de telemetría no altera respuestas ni prioridades");
    assert.equal(resOn.httpReqs, resOff.httpReqs, "telemetría ON no altera cantidad de solicitudes TTS");
    assert.equal(resBroken.httpReqs, resOff.httpReqs, "fallo de telemetría no altera solicitudes TTS");
  } finally {
    await srv.close();
  }
});

// ─── Seguridad final: acceso a /api/diagnostics/*, traversal y pertenencia WAV ACK ─
test("Seguridad final: guardas de /api/diagnostics/traces*, anti-traversal y pertenencia estricta de consumidores en ACK WAV", async () => {
  // 1) Política conservadora de acceso a diagnóstico cuando PANEL_PASSWORD está definida o vacía
  assert.equal(checkDiagnosticsAccess({ panelPassword: "secret", remoteAddress: "192.168.1.20" }).ok, true);
  assert.equal(checkDiagnosticsAccess({ panelPassword: "", remoteAddress: "127.0.0.1" }).ok, true);
  assert.equal(checkDiagnosticsAccess({ panelPassword: "", remoteAddress: "::1" }).ok, true);
  assert.equal(checkDiagnosticsAccess({ panelPassword: "", remoteAddress: "::ffff:127.0.0.1" }).ok, true);
  const lanDenied = checkDiagnosticsAccess({ panelPassword: "", remoteAddress: "192.168.1.50" });
  assert.equal(lanDenied.ok, false);
  assert.equal(lanDenied.status, 403);
  const proxyDenied = checkDiagnosticsAccess({ panelPassword: "", remoteAddress: "127.0.0.1", forwarded: "203.0.113.9" });
  assert.equal(proxyDenied.ok, false);
  assert.equal(proxyDenied.status, 403);

  // 2) Prevención de path traversal y nombres manipulados en exportación
  const tmpBase = mkdtempSync(join(tmpdir(), "nyra-sec-exp-"));
  try {
    assert.ok(resolveAuthorizedExportDir(tmpBase, "capture_2026-11-28").startsWith(tmpBase));
    assert.throws(() => resolveAuthorizedExportDir(tmpBase, "../escape"), /inválido|fuera/i);
    assert.throws(() => resolveAuthorizedExportDir(tmpBase, "..\\escape"), /inválido|fuera/i);
    assert.throws(() => resolveAuthorizedExportDir(tmpBase, "sub/nested"), /inválido/i);
    assert.throws(() => resolveAuthorizedExportDir(tmpBase, ""), /inválido/i);
  } finally {
    rmSync(tmpBase, { recursive: true, force: true });
  }

  // 3) Verificar en server.js que /api/diagnostics/traces y /export usan requireDiagnosticsAccess, 409 si telemetría off sin trazas, y authorizedBaseDir
  const serverSrc = readFileSync(new URL("../server.js", import.meta.url), "utf8");
  assert.match(serverSrc, /app\.get\("\/api\/diagnostics\/traces"[\s\S]{0,180}requireDiagnosticsAccess\(req, res\)/);
  assert.match(serverSrc, /app\.post\("\/api\/diagnostics\/traces\/export"[\s\S]{0,180}requireDiagnosticsAccess\(req, res\)/);
  assert.match(serverSrc, /status\(409\)\.json/);
  assert.match(serverSrc, /authorizedBaseDir:\s*baseDir/);

  // 4) ACK de reproducción WAV: solo vistas que participaron al despachar tts_play pueden enviar ACK
  const srv = await startMockTtsServer({ sglangWavTotalMs: 20 });
  const tr = new TurnTracer({ enabled: true, category: "mock" });
  const h = createIsolatedHarness({ profileId: "C", serverUrl: srv.url, tracer: tr });
  try {
    const vAud = { readyState: 1, ttsStream: false, ttsMute: false, _viewId: "wav_aud_1" };
    const vMuted = { readyState: 1, ttsStream: false, ttsMute: true, _viewId: "wav_muted_1" };
    const vOutsider = { readyState: 1, ttsStream: false, ttsMute: false, _viewId: "wav_outsider" };
    const vPanel = { readyState: 1, ttsStream: false, isPanel: true, _viewId: "panel_1" };
    h.tts.streams.hub = { clients: () => [vAud, vMuted], send() {} };
    h.tts._broadcast = () => {}; // no auto-enviar ACK para probar manualmente

    const tid = tr.startTurn({ source: "live", platform: "tiktok", requestedEngine: "sglang", playbackMode: "wav" });
    const synth = await h.tts.synth("Verificación de consumidores WAV.", { source: "live", returnMeta: true, turnId: tid });
    await h.tts.play(synth.url, "Verificación de consumidores WAV.", false, { turnId: tid });

    // Vista ajena que conoce la URL exacta y vista panel deben ser rechazadas
    assert.equal(h.tts.onPlaybackStarted(vOutsider, { url: synth.url, mode: "wav", startMethod: "html_audio_onplaying", clientRenderStartMs: 5 }), false);
    assert.equal(h.tts.onPlaybackStarted(vPanel, { url: synth.url, mode: "wav", startMethod: "html_audio_onplaying" }), false);
    assert.equal(tr.getActive(tid).playback.estimatedPlaybackStartRelMs, null);

    // Vista silenciada participante no pisa la métrica principal audible
    assert.equal(h.tts.onPlaybackStarted(vMuted, { url: synth.url, mode: "wav", startMethod: "html_audio_onplaying", mute: true, clientRenderStartMs: 7 }), true);
    assert.equal(tr.getActive(tid).playback.estimatedPlaybackStartRelMs, null);

    // Vista audible participante fija inicio; duplicado rechazado; tras stop, ACK tardío rechazado
    assert.equal(h.tts.onPlaybackStarted(vAud, { url: synth.url, mode: "wav", startMethod: "html_audio_onplaying", clientRenderStartMs: 12 }), true);
    assert.notEqual(tr.getActive(tid).playback.estimatedPlaybackStartRelMs, null);
    assert.equal(tr.getActive(tid).playback.startMethod, "html_audio_onplaying");
    assert.equal(h.tts.onPlaybackStarted(vAud, { url: synth.url, mode: "wav", startMethod: "html_audio_onplaying", clientRenderStartMs: 99 }), false);
    assert.equal(tr.getActive(tid).playback.duplicateStartAcks, 1);

    h.tts.stop("live");
    assert.equal(h.tts.onPlaybackStarted(vAud, { url: synth.url, mode: "wav" }), false);
    tr.finishTurn(tid, { state: "cancelled" });
  } finally {
    h.cleanup();
    await srv.close();
  }
});
