// Fase 2: reproducción progresiva de SGLang hacia las vistas del avatar (obs.html).
// Cada síntesis "stream" es una sesión con id propio. El servidor guarda el PCM recibido (acotado por el cliente
// HTTP) mientras espera su turno, y cuando TTSEngine.play() la llama lo reenvía a las vistas compatibles:
//   JSON  tts_stream_start {v, id, url, format, sampleRate, channels, mute, estDurationSec, priorRate}
//   BIN   [ "NYP1" | u32 id | u32 seq | u32 ageMs | PCM s16le mono 44100 ]   (cabecera de 16 bytes, PCM alineado)
//   JSON  tts_stream_end {id, totalSamples, partial}       fin de la GENERACIÓN/transmisión (no de la reproducción)
//   JSON  tts_stream_cancel {id, reason}                   cortar ya (Stop, consumidores perdidos, error sin audio)
// La vista responde tts_done {url:"stream:<id>"} recién cuando terminó de SONAR (fin de reproducción audible) y
// tts_stream_stats con sus métricas. Sin vistas compatibles, el mismo PCM se convierte a WAV (sin sintetizar de nuevo).
import { mkdirSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export const STREAM_PROTOCOL_VERSION = 1;
export const FRAME_MAGIC = "NYP1";
export const FRAME_HEADER_BYTES = 16;
const SAMPLE_RATE = 44100;
const MAX_WS_BUFFERED = 4 * 1024 * 1024;   // > ~23 s de audio sin salir: la vista está trabada
const UNCLAIMED_MS = 120000;               // una sesión lista que nadie reprodujo se descarta
const MAX_TOTAL_BYTES = 64 * 1024 * 1024;  // PCM retenido sumando TODAS las sesiones (cada una además ≤ 25 MB)
const GEN_TIMEOUT_MS = 330000;             // tope del turno mientras se genera: 300 s del cliente HTTP + margen
const PRIOR_RATE = 0.63;                   // s de audio por s de reloj (RTF ~1.6, benchmarks L4)
export const PRIOR_CPS = 14;               // caracteres por segundo de voz al arrancar (sin historial)

export function encodeFrame(id, seq, ageMs, pcm) {
  const out = Buffer.allocUnsafe(FRAME_HEADER_BYTES + pcm.length);
  out.write(FRAME_MAGIC, 0, "ascii");
  out.writeUInt32LE(id >>> 0, 4);
  out.writeUInt32LE(seq >>> 0, 8);
  out.writeUInt32LE(Math.max(0, Math.min(0xffffffff, Math.round(ageMs))) >>> 0, 12);
  pcm.copy(out, FRAME_HEADER_BYTES);
  return out;
}

export function decodeFrame(buf) {
  if (buf.length < FRAME_HEADER_BYTES || buf.toString("ascii", 0, 4) !== FRAME_MAGIC) return null;
  return { id: buf.readUInt32LE(4), seq: buf.readUInt32LE(8), ageMs: buf.readUInt32LE(12), pcm: buf.subarray(FRAME_HEADER_BYTES) };
}

export function pcmToWav(chunks, sampleRate = SAMPLE_RATE) {
  const dataSize = chunks.reduce((s, c) => s + c.length, 0);
  const h = Buffer.alloc(44);
  h.write("RIFF", 0, "ascii"); h.writeUInt32LE(36 + dataSize, 4); h.write("WAVE", 8, "ascii");
  h.write("fmt ", 12, "ascii"); h.writeUInt32LE(16, 16); h.writeUInt16LE(1, 20); h.writeUInt16LE(1, 22);
  h.writeUInt32LE(sampleRate, 24); h.writeUInt32LE(sampleRate * 2, 28); h.writeUInt16LE(2, 32); h.writeUInt16LE(16, 34);
  h.write("data", 36, "ascii"); h.writeUInt32LE(dataSize, 40);
  return Buffer.concat([h, ...chunks]);
}

export class SglangStreamManager {
  constructor({ outDir, logger = console, now = () => performance.now(), maxTotalBytes = MAX_TOTAL_BYTES, unclaimedMs = UNCLAIMED_MS } = {}) {
    this.outDir = outDir;
    this.maxTotalBytes = maxTotalBytes;
    this.unclaimedMs = unclaimedMs;
    this.totalBytes = 0;        // PCM retenido en memoria (todas las sesiones)
    this.logger = logger;
    this.now = now;
    this.hub = null;            // { clients(): ws[] abiertos, send(ws, data) } — lo inyecta server.js
    this.streams = new Map();   // url → sesión
    this.seq = 0;
    this.cps = PRIOR_CPS;       // caracteres por segundo de voz (se aprende de cada sesión terminada)
    this.lastStats = null;      // { server, client } de la última sesión reproducida (panel)
    this.statsById = new Map(); // últimas sesiones: la próxima puede terminar de generar mientras suena la actual
  }

  // Vistas abiertas que anunciaron soporte de PCM progresivo (tts_caps)
  capableClients() {
    return (this.hub?.clients() || []).filter(c => c.readyState === 1 && c.ttsStream);
  }

  // Duración esperada por texto. Incierta (±20–30 %): la política del navegador la infla y corrige al primer corte.
  // Conservador hasta validar el gateway real: lo aprendido solo puede ALARGAR la estimación (voz más lenta que 14 cps),
  // nunca acortarla. Con las trazas reales, acortarla producía cortes en respuestas emocionales (más lentas).
  estimateDurationSec(text) {
    return Number((String(text || "").length / Math.min(this.cps, PRIOR_CPS)).toFixed(2));
  }

  create({ source, text }) {
    const id = ++this.seq;
    let settle;
    const st = {
      id, url: `stream:${id}`, source, chars: String(text || "").length,
      estDurationSec: this.estimateDurationSec(text),
      chunks: [], seqNext: 0, bytes: 0, lastAt: null, samples: 0, forwarded: false, participants: null,
      state: "pending", playing: false, targets: null, mute: false, error: null, partial: false,
      ctl: null, created: this.now(),
      m: { queueMs: null, headersMs: null, firstPcmMs: null, firstSentMs: null, genEndMs: null, chunks: 0, maxGapMs: 0, result: "pending" },
    };
    st.settled = new Promise(r => { settle = r; });
    st._settle = settle;
    this.streams.set(st.url, st);
    return st;
  }

  get(url) { return this.streams.get(url) || null; }

  onHeaders(st) { if (st.state === "pending") st.state = "generating"; }

  // Lanza si se pasa del presupuesto global aun después de descartar sesiones abandonadas: el cliente HTTP aborta la
  // descarga y la sesión falla de forma controlada (fallback si todavía no sonó nada).
  onPcm(st, buf, info) {
    if (st.state === "cancelled" || st.state === "released") return;
    if (this.totalBytes + buf.length > this.maxTotalBytes) this._evict(st);
    if (this.totalBytes + buf.length > this.maxTotalBytes) {
      st.m.memoryLimit = true;
      throw new Error(`memoria de streaming agotada (${this.maxTotalBytes} bytes entre todas las sesiones)`);
    }
    const at = this.now();
    if (st.m.firstPcmMs === null) st.m.firstPcmMs = Math.round(at - st.created);
    st.m.chunks += 1;
    if (st.lastAt !== null) st.m.maxGapMs = Math.max(st.m.maxGapMs, Math.round(at - st.lastAt));
    st.lastAt = at;
    st.chunks.push({ seq: st.seqNext++, buf, at });
    st.bytes += buf.length;
    this.totalBytes += buf.length;
    st.samples += buf.length / 2;
    if (st.playing && st.targets) this._flush(st);
  }

  onEnd(st, metrics = {}) {
    if (st.state === "cancelled" || st.state === "released") return;
    st.state = "done";
    st.m.genEndMs = Math.round(this.now() - st.created);
    st.m.result = "ok";
    Object.assign(st.m, { headersMs: metrics.headersMs ?? st.m.headersMs, queueMs: metrics.queueMs ?? st.m.queueMs, retries: metrics.retries });
    const audioSec = st.samples / SAMPLE_RATE;
    if (audioSec > 0.5 && st.chars > 10) { // aprende la velocidad de habla real (suaviza y acota)
      this.cps = Math.max(8, Math.min(25, 0.7 * this.cps + 0.3 * (st.chars / audioSec)));
    }
    st._settle();
    if (st.playing && st.targets) {
      this._flush(st);
      this._sendAll(st, { type: "tts_stream_end", id: st.id, url: st.url, totalSamples: st.samples, partial: false });
      st.cb?.setTimeoutMs?.(Math.round(audioSec * 1000) + 15000); // a lo sumo, lo que dura el audio desde ahora
    }
    this._scheduleUnclaimed(st);
    this._record(st);
  }

  // Falla de generación. Sin nada enviado a una vista → fallback seguro. Con audio ya enviado → fin parcial, sin repetir.
  onFail(st, err) {
    if (st.state === "cancelled" || st.state === "released") return;
    st.state = "failed";
    st.error = err;
    st.partial = Boolean(err?.partial);
    st.m.result = err?.cancelled ? "cancelled" : st.partial ? "partial" : "error";
    st._settle();
    if (st.playing && st.targets) {
      if (st.forwarded) {
        this.logger?.warn?.(`⚠️ SGLang stream #${st.id}: falla parcial tras ${(st.samples / SAMPLE_RATE).toFixed(2)} s de audio — se termina lo recibido, sin fallback`);
        st.m.partialFailure = true;
        this._flush(st);
        this._sendAll(st, { type: "tts_stream_end", id: st.id, url: st.url, totalSamples: st.samples, partial: true });
        st.cb?.setTimeoutMs?.(Math.round(st.samples / SAMPLE_RATE * 1000) + 15000);
      } else {
        this._sendAll(st, { type: "tts_stream_cancel", id: st.id, url: st.url, reason: "error" });
        st.targets = null;
        st.cb?.fallback?.(); // todavía no sonó nada: el fallback no duplica audio
      }
    }
    this._scheduleUnclaimed(st);
    this._record(st);
  }

  // Decide cómo suena una sesión cuando le toca el turno.
  //   {mode:"stream", timeoutMs} | {mode:"wav", url} | {mode:"fallback"} | {mode:"none"}
  // cb: { setTimeoutMs(ms), fallback() } para eventos posteriores (fin de generación / falla sin audio).
  async play(st, { mute = false, cb = null } = {}) {
    if (!st || st.state === "cancelled" || st.state === "released") return { mode: "none" };
    st.playing = true;
    st.mute = mute;
    st.cb = cb;
    if (st.state === "failed") return st.forwarded ? { mode: "none" } : { mode: "fallback" };
    const targets = this.capableClients();
    if (!targets.length) {
      // Ninguna vista compatible ahora: esperar el final y entregar WAV (mismo audio, sin sintetizar otra vez)
      this.logger?.log?.(`🎧 SGLang stream #${st.id}: sin vistas compatibles — se entrega como WAV`);
      await st.settled;
      if (st.state === "done") return { mode: "wav", url: this._toWav(st) };
      if (st.state === "failed") return { mode: "fallback" };
      return { mode: "none" };
    }
    st.targets = new Set(targets);
    // Participantes: solo ellos pueden confirmar el fin. El turno termina cuando todas las vistas AUDIBLES avisaron
    // (o se fueron); si todas están silenciadas (?mute o el audio ya va por Discord), cuando avisaron todas.
    st.participants = new Map(targets.map(ws => [ws, { audible: !mute && !ws.ttsMute, done: false }]));
    st.m.participants = targets.length;
    st.m.audibleParticipants = targets.filter(ws => !mute && !ws.ttsMute).length;
    st.m.playAtMs = Math.round(this.now() - st.created);
    this._record(st);
    this._sendAll(st, {
      type: "tts_stream_start", v: STREAM_PROTOCOL_VERSION, id: st.id, url: st.url, source: st.source,
      format: "pcm_s16le", sampleRate: SAMPLE_RATE, channels: 1, mute,
      estDurationSec: st.estDurationSec, priorRate: PRIOR_RATE,
    });
    this._flush(st);
    if (st.state === "done") {
      this._sendAll(st, { type: "tts_stream_end", id: st.id, url: st.url, totalSamples: st.samples, partial: false });
      return { mode: "stream", timeoutMs: Math.round(st.samples / SAMPLE_RATE * 1000) + 15000 };
    }
    // Generando: la duración real se desconoce (la estimación por texto puede quedarse corta), así que el tope es el
    // límite duro del cliente HTTP; al terminar la generación se reemplaza por "audio total + 15 s" (onEnd/onFail).
    return { mode: "stream", timeoutMs: GEN_TIMEOUT_MS };
  }

  // Stop / invalidación: corta la reproducción en las vistas y libera la memoria. Abortar el HTTP lo hace TTSEngine.
  cancel(st, reason = "stop") {
    if (!st || st.state === "cancelled" || st.state === "released") return false;
    const wasGenerating = st.state === "generating" || st.state === "pending";
    if (st.targets) this._sendAll(st, { type: "tts_stream_cancel", id: st.id, url: st.url, reason });
    st.state = "cancelled";
    st.m.result = "cancelled";
    st.m.cancelReason = reason;
    st._settle();
    this._record(st);
    this._drop(st);
    return wasGenerating;
  }

  // Falló o se canceló antes de las cabeceras: nadie la vio, solo se registra y se libera
  discard(st, result = "error") {
    if (!st || st.state === "released") return;
    st.state = "released";
    st.m.result = result;
    st._settle();
    this._record(st);
    this._drop(st);
  }

  toWav(st) { return this._toWav(st); }

  cancelSource(source = null) {
    const hit = [];
    for (const st of [...this.streams.values()]) if (!source || st.source === source) { this.cancel(st, "stop"); hit.push(st); }
    return hit;
  }

  // tts_done de la vista (o timeout de seguridad): la sesión terminó de sonar
  release(url) {
    const st = this.get(url);
    if (!st) return null;
    const stillGenerating = st.state === "generating" || st.state === "pending";
    if (st.state !== "cancelled") st.m.result = st.m.result === "pending" ? "released_early" : st.m.result;
    st.state = "released";
    st._settle();
    this._drop(st);
    return { st, stillGenerating };
  }

  // tts_done de una vista. Idempotente; ignora vistas que no participaron. → "complete" | "waiting" | "ignored"
  ack(st, ws) {
    const p = st?.participants?.get(ws);
    if (!p || st.state === "cancelled" || st.state === "released") return "ignored";
    p.done = true;
    return this._complete(st) ? "complete" : "waiting";
  }

  // Una vista se desconectó (o dejó de poder reproducir). → { st, empty, complete } de la sesión en que participaba
  clientGone(ws) {
    for (const st of this.streams.values()) {
      if (!st.participants?.has(ws) && !st.targets?.has(ws)) continue;
      st.targets?.delete(ws);
      st.participants?.delete(ws);
      const empty = !st.participants?.size;
      return { st, empty, complete: !empty && this._complete(st) };
    }
    return null;
  }

  _complete(st) {
    const ps = [...(st.participants?.values() || [])];
    if (!ps.length) return false;
    const audible = ps.filter(p => p.audible);
    return (audible.length ? audible : ps).every(p => p.done);
  }

  recordClientStats(msg) {
    const st = this.statsById.get(Number(msg?.id));
    if (!st || typeof msg.stats !== "object" || !msg.stats) return false;
    const num = (v, max = 1e7) => (Number.isFinite(Number(v)) ? Math.max(0, Math.min(max, Number(v))) : null);
    const s = msg.stats;
    st.client = {
      firstChunkMs: num(s.firstChunkMs), audibleStartMs: num(s.audibleStartMs), bufferAtStartSec: num(s.bufferAtStartSec, 600),
      underruns: num(s.underruns, 1000), underrunMs: num(s.underrunMs), receivedSamples: num(s.receivedSamples, 2e8),
      playedSamples: num(s.playedSamples, 2e8), contextRate: num(s.contextRate, 384000), outputLatencyMs: num(s.outputLatencyMs, 10000),
      result: ["done", "cancelled", "error"].includes(s.result) ? s.result : null,
    };
    this.lastStats = st;
    return true;
  }

  _record(st) {
    const rec = this.statsById.get(st.id) || { client: null };
    rec.server = { id: st.id, source: st.source, samples: st.samples, audioSec: Number((st.samples / SAMPLE_RATE).toFixed(3)), estDurationSec: st.estDurationSec, partial: st.partial, ...st.m };
    this.statsById.set(st.id, rec);
    while (this.statsById.size > 8) this.statsById.delete(this.statsById.keys().next().value);
    if (st.playing || !this.lastStats) this.lastStats = rec;
  }

  _sendAll(st, obj) {
    const msg = JSON.stringify(obj);
    for (const ws of st.targets || []) if (ws.readyState === 1) try { this.hub.send(ws, msg); } catch { }
  }

  // Reenvía en orden lo que falta. Una vista trabada (bufferedAmount alto) se saca de la sesión, sin frenar a las demás.
  // Lo enviado se suelta enseguida (en modo stream no se vuelve a necesitar): la memoria no crece con la duración.
  _flush(st) {
    while (st.chunks.length && st.targets) {
      const c = st.chunks.shift();
      st.bytes -= c.buf.length;
      this.totalBytes -= c.buf.length;
      const frame = encodeFrame(st.id, c.seq, this.now() - c.at, c.buf);
      for (const ws of [...st.targets]) {
        if (ws.readyState !== 1) { this._participantGone(st, ws); continue; }
        if ((ws.bufferedAmount || 0) > MAX_WS_BUFFERED) {
          this.logger?.warn?.(`⚠️ SGLang stream #${st.id}: vista lenta, se la saca de la sesión`);
          try { this.hub.send(ws, JSON.stringify({ type: "tts_stream_cancel", id: st.id, url: st.url, reason: "slow" })); } catch { }
          this._participantGone(st, ws);
          continue;
        }
        try { this.hub.send(ws, frame); } catch { this._participantGone(st, ws); }
        if (!st.targets) break;
      }
      if (!st.forwarded) { st.forwarded = true; st.m.firstSentMs = Math.round(this.now() - st.created); }
    }
  }

  // Una participante se fue en medio de _flush (lenta/cerrada): puede completar el turno o dejar la sesión sin oyentes
  _participantGone(st, ws) {
    st.targets?.delete(ws);
    st.participants?.delete(ws);
    if (!st.participants?.size) { st.targets = null; queueMicrotask(() => st.cb?.empty?.()); }
    else if (this._complete(st)) queueMicrotask(() => st.cb?.complete?.());
  }

  // Libera sesiones terminadas que nadie está reproduciendo (las más viejas primero), sin tocar la que pide memoria
  _evict(except) {
    for (const st of [...this.streams.values()]) {
      if (this.totalBytes <= this.maxTotalBytes * 0.75) break;
      if (st === except || st.playing || (st.state !== "done" && st.state !== "failed")) continue;
      this.logger?.warn?.(`⚠️ SGLang stream #${st.id}: descartada por memoria (nadie la reproducía)`);
      st.m.result = "evicted";
      this._record(st);
      this._drop(st);
    }
  }

  _toWav(st) {
    mkdirSync(this.outDir, { recursive: true });
    const id = `${Date.now()}${Math.floor(Math.random() * 1000)}`; // 13 primeros dígitos = timestamp (limpieza existente)
    const tmp = join(this.outDir, `${id}.wav.tmp`);
    const file = join(this.outDir, `${id}.wav`);
    try {
      writeFileSync(tmp, pcmToWav(st.chunks.map(c => c.buf)));
      renameSync(tmp, file);
    } catch (err) {
      try { unlinkSync(tmp); } catch { }
      throw err;
    }
    this._drop(st);
    return `/tts/${id}.wav`;
  }

  _scheduleUnclaimed(st) {
    if (st.playing) return;
    clearTimeout(st._unclaimed);
    st._unclaimed = setTimeout(() => { if (!st.playing) { st.m.result = "unclaimed"; this._record(st); this._drop(st); } }, this.unclaimedMs);
    st._unclaimed.unref?.();
  }

  _drop(st) {
    clearTimeout(st._unclaimed);
    this.totalBytes -= st.bytes;
    st.bytes = 0;
    st.chunks = [];
    st.participants = null;
    st.targets = null;
    st.cb = null;
    this.streams.delete(st.url);
  }
}
