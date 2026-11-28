// Reproductor progresivo de NYRA (hilo principal). Una sesión por tts_stream_start; el PCM llega en frames binarios
// [ "NYP1" | u32 id | u32 seq | u32 ageMs | PCM s16le ]. Chunks de ids desconocidos (cancelados/viejos) o seq repetidas
// se ignoran. La política adaptativa decide cuándo arrancar y cuándo reanudar tras un corte.
import { AdaptiveStartPolicy } from "./buffer-policy.js";

const HEADER = 16;
const MAX_SESSION_BYTES = 25 * 1024 * 1024;
const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export class StreamPlayer {
  static supported() {
    return typeof AudioWorkletNode !== "undefined" && typeof AudioContext !== "undefined" && window.isSecureContext !== false && littleEndian;
  }

  constructor() {
    this.ctx = null;
    this.sessions = new Map();
  }

  async init() {
    // Pedir 44100 evita remuestrear en JS (Chromium remuestrea a la salida). Si el navegador no lo acepta, el worklet
    // remuestrea de forma continua a la frecuencia que tenga el contexto.
    try { this.ctx = new AudioContext({ sampleRate: 44100, latencyHint: "interactive" }); }
    catch { this.ctx = new AudioContext({ latencyHint: "interactive" }); }
    await this.ctx.audioWorklet.addModule(new URL("./pcm-player-worklet.js", import.meta.url));
    return this;
  }

  get running() { return this.ctx?.state === "running"; }
  resume() { return this.ctx?.resume().catch(() => {}); }

  createSession(meta, cb) {
    this.sessions.get(meta.id)?.cancel("replaced");
    const s = new StreamSession(this, meta, cb);
    this.sessions.set(meta.id, s);
    return s;
  }

  onBinary(buf) {
    if (!(buf instanceof ArrayBuffer) || buf.byteLength < HEADER || (buf.byteLength - HEADER) % 2) return false;
    const dv = new DataView(buf);
    if (dv.getUint32(0, false) !== 0x4e595031) return false; // "NYP1"
    const s = this.sessions.get(dv.getUint32(4, true));
    if (!s) return false; // tardío o de una sesión cancelada
    return s.push(dv.getUint32(8, true), new Int16Array(buf, HEADER), dv.getUint32(12, true));
  }
}

class StreamSession {
  constructor(player, meta, cb = {}) {
    this.player = player;
    this.meta = meta;
    this.id = meta.id;
    this.url = meta.url;
    this.cb = cb;
    this.rate = meta.sampleRate || 44100;
    const ctx = player.ctx;
    this.node = new AudioWorkletNode(ctx, "nyra-pcm-player", { numberOfInputs: 0, outputChannelCount: [1], processorOptions: { inRate: this.rate } });
    this.gain = ctx.createGain();
    this.gain.gain.value = meta.mute ? 0 : 1; // ?mute o ya suena por Discord: no se oye, pero el lip-sync sigue
    this.node.connect(this.gain).connect(ctx.destination);
    this.policy = new AdaptiveStartPolicy({
      estDurationSec: meta.estDurationSec || null, priorRate: meta.priorRate || 0.63,
      // Fase 3: respuesta en dos pedidos (1.ª oración anticipada + resto)
      ...(meta.segmentBoundarySamples ? {
        segmentBoundarySec: meta.segmentBoundarySamples / (meta.sampleRate || 44100),
        segmentGapSec: meta.segmentGapSec || 3.8,
        segmentStartedAt: (performance.now() - (meta.segmentAgeMs || 0)) / 1000,
      } : {}),
    });
    this.node.port.onmessage = ({ data }) => this._onWorklet(data);
    this.t0 = performance.now();
    this.lastSeq = -1;
    this.received = 0;
    this.played = 0;
    this.level = 0;
    this.ended = false;
    this.begun = false;
    this.playing = false;
    this.closed = false;
    this.underrunAt = null;
    this.stats = { firstChunkMs: null, audibleStartMs: null, bufferAtStartSec: null, underruns: 0, underrunMs: 0, seqGaps: 0, duplicates: 0, receivedSamples: 0, playedSamples: 0, contextRate: ctx.sampleRate, outputLatencyMs: null, result: null };
  }

  get playedSec() { return this.played / this.rate; }
  get totalSec() { return this.ended ? this.received / this.rate : null; }

  begin() {
    if (this.closed || this.begun) return;
    this.begun = true;
    this.timer = setInterval(() => this._tick(), 50);
    this._tick();
  }

  push(seq, pcm, ageMs = 0) {
    if (this.closed || this.ended) return false;
    if (seq <= this.lastSeq) { this.stats.duplicates += 1; return false; }
    // Tope por sesión igual al del servidor (25 MB): un emisor defectuoso no puede llenar la memoria del navegador
    if ((this.received + pcm.length) * 2 > MAX_SESSION_BYTES) { this._finish("error", "too_large"); return false; }
    if (seq !== this.lastSeq + 1) this.stats.seqGaps += 1; // no debería pasar sobre TCP; se registra
    this.lastSeq = seq;
    const now = performance.now();
    if (this.stats.firstChunkMs === null) this.stats.firstChunkMs = Math.round(now - this.t0);
    this.received += pcm.length;
    // ageMs: cuánto esperó el chunk en el servidor (precargado); así la velocidad de generación no se distorsiona
    this.policy.onArrival((now - ageMs) / 1000, this.received / this.rate);
    this.node.port.postMessage({ type: "push", pcm }, [pcm.buffer]); // se transfiere el frame entero (sin copiar)
    this._tick();
    return true;
  }

  // El servidor avisa dónde empieza el segundo pedido (si el primero terminó después de tts_stream_start)
  setSegment(boundarySamples, gapSec = 3.8, ageMs = 0) {
    if (this.closed || !(boundarySamples > 0)) return;
    this.policy.segmentBoundarySec = boundarySamples / this.rate;
    this.policy.segmentGapSec = gapSec;
    this.policy.segmentStartedAt = (performance.now() - (ageMs || 0)) / 1000;
    this._tick();
  }

  end(_totalSamples, partial = false) {
    if (this.closed) return;
    this.ended = true;
    this.partial = partial;
    this.policy.onEnd(performance.now() / 1000, this.received / this.rate);
    this.node.port.postMessage({ type: "end" });
    if (!this.received) this._finish(partial ? "error" : "done");
    else this._tick();
  }

  _tick() {
    if (!this.begun || this.playing || this.closed) return;
    if (this.player.ctx.state !== "running") { this.player.resume(); return; }
    if (!this.policy.shouldPlay(performance.now() / 1000, this.playedSec)) return;
    this.playing = true;
    this._closeUnderrun();
    if (this.stats.bufferAtStartSec === null) this.stats.bufferAtStartSec = Number(((this.received - this.played) / this.rate).toFixed(3));
    this.node.port.postMessage({ type: "play" });
  }

  _onWorklet(d) {
    if (this.closed) return;
    if (typeof d.consumed === "number") this.played = d.consumed;
    if (d.type === "tick") { this.level = d.state === "playing" ? d.level : 0; return; }
    if (d.type === "started") {
      // inicio estimado del renderizado: no es el sonido físico en el parlante (falta la latencia del dispositivo)
      const ctx = this.player.ctx;
      this.stats.outputLatencyMs = Math.round(((ctx.outputLatency || 0) + (ctx.baseLatency || 0)) * 1000);
      if (this.stats.audibleStartMs === null) {
        this.stats.audibleStartMs = Math.round(performance.now() - this.t0);
        this.cb.onAudible?.(this);
      }
      return;
    }
    if (d.type === "underrun") {
      this.playing = false;
      this.level = 0;
      this.stats.underruns += 1;
      this.underrunAt = performance.now();
      this.policy.onUnderrun();
      this.cb.onUnderrun?.(this);
      return;
    }
    if (d.type === "drained") this._finish(this.partial ? "error" : "done");
  }

  // Se reanudó tras un corte (el worklet vuelve a "playing" con play()): contabilizar la duración del corte
  _closeUnderrun() {
    if (this.underrunAt !== null) { this.stats.underrunMs += Math.round(performance.now() - this.underrunAt); this.underrunAt = null; }
  }

  cancel(reason = "cancelled") { this._finish("cancelled", reason); }

  _finish(result, reason = null) {
    if (this.closed) return;
    this._closeUnderrun();
    this.closed = true;
    this.level = 0;
    clearInterval(this.timer);
    try { this.node.port.postMessage({ type: "stop" }); } catch { }
    try { this.node.disconnect(); this.gain.disconnect(); } catch { }
    this.node.port.onmessage = null;
    this.player.sessions.delete(this.id);
    Object.assign(this.stats, { receivedSamples: this.received, playedSamples: this.played, result: result === "done" ? "done" : result === "cancelled" ? "cancelled" : "error" });
    this.cb.onFinish?.(this, result, reason);
  }
}
