// NYRA en un canal de voz de Discord: escucha a cada persona por separado (Whisper local) y habla con su misma voz.
// Lo usan server.js y scripts/discord-prueba.mjs. No guarda audio.
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Client, Events, GatewayIntentBits } from "discord.js";
import * as voice from "@discordjs/voice";
import OpusScript from "opusscript";

const ROOT = dirname(fileURLToPath(import.meta.url));
const PAUSE_MS = 800;               // silencio que cierra una frase
const MIN_BYTES = 16000 * 2 * 0.3;  // menos de 0,3 s es ruido
const MAX_BYTES = 16000 * 2 * 20;   // frase más larga: se corta y sigue en otra
const MAX_WAITING = 10;             // frases esperando a Whisper; más que eso ya es charla vieja
export const VALID_LOCAL_PAUSE_MS = [400, 600, 800, 900, 1000, 1200];
export const LOCAL_LEGACY_DEFAULTS = Object.freeze({
  profile: "legacy",
  beamSize: 5,
  temperature: 0.0,
  vadFilter: true,
  vadThreshold: 0.8,
  hotwords: "",
  promptMode: "narrative",
  secondPassMode: "always_only_name",
  filterProfile: "legacy",
  pauseMs: PAUSE_MS,
});
// Al entrar saluda. Sin aviso de que escucha: el grupo ya está de acuerdo (lo pidió el streamer el 2026-11-22)
export const GREETING = "Hola pibes, ¿qué sale hoy?";

// La librería entrega solo el opus: se le deja su marca de tiempo (RTP), y si uno no se descifra se saltea
// (la librería tiraba la frase entera)
export function keepTiming(receiver) {
  const parse = receiver.parsePacket.bind(receiver);
  let failed = 0;
  receiver.parsePacket = (msg, ...rest) => {
    try {
      const opus = parse(msg, ...rest);
      if (!opus) return null;
      const packet = Buffer.from(opus);
      packet.ts = msg.readUInt32BE(4);
      return packet;
    } catch (err) {
      if (++failed % 100 === 1) console.error(`🎧 Paquete de audio sin descifrar (${failed}): ${err.message}`);
      return null;
    }
  };
}

// Paquetes opus de una persona → audio mono de 16 kHz, en memoria
export async function readPhrase(packets, pauseMs = PAUSE_MS) {
  const decoder = new OpusScript(16000, 1);
  const parts = [];
  const tStart = Date.now();
  let bytes = 0, last = 0, endTs = null;
  try {
    for await (const packet of packets) {
      try {
        const now = Date.now();
        const timed = packet.ts !== undefined;
        // Discord no manda nada en las pausas: se reponen por la marca de tiempo, que no se mueve si la red se atrasa
        let gap = timed && endTs !== null ? ((packet.ts - endTs) | 0) / 48 : 0;
        if (gap < -1000) gap = 0; // reloj reiniciado
        else if (gap < -5) continue; // repetido o fuera de orden
        if (gap < 20 && last) { // sin marca, o un cliente que no adelanta el reloj en las pausas: por llegada
          const waited = now - last - 20;
          if (waited >= (timed ? 250 : 100)) gap = waited;
        }
        last = now;
        if (gap >= 20) {
          const silence = Buffer.alloc(Math.round(Math.min(gap, pauseMs) * 16) * 2);
          parts.push(silence);
          bytes += silence.length;
        }
        const pcm = Buffer.from(decoder.decode(packet));
        if (timed) endTs = (packet.ts + pcm.length / 2 * 3) >>> 0; // 16 kHz → reloj de 48 kHz
        parts.push(pcm);
        bytes += pcm.length;
      } catch { } // un paquete roto no tira la frase
      if (bytes >= MAX_BYTES) break;
    }
  } finally {
    decoder.delete();
  }
  return { pcm: Buffer.concat(parts), cut: bytes >= MAX_BYTES, segmentationMs: Date.now() - tStart };
}

// Un solo lector por persona. Se libera al cortar la frase, no al terminar de transcribirla
export function listenTo(receiver, onPhrase, { pauseMs = PAUSE_MS, getPauseMs = null } = {}) {
  const open = new Set();
  const listen = async uid => {
    if (open.has(uid)) return;
    open.add(uid);
    try {
      const effectivePause = typeof getPauseMs === "function" ? getPauseMs() : pauseMs;
      let phrase;
      try {
        // suscribirse antes de cualquier await: si no, se pierde el principio, justo donde va el nombre
        phrase = await readPhrase(receiver.subscribe(uid, { end: { behavior: voice.EndBehaviorType.AfterSilence, duration: effectivePause } }), effectivePause);
      } finally {
        open.delete(uid);
      }
      if (phrase.cut) setImmediate(() => listen(uid)); // recién cuando la suscripción anterior se cerró
      if (phrase.pcm.length >= MIN_BYTES) await onPhrase(uid, phrase.pcm, phrase.segmentationMs);
    } catch (err) {
      if (err.code !== "ERR_STREAM_PREMATURE_CLOSE") console.error("🎧 Audio de Discord:", err.message); // ese es solo salir con alguien hablando
    }
  };
  return listen;
}

export class Whisper {
  constructor(onExit = () => { }) {
    this.pending = new Map();
    this.id = 0;
    this.info = { model: null, device: null, computeType: null, cudaError: null, where: null };
    // sin las claves de NYRA: el proceso de Python no las necesita
    const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => !/TOKEN|SECRET|PASSWORD|API_KEY/i.test(k)));
    this.proc = spawn(join(ROOT, ".whisper", "venv", "Scripts", "python.exe"), [join(ROOT, "scripts", "whisper-worker.py")],
      { windowsHide: true, stdio: ["pipe", "pipe", "inherit"], env });
    this.ready = new Promise((resolve, reject) => {
      const fail = err => {
        reject(err);
        for (const p of this.pending.values()) p.reject(err);
        this.pending.clear();
        if (!this.stopped) onExit(err);
      };
      this.proc.on("error", err => fail(new Error(`Whisper no arrancó: ${err.message}`)));
      this.proc.on("exit", code => fail(new Error(`Whisper se cerró (código ${code})`)));
      createInterface({ input: this.proc.stdout }).on("line", line => {
        let m; try { m = JSON.parse(line); } catch { return; }
        if (m.ready) {
          this.info = {
            where: m.where || null,
            requestedModel: m.requested_model || "large-v3-turbo",
            model: m.model || null,
            device: m.device || null,
            computeType: m.compute_type || null,
            cudaError: m.cuda_error || null,
            fallbackReason: m.fallback_reason || null,
            initMs: m.init_ms ?? null,
          };
          return resolve(m.where);
        }
        const p = this.pending.get(m.id);
        this.pending.delete(m.id);
        if (m.model) this.info.model = m.model;
        if (m.device) this.info.device = m.device;
        if (m.compute_type) this.info.computeType = m.compute_type;
        if (m.fallback_reason) this.info.fallbackReason = m.fallback_reason;
        const totalMs = p ? Date.now() - p.enqueuedAt : (m.ms || 0);
        const queueMs = p && typeof m.ms === "number" ? Math.max(0, totalMs - m.ms) : 0;
        if (m.error) p?.reject(Object.assign(new Error(m.error), { status: m.status || "worker_error" }));
        else p?.resolve({ ...m, queue_ms: queueMs, total_ms: totalMs, pending_count: p.pendingCount });
      });
    });
    this.ready.catch(() => { }); // lo informa quien lo espera
    this.proc.stdin.on("error", () => { }); // si murió, lo informa el evento exit
  }

  transcribe(pcm, opts = {}) {
    const pendingCount = this.pending.size;
    if (pendingCount >= MAX_WAITING) {
      return Promise.resolve({
        text: "",
        ms: 0,
        queue_ms: 0,
        total_ms: 0,
        pending_count: pendingCount,
        status: "queue_full",
        discard_reason: "queue_full",
      });
    }
    return new Promise((resolve, reject) => {
      this.pending.set(++this.id, { resolve, reject, enqueuedAt: Date.now(), pendingCount });
      const payload = { id: this.id, pcm: pcm.toString("base64") };
      if (opts.profile) payload.profile = opts.profile;
      if (opts.noPrompt) payload.no_prompt = true;
      if (opts.usePrompt !== undefined) payload.use_prompt = opts.usePrompt;
      if (opts.promptMode) payload.prompt_mode = opts.promptMode;
      if (opts.hotwords !== undefined) payload.hotwords = opts.hotwords;
      if (opts.vadFilter !== undefined) payload.vad_filter = opts.vadFilter;
      if (opts.vadThreshold !== undefined) payload.vad_threshold = opts.vadThreshold;
      if (opts.beamSize !== undefined) payload.beam_size = opts.beamSize;
      if (opts.temperature !== undefined) payload.temperature = opts.temperature;
      if (opts.secondPassMode) payload.second_pass_mode = opts.secondPassMode;
      if (opts.filterProfile) payload.filter_profile = opts.filterProfile;
      this.proc.stdin.write(JSON.stringify(payload) + "\n");
    });
  }

  stop() { this.stopped = true; this.proc.kill(); }
}

export class DiscordVoice {
  // synth(texto) → archivo o URL de audio · onSpeech({ id, name, streamer, text, raw, ms, secs }) · onState({ state, ... })
  constructor({ synth, onSpeech, onState = () => { }, listenOnly = false, whisperRemote = null, localSettingsFile = null }) {
    Object.assign(this, { synth, onSpeech, onState, listenOnly, whisperRemote }); // listenOnly: entra muda y sin saludar, solo escucha
    this.localSettingsFile = resolve(localSettingsFile || join(ROOT, "whisper_local_settings.json"));
    this.localConfig = this._loadLocalConfig();
    this.state = { state: "off" };
    this.run = 0; // invalida un arranque en curso si tocan SALIR
    this.sessionId = null;
    this._fragmentSeq = 0;
    this.localMetrics = {
      lastTranscript: "",
      lastMs: 0,
      lastSecs: 0,
      discardedCount: 0,
      lastDiscardReason: null,
      lastError: null,
    };
  }

  _loadLocalConfig() {
    try {
      if (!existsSync(this.localSettingsFile)) return { ...LOCAL_LEGACY_DEFAULTS };
      const parsed = JSON.parse(readFileSync(this.localSettingsFile, "utf8"));
      return this._validateLocalPatch(parsed, { ...LOCAL_LEGACY_DEFAULTS });
    } catch {
      return { ...LOCAL_LEGACY_DEFAULTS };
    }
  }

  _validateLocalPatch(patch = {}, base = this.localConfig || LOCAL_LEGACY_DEFAULTS) {
    const next = { ...base };
    if (patch.profile !== undefined) {
      const p = String(patch.profile || "").trim().toLowerCase();
      if (p !== "legacy" && p !== "experimental") throw new Error("Perfil STT local inválido (legacy o experimental)");
      next.profile = p;
      if (p === "legacy" && Object.keys(patch).length === 1) {
        return { ...LOCAL_LEGACY_DEFAULTS };
      }
    }
    if (patch.beamSize !== undefined) {
      const b = Math.round(Number(patch.beamSize));
      if (!Number.isFinite(b) || b < 1 || b > 5) throw new Error("Beam size local inválido (1..5)");
      next.beamSize = b;
    }
    if (patch.temperature !== undefined) {
      const t = Number(patch.temperature);
      if (!Number.isFinite(t) || t < 0 || t > 1) throw new Error("Temperatura local inválida (0.0..1.0)");
      next.temperature = Number(t.toFixed(2));
    }
    if (patch.vadFilter !== undefined) next.vadFilter = Boolean(patch.vadFilter);
    if (patch.vadThreshold !== undefined) {
      const v = Number(patch.vadThreshold);
      if (!Number.isFinite(v) || v < 0.1 || v > 0.95) throw new Error("VAD threshold local inválido (0.10..0.95)");
      next.vadThreshold = Number(v.toFixed(2));
    }
    if (patch.hotwords !== undefined) {
      next.hotwords = String(patch.hotwords || "").slice(0, 300).trim();
    }
    if (patch.promptMode !== undefined) {
      const pm = String(patch.promptMode || "").trim().toLowerCase();
      if (!["narrative", "none", "minimal"].includes(pm)) throw new Error("Modo de initial_prompt inválido");
      next.promptMode = pm;
    }
    if (patch.secondPassMode !== undefined) {
      const sp = String(patch.secondPassMode || "").trim().toLowerCase();
      if (!["always_only_name", "none", "confidence"].includes(sp)) throw new Error("Modo de segunda inferencia inválido");
      next.secondPassMode = sp;
    }
    if (patch.filterProfile !== undefined) {
      const fp = String(patch.filterProfile || "").trim().toLowerCase();
      if (!["legacy", "confidence_ghost", "no_ghost"].includes(fp)) throw new Error("Perfil de filtros inválido");
      next.filterProfile = fp;
    }
    if (patch.pauseMs !== undefined) {
      const pm = Math.round(Number(patch.pauseMs));
      if (!VALID_LOCAL_PAUSE_MS.includes(pm)) throw new Error(`Pausa Discord inválida (permitidas: ${VALID_LOCAL_PAUSE_MS.join(", ")} ms)`);
      next.pauseMs = pm;
    }
    return next;
  }

  updateLocalConfig(patch = {}) {
    this.localConfig = this._validateLocalPatch(patch, this.localConfig);
    mkdirSync(resolve(this.localSettingsFile, ".."), { recursive: true });
    const tmp = `${this.localSettingsFile}.${process.pid}.${Date.now()}.tmp`;
    writeFileSync(tmp, JSON.stringify(this.localConfig, null, 2));
    renameSync(tmp, this.localSettingsFile);
    return { ...this.localConfig };
  }

  get on() { return this.state.state === "on"; }

  // Quiénes están en el canal: así sabe quién es "monar documcara"
  names() {
    try { return [...this.channel.members.values()].filter(m => !m.user.bot).map(m => m.displayName); }
    catch { return []; } // sin canal
  }

  _set(state) { this.state = state; this.onState(state); }

  async start() {
    if (this.state.state === "on" || this.state.state === "starting") return;
    const run = ++this.run;
    const check = () => { if (run !== this.run) throw new Error("cancelado"); };
    const fail = err => {
      if (run !== this.run) return;
      console.error("🎧 Discord:", err.message);
      this.stop(err.message);
    };
    this._set({ state: "starting" });
    try {
      const keys = ["DISCORD_BOT_TOKEN", "DISCORD_GUILD_ID", "DISCORD_VOICE_CHANNEL_ID"];
      const missing = keys.filter(k => !process.env[k]?.trim());
      if (missing.length) throw new Error(`Falta en .env: ${missing.join(", ")}`);
      const [token, guildId, channelId] = keys.map(k => process.env[k].trim());

      this.whisper = new Whisper(fail);
      const client = this.client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates] });
      client.on("error", err => console.error("🎧 Discord:", err.message));
      const online = new Promise(resolve => client.once(Events.ClientReady, resolve));
      const where = await this.whisper.ready; check();
      await client.login(token); check();
      await online; check();
      const guild = await client.guilds.fetch(guildId);
      const channel = await guild.channels.fetch(channelId); check();
      this.channel = channel;
      if (!channel?.isVoiceBased()) throw new Error("DISCORD_VOICE_CHANNEL_ID no es un canal de voz");
      if (!channel.joinable || !channel.speakable) throw new Error("Al bot le falta Ver canal, Conectar o Hablar en ese canal");
      // Por defecto el streamer es quien creó el BOT (no el servidor, que puede ser de un amigo): decide en quién confía
      const owner = (await client.application.fetch()).owner;
      this.streamerId = process.env.STREAMER_DISCORD_ID?.trim() || owner?.ownerId || owner?.id; check();

      const conn = this.conn = voice.joinVoiceChannel({ guildId, channelId, adapterCreator: guild.voiceAdapterCreator, selfDeaf: false, selfMute: this.listenOnly });
      conn.on("error", err => console.error("🎧 Voz:", err.message));
      await voice.entersState(conn, voice.VoiceConnectionStatus.Ready, 30000); check();
      conn.on(voice.VoiceConnectionStatus.Disconnected, async () => {
        try {
          // un corte corto se reconecta solo
          await Promise.race([
            voice.entersState(conn, voice.VoiceConnectionStatus.Signalling, 5000),
            voice.entersState(conn, voice.VoiceConnectionStatus.Connecting, 5000),
          ]);
        } catch { fail(new Error("se cortó la conexión de voz (o la sacaron del canal)")); }
      });

      this.player = voice.createAudioPlayer();
      this.player.on("error", err => console.error("🎧 Reproduciendo:", err.message));
      conn.subscribe(this.player);
      client.on(Events.VoiceStateUpdate, (before, after) => {
        if (after.id === client.user.id) {
          // solo escucha en el canal configurado: si la mueven, se va
          if (after.channelId && after.channelId !== channelId) fail(new Error("la movieron de canal"));
        }
      });

      // Primero saluda, recién después escucha
      if (!this.listenOnly && !await this._notice(GREETING)) console.warn("🎧 No pude saludar al entrar: sigo escuchando igual");
      check();
      this.sessionId = `${Date.now().toString(36)}-${run}`;
      this._fragmentSeq = 0;
      this.whisperRemote?.resetCancellation();
      const listen = listenTo(conn.receiver, async (uid, pcm, segmentationMs = 0) => {
        if (run !== this.run || uid === client.user.id) return;
        const capturedAt = Date.now();
        const fragmentId = `${this.sessionId}:${++this._fragmentSeq}`;
        const member = await guild.members.fetch(uid);
        if (member.user.bot || run !== this.run) return;
        const secs = pcm.length / 32000;
        const res = await this._transcribePhrase(pcm, { uid, capturedAt, secs, run, fragmentId, segmentationMs });
        if (run !== this.run) return; // descartar resultado tardío si ya se detuvo o reinició
        if (res.text || res.raw || res.meta?.discardReason === "queue_full" || res.meta?.discardReason === "inference_error" || res.meta?.discardReason === "worker_error") {
          await this.onSpeech({ id: uid, name: member.displayName, streamer: uid === this.streamerId, text: res.text, raw: res.raw, ms: res.ms, secs, meta: res.meta });
        }
      }, { getPauseMs: () => this.localConfig.pauseMs });
      keepTiming(conn.receiver);
      conn.receiver.speaking.on("start", listen);

      const streamer = this.streamerId && await guild.members.fetch(this.streamerId).catch(() => null); check();
      this._set({ state: "on", channel: channel.name, streamer: streamer ? streamer.displayName : "sin identificar (STREAMER_DISCORD_ID)", whisper: where });
    } catch (err) {
      fail(err);
    }
  }

  async _notice(text) {
    if (!this.player || this.player.state.status !== voice.AudioPlayerStatus.Idle || this.noticing) return false;
    this.noticing = true;
    try {
      const file = await this.synth(text);
      if (!file || !this.player || this.player.state.status !== voice.AudioPlayerStatus.Idle) return false; // empezó a hablar: no saluda encima
      this.noticePlaying = true;
      const complete = await this._play(file);
      if (this.noticePlaying && complete) return true;
      return false;
    } catch (err) {
      console.error("🎧 Aviso de Discord:", err.message);
      return false;
    } finally {
      this.noticePlaying = false;
      this.noticing = false;
    }
  }

  // Dice algo en el canal; termina cuando dejó de sonar
  play(file) {
    if (!this.on || !file) return Promise.resolve(false);
    this.noticePlaying = false; // una respuesta corta el saludo
    return this._play(file);
  }

  _play(file) {
    const { player } = this;
    const res = voice.createAudioResource(file);
    return new Promise(resolve => {
      const end = ok => { clearTimeout(timer); player.off("stateChange", onChange); resolve(ok); };
      const onChange = (before, after) => {
        if (before.resource !== res || after.resource === res) return;
        end(after.status === voice.AudioPlayerStatus.Idle && res.playbackDuration > 0);
      };
      const timer = setTimeout(() => end(false), 90000);
      player.on("stateChange", onChange);
      player.play(res);
    });
  }

  async _transcribePhrase(pcm, { uid, capturedAt, secs, run, fragmentId = null, segmentationMs = 0 }) {
    const useRemote = this.whisperRemote && this.whisperRemote.config.engine === "remote";
    const requestedEngine = useRemote ? "remote" : "local";
    let fallbackReason = null;
    if (useRemote) {
      try {
        const res = await this.whisperRemote.transcribePcm(pcm, { speakerId: uid, capturedAt });
        return {
          text: res.text,
          raw: null,
          ms: res.ms,
          meta: {
            requestedEngine,
            effectiveEngine: "remote",
            requestedModel: this.whisperRemote.config.model,
            effectiveModel: res.model || this.whisperRemote.config.model,
            device: res.device || "cpu",
            computeType: res.quantization || "int8",
            fallbackReason: null,
            sessionId: this.sessionId,
            fragmentId,
            audioDurationSec: Number(secs.toFixed(3)),
            segmentationMs,
            totalMs: res.ms,
            discardReason: res.text ? null : "valid_silence",
          },
        };
      } catch (err) {
        if (err.cancelled || run !== this.run) {
          return { text: "", raw: null, ms: 0, meta: { requestedEngine, effectiveEngine: "remote", sessionId: this.sessionId, fragmentId, discardReason: run !== this.run ? "late_result" : "cancelled" } };
        }
        if (err.stale) return { text: "", raw: null, ms: 0, meta: { requestedEngine, effectiveEngine: "remote", sessionId: this.sessionId, fragmentId, discardReason: "expired" } };
        if (err.queueFull) return { text: "", raw: null, ms: 0, meta: { requestedEngine, effectiveEngine: "remote", sessionId: this.sessionId, fragmentId, discardReason: "queue_full" } };
        const ageMs = Date.now() - capturedAt;
        if (!this.whisperRemote.config.fallbackLocal || ageMs > this.whisperRemote.config.maxAgeMs) {
          return { text: "", raw: null, ms: 0, meta: { requestedEngine, effectiveEngine: "remote", sessionId: this.sessionId, fragmentId, discardReason: ageMs > this.whisperRemote.config.maxAgeMs ? "expired" : "inference_error" } };
        }
        fallbackReason = `remote_error: ${err.message}`;
        this.whisperRemote.recordFallback();
        console.warn(`🎧 STT remoto falló (${err.message}) → fallback a Whisper local`);
      }
    }
    try {
      const lc = this.localConfig || LOCAL_LEGACY_DEFAULTS;
      const localRes = await this.whisper.transcribe(pcm, {
        profile: lc.profile === "experimental" ? "EXPERIMENTAL" : "LEGACY",
        beamSize: lc.beamSize,
        temperature: lc.temperature,
        vadFilter: lc.vadFilter,
        vadThreshold: lc.vadThreshold,
        hotwords: lc.hotwords,
        promptMode: lc.promptMode,
        secondPassMode: lc.secondPassMode,
        filterProfile: lc.filterProfile,
      });
      const discardReason = localRes.text
        ? null
        : (localRes.discard_reason === "vad_no_speech" && !localRes.raw ? "valid_silence" : (localRes.discard_reason || "empty_transcript"));
      this.localMetrics.lastTranscript = localRes.text || "";
      this.localMetrics.lastMs = localRes.ms || 0;
      this.localMetrics.lastSecs = secs;
      if (!localRes.text && (localRes.raw || discardReason === "queue_full")) {
        this.localMetrics.discardedCount += 1;
        this.localMetrics.lastDiscardReason = discardReason;
      }
      return {
        ...localRes,
        meta: {
          requestedEngine,
          effectiveEngine: "local",
          requestedModel: localRes.requested_model || this.whisper?.info?.requestedModel || "large-v3-turbo",
          effectiveModel: localRes.model || this.whisper?.info?.model || null,
          device: localRes.device || this.whisper?.info?.device || null,
          computeType: localRes.compute_type || this.whisper?.info?.computeType || null,
          fallbackReason: fallbackReason || localRes.fallback_reason || this.whisper?.info?.fallbackReason || null,
          sessionId: this.sessionId,
          fragmentId,
          audioDurationSec: localRes.audio_duration_sec ?? Number(secs.toFixed(3)),
          postVadDurationSec: localRes.post_vad_duration_sec ?? null,
          segmentationMs,
          queueMs: localRes.queue_ms ?? 0,
          pendingCount: localRes.pending_count ?? 0,
          inferenceMs: localRes.inference_ms ?? localRes.ms ?? 0,
          postprocessMs: localRes.postprocess_ms ?? 0,
          totalMs: localRes.total_ms ?? localRes.ms ?? 0,
          rtf: localRes.rtf ?? null,
          beamSize: localRes.beam_size ?? null,
          temperature: localRes.temperature ?? null,
          vadThreshold: localRes.vad_threshold ?? null,
          profile: localRes.profile || "A",
          usePrompt: localRes.use_prompt ?? null,
          hotwords: localRes.hotwords ?? null,
          passes: localRes.passes ?? 1,
          avgLogprob: localRes.avg_logprob ?? null,
          noSpeechProb: localRes.no_speech_prob ?? null,
          compressionRatio: localRes.compression_ratio ?? null,
          discardReason: run !== this.run ? "late_result" : discardReason,
        },
      };
    } catch (err) {
      this.localMetrics.lastError = err.message;
      this.localMetrics.discardedCount += 1;
      this.localMetrics.lastDiscardReason = err.status || "inference_error";
      return {
        text: "",
        raw: null,
        ms: 0,
        meta: {
          requestedEngine,
          effectiveEngine: "local",
          sessionId: this.sessionId,
          fragmentId,
          discardReason: err.status || "inference_error",
          error: err.message,
        },
      };
    }
  }

  getSttStatus() {
    const remoteCfg = this.whisperRemote?.getPublicConfig() || null;
    const selectedEngine = remoteCfg?.engine || "local";
    const localInfo = this.whisper?.info || {};
    return {
      selectedEngine,
      effectiveEngine: selectedEngine === "remote" && remoteCfg?.metrics?.status === "offline" && remoteCfg?.fallbackLocal ? "local" : selectedEngine,
      local: {
        ready: Boolean(this.whisper),
        ...localInfo,
        ...this.localMetrics,
      },
      localConfig: { ...this.localConfig },
      remote: remoteCfg,
    };
  }

  stopAudio() { this.player?.stop(true); }

  stop(error) {
    this.run++;
    this.whisperRemote?.abortAll();
    this.player?.stop(true);
    try { this.conn?.destroy(); } catch { } // ya destruida
    this.client?.destroy().catch(() => { });
    this.whisper?.stop();
    this.player = this.conn = this.client = this.whisper = this.channel = null;
    this._set(error ? { state: "off", error } : { state: "off" });
  }
}
