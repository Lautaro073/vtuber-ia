// NYRA en un canal de voz de Discord: escucha a cada persona por separado (Whisper local) y habla con su misma voz.
// Lo usan server.js y scripts/discord-prueba.mjs. No guarda audio.
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Client, Events, GatewayIntentBits } from "discord.js";
import * as voice from "@discordjs/voice";
import OpusScript from "opusscript";

const ROOT = dirname(fileURLToPath(import.meta.url));
const PAUSE_MS = 800;               // silencio que cierra una frase
const MIN_BYTES = 16000 * 2 * 0.3;  // menos de 0,3 s es ruido
const MAX_BYTES = 16000 * 2 * 20;   // frase más larga: se corta y sigue en otra
const MAX_WAITING = 10;             // frases esperando a Whisper; más que eso ya es charla vieja
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
export async function readPhrase(packets) {
  const decoder = new OpusScript(16000, 1);
  const parts = [];
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
          const silence = Buffer.alloc(Math.round(Math.min(gap, PAUSE_MS) * 16) * 2);
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
  return { pcm: Buffer.concat(parts), cut: bytes >= MAX_BYTES };
}

// Un solo lector por persona. Se libera al cortar la frase, no al terminar de transcribirla
export function listenTo(receiver, onPhrase) {
  const open = new Set();
  const listen = async uid => {
    if (open.has(uid)) return;
    open.add(uid);
    try {
      let phrase;
      try {
        // suscribirse antes de cualquier await: si no, se pierde el principio, justo donde va el nombre
        phrase = await readPhrase(receiver.subscribe(uid, { end: { behavior: voice.EndBehaviorType.AfterSilence, duration: PAUSE_MS } }));
      } finally {
        open.delete(uid);
      }
      if (phrase.cut) setImmediate(() => listen(uid)); // recién cuando la suscripción anterior se cerró
      if (phrase.pcm.length >= MIN_BYTES) await onPhrase(uid, phrase.pcm);
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
        if (m.ready) return resolve(m.where);
        const p = this.pending.get(m.id);
        this.pending.delete(m.id);
        if (m.error) p?.reject(new Error(m.error)); else p?.resolve(m);
      });
    });
    this.ready.catch(() => { }); // lo informa quien lo espera
    this.proc.stdin.on("error", () => { }); // si murió, lo informa el evento exit
  }

  transcribe(pcm) {
    if (this.pending.size >= MAX_WAITING) return Promise.resolve({ text: "" });
    return new Promise((resolve, reject) => {
      this.pending.set(++this.id, { resolve, reject });
      this.proc.stdin.write(JSON.stringify({ id: this.id, pcm: pcm.toString("base64") }) + "\n");
    });
  }

  stop() { this.stopped = true; this.proc.kill(); }
}

export class DiscordVoice {
  // synth(texto) → archivo o URL de audio · onSpeech({ id, name, streamer, text, raw, ms, secs }) · onState({ state, ... })
  constructor({ synth, onSpeech, onState = () => { } }) {
    Object.assign(this, { synth, onSpeech, onState });
    this.state = { state: "off" };
    this.run = 0; // invalida un arranque en curso si tocan SALIR
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

      const conn = this.conn = voice.joinVoiceChannel({ guildId, channelId, adapterCreator: guild.voiceAdapterCreator, selfDeaf: false, selfMute: false });
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
      if (!await this._notice(GREETING)) console.warn("🎧 No pude saludar al entrar: sigo escuchando igual");
      check();
      const listen = listenTo(conn.receiver, async (uid, pcm) => {
        if (run !== this.run || uid === client.user.id) return;
        const member = await guild.members.fetch(uid);
        if (member.user.bot) return;
        const { text, raw, ms } = await this.whisper.transcribe(pcm);
        // también lo que se descartó (raw): es lo que sirve para pulir los filtros
        if (run === this.run && (text || raw)) await this.onSpeech({ id: uid, name: member.displayName, streamer: uid === this.streamerId, text, raw, ms, secs: pcm.length / 32000 });
      });
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

  stopAudio() { this.player?.stop(true); }

  stop(error) {
    this.run++;
    this.player?.stop(true);
    try { this.conn?.destroy(); } catch { } // ya destruida
    this.client?.destroy().catch(() => { });
    this.whisper?.stop();
    this.player = this.conn = this.client = this.whisper = this.channel = null;
    this._set(error ? { state: "off", error } : { state: "off" });
  }
}
