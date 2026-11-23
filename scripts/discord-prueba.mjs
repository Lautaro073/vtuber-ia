// Primera prueba de Discord, todavía separada del server:
//   node scripts/discord-prueba.mjs               entra al canal, avisa que escucha y muestra lo que entendió de cada persona
//   node scripts/discord-prueba.mjs --sin-discord prueba la misma cadena (voz, opus, Whisper, nombre) sin token
import { execFileSync } from "node:child_process";
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { Readable } from "node:stream";
import { setTimeout as sleep } from "node:timers/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import * as voice from "@discordjs/voice";
import OpusScript from "opusscript";
import { MsEdgeTTS, OUTPUT_FORMAT } from "msedge-tts";
import { DiscordVoice, Whisper, readPhrase, listenTo } from "../discord-voice.js";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
try { process.loadEnvFile(join(ROOT, ".env")); } catch { } // sin .env solo anda --sin-discord
const WAKE_RE = new Function(`${readFileSync(join(ROOT, "server.js"), "utf8").match(/^const WAKE_RE = .*$/m)[0]}\nreturn WAKE_RE;`)();
async function synth(text, dir, name) {
  const tts = new MsEdgeTTS();
  await tts.setMetadata(process.env.EDGE_VOICE || "es-AR-ElenaNeural", OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3);
  const out = mkdtempSync(join(dir, `${name}-`));
  await tts.toFile(out, text);
  tts.close?.();
  return join(out, "audio.mp3");
}

async function selfCheck(dir) {
  const whisper = new Whisper();
  try {
    console.log(`Whisper: ${await whisper.ready}`);
    await checkAudio(dir, whisper);
  } finally {
    whisper.stop();
  }
}

async function checkAudio(dir, whisper) {
  for (const [text, wake] of [["Nira, decime quién va ganando.", true], ["Mirá el mapa que viene el jungla.", false]]) {
    const mp3 = await synth(text, dir, "check");
    const raw = execFileSync("ffmpeg", ["-v", "error", "-i", mp3, "-f", "s16le", "-ar", "48000", "-ac", "2", "-"], { maxBuffer: 2e7 });
    const encoder = new OpusScript(48000, 2); // como manda Discord
    const packets = [];
    for (let i = 0; i + 3840 <= raw.length; i += 3840) packets.push(Buffer.from(encoder.encode(raw.subarray(i, i + 3840), 960)));
    encoder.delete();
    const { text: heard, ms } = await whisper.transcribe((await readPhrase(Readable.from(packets))).pcm);
    console.log(`«${text}» → «${heard}» (${ms} ms)`);
    assert.equal(WAKE_RE.test(heard), wake, wake ? "no la reconoció" : "se despertó sola");

    let playable = 0;
    for await (const _ of voice.createAudioResource(mp3).playStream) playable++;
    assert.ok(playable > 20, "el mp3 no se pudo convertir para Discord");
  }
  // Frases seguidas, con pausa adentro y la anterior todavía transcribiéndose: un lector por persona
  const receiver = new voice.VoiceReceiver({});
  const got = [];
  let reading = 0, maxReading = 0;
  const subscribe = receiver.subscribe.bind(receiver);
  receiver.subscribe = (...args) => {
    maxReading = Math.max(maxReading, ++reading);
    return subscribe(...args).once("close", () => reading--);
  };
  receiver.speaking.on("start", listenTo(receiver, async (uid, pcm) => { got.push(pcm.length / 32); await sleep(1300); }));
  const encoder = new OpusScript(16000, 1);
  const frame = Buffer.from(encoder.encode(Buffer.alloc(640, 9), 320));
  encoder.delete();
  const talk = async ms => {
    for (const end = Date.now() + ms; Date.now() < end; await sleep(20)) {
      receiver.speaking.onPacket("x");
      receiver.subscriptions.get("x")?.push(frame);
    }
  };
  await talk(500);
  await sleep(950);
  await talk(1200); // la transcripción anterior termina acá adentro
  await sleep(350);
  await talk(500);
  await sleep(1600);
  assert.equal(got.length, 2, "cortó mal las frases");
  assert.equal(maxReading, 1, "dos lectores para la misma persona");
  assert.ok(got[1] > 1250, "perdió las pausas o parte del audio");

  console.log("cadena de audio sin Discord: ok");
}

async function runBot(dir) {
  await new Promise((resolve, reject) => {
    const bot = new DiscordVoice({
      synth: text => synth(text, dir, "voz"),
      onSpeech: ({ name, streamer, text }) =>
        console.log(`${WAKE_RE.test(text) ? "✓ LA NOMBRÓ" : "  escucha   "}  ${name}${streamer ? " (streamer)" : ""}: «${text}»`),
      onState: s => {
        if (s.state === "on") console.log(`Adentro de "${s.channel}" (Whisper ${s.whisper}). Ctrl+C para salir.`);
        else if (s.error) reject(new Error(s.error));
        else if (s.state === "off") resolve();
      },
    });
    process.once("SIGINT", () => bot.stop());
    bot.start();
  });
}

const dir = mkdtempSync(join(tmpdir(), "nyra-discord-"));
try {
  await (process.argv.includes("--sin-discord") ? selfCheck(dir) : runBot(dir));
} catch (err) {
  console.error(`❌ ${err.message}`);
  process.exitCode = 1;
} finally {
  rmSync(dir, { recursive: true, force: true });
}
