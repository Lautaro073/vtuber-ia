// Benchmark reproducible de STT: compara Whisper local (con y sin initial_prompt) vs Faster Whisper remoto.
// Uso: node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json [--remote-url https://...] [--fish-concurrent 0|1]
// El manifest.json debe ser un array de { file: "clip.wav", ref: "transcripción humana", silence?: false, hasNyra?: true }

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { Whisper } from "../discord-voice.js";
import { WhisperRemoteClient } from "../lib/stt/whisper-remote-client.js";

function levenshtein(a, b) {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) => Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[m][n];
}

const norm = s => String(s || "").toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, "").replace(/\s+/g, " ").trim();
const wer = (ref, hyp) => {
  const r = norm(ref).split(" ").filter(Boolean);
  const h = norm(hyp).split(" ").filter(Boolean);
  return r.length ? levenshtein(r, h) / r.length : (h.length ? 1 : 0);
};
const cer = (ref, hyp) => {
  const r = [...norm(ref)];
  const h = [...norm(hyp)];
  return r.length ? levenshtein(r, h) / r.length : (h.length ? 1 : 0);
};
const pct = (arr, p) => {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
};

const iMan = process.argv.indexOf("--manifest");
const manifestPath = resolve(iMan >= 0 && process.argv[iMan + 1] ? process.argv[iMan + 1] : "voices/stt/manifest.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const voicesDir = resolve(manifestPath, "..");

function readPcmFromWav(filePath) {
  const buf = readFileSync(filePath);
  const idx = buf.indexOf(Buffer.from("data", "ascii"));
  if (idx < 0) throw new Error(`WAV sin chunk data: ${filePath}`);
  return buf.subarray(idx + 8);
}

const localWhisper = new Whisper();
await localWhisper.ready;
console.log("Local Whisper:", localWhisper.info);

async function evalMode(label, runFn) {
  const wers = [], cers = [], latencies = [];
  let silenceHalluc = 0, silenceTotal = 0, nyraOk = 0, nyraTotal = 0;
  for (const item of manifest) {
    const pcm = readPcmFromWav(join(voicesDir, item.file));
    const res = await runFn(pcm, item);
    latencies.push(res.ms);
    if (item.silence) {
      silenceTotal++;
      if (norm(res.text)) silenceHalluc++;
    } else {
      wers.push(wer(item.ref, res.text));
      cers.push(cer(item.ref, res.text));
    }
    if (item.hasNyra) {
      nyraTotal++;
      if (/\b(nyra|nira|ñira)\b/i.test(res.text)) nyraOk++;
    }
  }
  const avg = a => a.length ? (a.reduce((x, y) => x + y, 0) / a.length).toFixed(3) : "n/a";
  console.log(`${label}: WER=${avg(wers)} CER=${avg(cers)} NYRA=${nyraOk}/${nyraTotal} SilHalluc=${silenceHalluc}/${silenceTotal} p50=${pct(latencies, 50)}ms p95=${pct(latencies, 95)}ms`);
}

try {
  await evalMode("1. Local actual", pcm => localWhisper.transcribe(pcm));
  await evalMode("2. Local sin initial_prompt", pcm => localWhisper.transcribe(pcm, { noPrompt: true }));
  if (process.env.WHISPER_REMOTE_URL && process.env.WHISPER_REMOTE_API_KEY) {
    for (const model of ["medium", "large-v3-turbo", "large-v3"]) {
      const rem = new WhisperRemoteClient({
        config: { url: process.env.WHISPER_REMOTE_URL, apiKey: process.env.WHISPER_REMOTE_API_KEY, model },
        voicesDir,
        settingsFile: join(voicesDir, ".bench-stt.json"),
      });
      await evalMode(`3. Remoto ${model}`, pcm => rem.transcribePcm(pcm));
    }
  } else {
    console.log("Remoto omitido: falta WHISPER_REMOTE_URL / WHISPER_REMOTE_API_KEY en el entorno.");
  }
} finally {
  localWhisper.stop();
}
