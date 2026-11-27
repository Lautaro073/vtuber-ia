import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  WhisperRemoteClient,
  pcmToWavBuffer,
  buildMultipartFormData,
} from "../lib/stt/whisper-remote-client.js";
import { FishAudioClient } from "../lib/tts/fish-audio-client.js";
import { DiscordVoice } from "../discord-voice.js";

async function startMockGateway({ sttKey = "stt-secret", fishKey = "fish-secret", mode = "ok", delayMs = 0, busyCount = 0 } = {}) {
  const sttRequests = [];
  let remainingBusy = busyCount;
  const server = createServer((req, res) => {
    const auth = req.headers.authorization;
    if (req.url === "/v1/stt/health") {
      if (auth !== `Bearer ${sttKey}`) return res.writeHead(401).end(JSON.stringify({ error: "Unauthorized" }));
      return res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({
        status: "ok", model: "large-v3-turbo", device: "cpu", compute_type: "int8",
      }));
    }
    if (req.url === "/v1/audio/transcriptions" && req.method === "POST") {
      if (auth !== `Bearer ${sttKey}`) return res.writeHead(401).end(JSON.stringify({ error: "Unauthorized" }));
      const chunks = [];
      req.on("data", c => chunks.push(c));
      req.on("end", () => {
        const raw = Buffer.concat(chunks);
        sttRequests.push({ headers: req.headers, bodyStr: raw.toString("latin1"), size: raw.length });
        setTimeout(() => {
          if (res.destroyed) return;
          if (remainingBusy > 0) {
            remainingBusy--;
            return res.writeHead(429).end(JSON.stringify({ error: "Busy" }));
          }
          if (mode === "error500") return res.writeHead(500).end("Error");
          if (mode === "silence") {
            return res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({
              text: "", language: "es", duration: 0.5, segments: [], model: "large-v3-turbo",
              metrics: { decode_ms: 4, inference_ms: 40, total_ms: 44, rtf: 0.08 },
            }));
          }
          res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({
            text: "Che Nyra, tirá la ulti en dragón.",
            language: "es",
            duration: 1.5,
            segments: [{ text: "Che Nyra, tirá la ulti en dragón.", avg_logprob: -0.25 }],
            model: "large-v3-turbo",
            metrics: { decode_ms: 10, inference_ms: 350, total_ms: 360, rtf: 0.24 },
          }));
        }, delayMs);
      });
      return;
    }
    if (req.url === "/v1/tts" && req.method === "POST") {
      if (auth !== `Bearer ${fishKey}`) return res.writeHead(401).end(JSON.stringify({ error: "Unauthorized" }));
      if (mode === "fish422") return res.writeHead(422).end(JSON.stringify({ error: "Invalid reference_id" }));
      return res.writeHead(500).end();
    }
    res.writeHead(404).end();
  });
  await new Promise(r => server.listen(0, "127.0.0.1", r));
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    sttRequests,
    close: () => new Promise(r => server.close(r)),
  };
}

test("WhisperRemoteClient health, multipart WAV 16kHz upload, hotwords, and independent auth from Fish Audio", async () => {
  const gw = await startMockGateway({ sttKey: "stt-key", fishKey: "fish-key" });
  const tmp = mkdtempSync(join(tmpdir(), "nyra-stt-ok-"));

  try {
    // Wrong STT key (even if it's Fish key) must fail 401
    const bad = new WhisperRemoteClient({
      config: { url: gw.url, apiKey: "fish-key" },
      voicesDir: tmp,
      settingsFile: join(tmp, "s1.json"),
    });
    await assert.rejects(() => bad.checkHealth(), /401/);

    const client = new WhisperRemoteClient({
      config: {
        url: gw.url,
        apiKey: "stt-key",
        model: "large-v3-turbo",
        hotwords: "Nyra, Wild Rift, Dragón",
        beamSize: 3,
      },
      voicesDir: tmp,
      settingsFile: join(tmp, "s2.json"),
    });

    const h = await client.checkHealth();
    assert.equal(h.ok, true);
    assert.equal(client.metrics.effectiveModel, "large-v3-turbo");
    assert.equal(client.metrics.effectiveDevice, "cpu");
    assert.equal(client.metrics.effectiveQuantization, "int8");

    const pcm = Buffer.alloc(16000 * 2); // 1s 16kHz mono s16le
    const wav = pcmToWavBuffer(pcm);
    assert.equal(wav.toString("ascii", 0, 4), "RIFF");
    assert.equal(wav.readUInt32LE(24), 16000);
    assert.equal(wav.readUInt16LE(22), 1);

    const res = await client.transcribePcm(pcm, { speakerId: "u1" });
    assert.equal(res.text, "Che Nyra, tirá la ulti en dragón.");
    assert.equal(gw.sttRequests.length, 1);
    assert.match(gw.sttRequests[0].bodyStr, /name="hotwords"/);
    assert.match(gw.sttRequests[0].bodyStr, /Nyra, Wild Rift/);
    assert.match(gw.sttRequests[0].bodyStr, /RIFF/);
    client.updateConfig({ beamSize: 4 });
    assert.doesNotMatch(readFileSync(join(tmp, "s2.json"), "utf8"), /"apiKey"/);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
    await gw.close();
  }
});

test("WhisperRemoteClient handles HTTP 429 retry, empty silence without fallback, stale discard, and manual cancellation", async () => {
  const busyGw = await startMockGateway({ sttKey: "k", busyCount: 1 });
  const silenceGw = await startMockGateway({ sttKey: "k", mode: "silence" });
  const slowGw = await startMockGateway({ sttKey: "k", delayMs: 250 });
  const tmp = mkdtempSync(join(tmpdir(), "nyra-stt-edge-"));

  try {
    // 1. Single bounded retry on HTTP 429 succeeds
    const cBusy = new WhisperRemoteClient({ config: { url: busyGw.url, apiKey: "k", maxAgeMs: 5000 }, voicesDir: tmp, settingsFile: join(tmp, "b.json") });
    const rBusy = await cBusy.transcribePcm(Buffer.alloc(16000), { speakerId: "u1" });
    assert.equal(rBusy.text, "Che Nyra, tirá la ulti en dragón.");
    assert.equal(busyGw.sttRequests.length, 2);

    // 2. Stale fragment expired before queue is discarded without calling remote
    await assert.rejects(
      () => cBusy.transcribePcm(Buffer.alloc(16000), { speakerId: "u1", capturedAt: Date.now() - 10000 }),
      err => err.stale === true
    );
    assert.equal(cBusy.metrics.discardedCount, 1);
    assert.equal(cBusy.metrics.lastDiscardReason, "stale_before_queue");

    // 3. Empty silence (200 OK with text="") does NOT trigger local fallback in DiscordVoice
    const cSil = new WhisperRemoteClient({ config: { url: silenceGw.url, apiKey: "k", engine: "remote", fallbackLocal: true }, voicesDir: tmp, settingsFile: join(tmp, "sil.json") });
    let localCalls = 0;
    const dv = new DiscordVoice({ synth: async () => null, onSpeech: async () => {}, whisperRemote: cSil });
    dv.whisper = { transcribe: async () => { localCalls++; return { text: "no debe correr" }; }, info: {} };
    const silOut = await dv._transcribePhrase(Buffer.alloc(16000), { uid: "u1", capturedAt: Date.now(), secs: 0.5, run: 0 });
    assert.equal(silOut.text, "");
    assert.equal(localCalls, 0);

    // 4. Manual cancellation aborts without local fallback
    const cCancel = new WhisperRemoteClient({ config: { url: slowGw.url, apiKey: "k", engine: "remote", fallbackLocal: true }, voicesDir: tmp, settingsFile: join(tmp, "c.json") });
    const dv2 = new DiscordVoice({ synth: async () => null, onSpeech: async () => {}, whisperRemote: cCancel });
    dv2.whisper = { transcribe: async () => { localCalls++; return { text: "no debe correr" }; }, info: {} };
    const pCancel = dv2._transcribePhrase(Buffer.alloc(16000), { uid: "u1", capturedAt: Date.now(), secs: 0.5, run: 0 });
    setTimeout(() => cCancel.abortAll(), 30);
    const cancelOut = await pCancel;
    assert.equal(cancelOut.text, "");
    assert.equal(localCalls, 0);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
    await Promise.all([busyGw.close(), silenceGw.close(), slowGw.close()]);
  }
});

test("FishAudioClient rejects HTTP 422 invalid reference_id explicitly without silent random voice", async () => {
  const gw = await startMockGateway({ fishKey: "fish-secret", mode: "fish422" });
  const tmp = mkdtempSync(join(tmpdir(), "nyra-fish-422-"));
  try {
    const fish = new FishAudioClient({
      config: { url: gw.url, apiKey: "fish-secret", referenceId: "nonexistent-id" },
      voicesDir: tmp,
      profilesFile: join(tmp, "fv.json"),
      settingsFile: join(tmp, "ts.json"),
    });
    await assert.rejects(() => fish.synthesizeWav("Hola"), err => err.invalidReference === true && err.status === 422);
    assert.equal(fish.metrics.status, "invalid_reference");
  } finally {
    rmSync(tmp, { recursive: true, force: true });
    await gw.close();
  }
});

test("Regression: empty stt_settings.json fields do not shadow .env, Whisper errors name WHISPER_REMOTE_URL, and STT/TTS work independently", async () => {
  const gw = await startMockGateway({ sttKey: "stt-only-key", fishKey: "fish-only-key" });
  const tmp = mkdtempSync(join(tmpdir(), "nyra-stt-indep-"));
  const sttSettings = join(tmp, "stt_settings.json");
  const ttsSettings = join(tmp, "tts_settings.json");

  try {
    // Empty strings in stt_settings.json must NOT overwrite valid .env values
    writeFileSync(sttSettings, JSON.stringify({ engine: "remote", url: "", hotwords: "" }));
    const sttFromEnv = new WhisperRemoteClient({
      config: { url: gw.url, apiKey: "stt-only-key", hotwords: "Nyra, Wild Rift" },
      voicesDir: tmp,
      settingsFile: sttSettings,
    });
    assert.equal(sttFromEnv.config.url, gw.url);
    assert.equal(sttFromEnv.config.hotwords, "Nyra, Wild Rift");

    // Missing WHISPER_REMOTE_URL must report WHISPER_REMOTE_URL (never FISH_TTS_URL)
    const noUrl = new WhisperRemoteClient({
      config: { url: "", apiKey: "stt-only-key" },
      voicesDir: tmp,
      settingsFile: join(tmp, "empty.json"),
    });
    await assert.rejects(() => noUrl.checkHealth(), /WHISPER_REMOTE_URL no configurada/);

    // Whisper works with Fish completely unconfigured
    const fishUnconfigured = new FishAudioClient({
      config: { url: "", apiKey: "" },
      voicesDir: tmp,
      profilesFile: join(tmp, "fv.json"),
      settingsFile: ttsSettings,
    });
    assert.equal(fishUnconfigured.config.url, "");
    const hStt = await sttFromEnv.checkHealth();
    assert.equal(hStt.ok, true);

    // Fish works with Whisper completely unconfigured
    const sttUnconfigured = new WhisperRemoteClient({
      config: { url: "", apiKey: "" },
      voicesDir: tmp,
      settingsFile: join(tmp, "stt_empty.json"),
    });
    assert.equal(sttUnconfigured.config.url, "");
    const fishConfigured = new FishAudioClient({
      config: { url: gw.url, apiKey: "fish-only-key" },
      voicesDir: tmp,
      profilesFile: join(tmp, "fv2.json"),
      settingsFile: join(tmp, "ts2.json"),
    });
    await assert.rejects(() => fishConfigured.synthesizeWav("Hola"), /422|500/); // reached Fish endpoint with Fish key
  } finally {
    rmSync(tmp, { recursive: true, force: true });
    await gw.close();
  }
});
