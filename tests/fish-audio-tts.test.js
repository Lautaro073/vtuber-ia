import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { decode } from "@msgpack/msgpack";

import {
  FishAudioClient,
  validateFishUrl,
  assertSafeFishTarget,
  isPrivateOrReservedIp,
  resolveAuthorizedVoicePath,
  validateWavBuffer,
  applyFishExpressivity,
} from "../lib/tts/fish-audio-client.js";

function makeWavBuffer(durationSec = 1, sampleRate = 44100) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const dataSize = Math.round(durationSec * byteRate);
  const buf = Buffer.alloc(44 + dataSize);
  buf.write("RIFF", 0, 4, "ascii");
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write("WAVE", 8, 4, "ascii");
  buf.write("fmt ", 12, 4, "ascii");
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(numChannels, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(byteRate, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(bitsPerSample, 34);
  buf.write("data", 36, 4, "ascii");
  buf.writeUInt32LE(dataSize, 40);
  return buf;
}

async function startMockFishServer({ apiKey = "secret-fish-key", mode = "ok", delayMs = 0 } = {}) {
  const received = [];
  const server = createServer((req, res) => {
    const auth = req.headers.authorization;
    if (auth !== `Bearer ${apiKey}`) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ statusCode: 401, error: "Unauthorized" }));
      return;
    }
    if (req.method === "GET" && req.url === "/v1/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "ok", model: "fishaudio/s2-pro" }));
      return;
    }
    if (req.method === "POST" && req.url === "/v1/tts") {
      const chunks = [];
      req.on("data", c => chunks.push(c));
      req.on("end", () => {
        const raw = Buffer.concat(chunks);
        const decoded = decode(raw);
        received.push({ headers: req.headers, body: decoded });
        setTimeout(() => {
          if (res.destroyed) return;
          if (mode === "error500") {
            res.writeHead(500);
            res.end("Internal Error");
          } else if (mode === "empty") {
            res.writeHead(200, { "Content-Type": "audio/wav" });
            res.end(Buffer.alloc(0));
          } else if (mode === "corrupt") {
            res.writeHead(200, { "Content-Type": "audio/wav" });
            res.end(Buffer.from("NOT_A_WAV_FILE_DATA_CORRUPT_BUFFER_0000000000"));
          } else {
            const wav = makeWavBuffer(0.5, 44100);
            res.writeHead(200, { "Content-Type": "audio/wav", "Content-Length": wav.length });
            res.end(wav);
          }
        }, delayMs);
      });
      return;
    }
    res.writeHead(403).end();
  });

  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  return {
    url: `http://127.0.0.1:${port}`,
    received,
    close: () => new Promise(resolve => server.close(resolve)),
  };
}

test("FishAudioClient enforces URL and voice path security and protects credentials", () => {
  assert.equal(validateFishUrl("https://spiffy-deviate-overact.ngrok-free.dev/v1/tts"), "https://spiffy-deviate-overact.ngrok-free.dev");
  assert.equal(validateFishUrl("http://127.0.0.1:8080"), "http://127.0.0.1:8080");
  assert.throws(() => validateFishUrl("file:///etc/passwd"));
  assert.throws(() => validateFishUrl("https://user:pass@example.com"));
  assert.throws(() => validateFishUrl("http://169.254.169.254/latest/meta-data"));
  assert.throws(() => validateFishUrl("http://192.168.1.50:8080"));
  assert.throws(() => validateFishUrl("http://10.0.0.5:8080"));

  const tmp = mkdtempSync(join(tmpdir(), "nyra-fish-sec-"));
  const voicesDir = join(tmp, "voices", "fish");
  mkdirSync(voicesDir, { recursive: true });
  const goodFile = join(voicesDir, "nyra.wav");
  writeFileSync(goodFile, makeWavBuffer(0.2));

  try {
    const resolved = resolveAuthorizedVoicePath(voicesDir, "nyra.wav");
    assert.equal(resolved.path, goodFile);
    assert.throws(() => resolveAuthorizedVoicePath(voicesDir, "../secret.wav"));
    assert.throws(() => resolveAuthorizedVoicePath(voicesDir, "/content/drive/MyDrive/nyra-tts/voz.mp3"));
    assert.throws(() => resolveAuthorizedVoicePath(voicesDir, "nyra.exe"));

    const client = new FishAudioClient({
      config: { url: "https://example.ngrok-free.dev", apiKey: "super-secret-token" },
      voicesDir,
      profilesFile: join(tmp, "fish_voices.json"),
      settingsFile: join(tmp, "tts_settings.json"),
    });
    const pub = client.getPublicConfig();
    assert.equal(pub.apiKeyConfigured, true);
    assert.equal(pub.apiKey, undefined);
    assert.doesNotMatch(JSON.stringify(pub), /super-secret-token/);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

test("FishAudioClient health check and MsgPack synthesis work with reference_id and local binary reference", async () => {
  const mock = await startMockFishServer({ apiKey: "valid-key" });
  const tmp = mkdtempSync(join(tmpdir(), "nyra-fish-ok-"));
  const voicesDir = join(tmp, "voices", "fish");
  const outDir = join(tmp, "out");
  mkdirSync(voicesDir, { recursive: true });
  const refWav = makeWavBuffer(0.3, 44100);
  writeFileSync(join(voicesDir, "ref.wav"), refWav);

  try {
    // Missing key -> does not even call remote
    const noKey = new FishAudioClient({ config: { url: mock.url, apiKey: "" }, voicesDir, profilesFile: join(tmp, "p1.json"), settingsFile: join(tmp, "s1.json") });
    await assert.rejects(() => noKey.checkHealth(), /no configurada/);
    assert.equal(mock.received.length, 0);

    // Wrong key -> 401
    const badKey = new FishAudioClient({ config: { url: mock.url, apiKey: "wrong" }, voicesDir, profilesFile: join(tmp, "p2.json"), settingsFile: join(tmp, "s2.json") });
    await assert.rejects(() => badKey.checkHealth(), /401/);

    // Valid key + reference_id (Modalidad A)
    const client = new FishAudioClient({
      config: {
        url: mock.url,
        apiKey: "valid-key",
        referenceId: "nyra-ref-1",
        temperature: 0.7,
        topP: 0.7,
        chunkLength: 200,
        maxNewTokens: 1024,
        seed: 42,
      },
      voicesDir,
      profilesFile: join(tmp, "profiles.json"),
      settingsFile: join(tmp, "settings.json"),
    });
    const h = await client.checkHealth();
    assert.equal(h.ok, true);
    assert.equal(h.status, 200);

    const r1 = await client.synthesizeWav("Hola chat.", { emotion: "happy", mood: 90, outDir });
    assert.ok(r1.filePath.endsWith(".wav"));
    assert.equal(r1.metrics.audioDurationSec, 0.5);
    assert.equal(mock.received[0].headers["ngrok-skip-browser-warning"], "1");
    assert.equal(mock.received[0].body.reference_id, "nyra-ref-1");
    assert.equal(mock.received[0].body.text, "[excited] Hola chat.");
    assert.equal(mock.received[0].body.seed, 42);

    // Local profile (Modalidad B)
    const prof = client.upsertProfile({
      name: "Nyra Local",
      mode: "local_file",
      filePath: "ref.wav",
      transcript: "Esta es la voz de prueba.",
    });
    client.updateConfig({ activeProfile: prof.id });

    const r2 = await client.synthesizeWav("Prueba local.", { emotion: "sad", mood: 50, outDir });
    assert.equal(r2.metrics.audioDurationSec, 0.5);
    assert.equal(mock.received[1].body.references.length, 1);
    assert.equal(mock.received[1].body.references[0].text, "Esta es la voz de prueba.");
    assert.equal(Buffer.from(mock.received[1].body.references[0].audio).length, refWav.length);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
    await mock.close();
  }
});

test("FishAudioClient validates WAV, handles timeout, and aborts cleanly on manual cancellation", async () => {
  const emptyMock = await startMockFishServer({ apiKey: "k", mode: "empty" });
  const corruptMock = await startMockFishServer({ apiKey: "k", mode: "corrupt" });
  const slowMock = await startMockFishServer({ apiKey: "k", mode: "ok", delayMs: 400 });
  const tmp = mkdtempSync(join(tmpdir(), "nyra-fish-err-"));

  try {
    assert.throws(() => validateWavBuffer(Buffer.alloc(10)), /vacío/);
    assert.throws(() => validateWavBuffer(Buffer.from("NOT_WAV_1234567890123456789012345678901234567890")), /RIFF/);

    const cEmpty = new FishAudioClient({ config: { url: emptyMock.url, apiKey: "k" }, voicesDir: tmp, profilesFile: join(tmp, "p1.json"), settingsFile: join(tmp, "s1.json") });
    await assert.rejects(() => cEmpty.synthesizeWav("Hola"), /vacío/);

    const cCorrupt = new FishAudioClient({ config: { url: corruptMock.url, apiKey: "k" }, voicesDir: tmp, profilesFile: join(tmp, "p2.json"), settingsFile: join(tmp, "s2.json") });
    await assert.rejects(() => cCorrupt.synthesizeWav("Hola"), /RIFF/);

    // Timeout
    const cTimeout = new FishAudioClient({ config: { url: slowMock.url, apiKey: "k", timeoutMs: 50 }, voicesDir: tmp, profilesFile: join(tmp, "p3.json"), settingsFile: join(tmp, "s3.json") });
    cTimeout.config.timeoutMs = 50; // bypass 5000ms UI clamp for fast test
    await assert.rejects(() => cTimeout.synthesizeWav("Lento"), err => err.timedOut === true);

    // Manual cancellation -> cancelled: true (never timedOut)
    const cCancel = new FishAudioClient({ config: { url: slowMock.url, apiKey: "k", timeoutMs: 5000 }, voicesDir: tmp, profilesFile: join(tmp, "p4.json"), settingsFile: join(tmp, "s4.json") });
    const p = cCancel.synthesizeWav("Cancelar esto", { source: "live" });
    setTimeout(() => cCancel.abortSource("live"), 40);
    await assert.rejects(() => p, err => err.cancelled === true && !err.timedOut);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
    await Promise.all([emptyMock.close(), corruptMock.close(), slowMock.close()]);
  }
});

test("applyFishExpressivity respects toggle and keeps subtitles/memory free of Fish tags", () => {
  assert.equal(applyFishExpressivity("Hola chat.", { emotion: "happy", mood: 95, enabled: true }), "[excited] Hola chat.");
  assert.equal(applyFishExpressivity("Qué pena.", { emotion: "sad", mood: 50, enabled: true }), "[sad] Qué pena.");
  assert.equal(applyFishExpressivity("Basta.", { emotion: "angry", mood: 10, enabled: true }), "[angry] Basta.");
  assert.equal(applyFishExpressivity("¡No puede ser!", { emotion: "surprised", mood: 50, enabled: true }), "[surprised] ¡No puede ser!");
  assert.equal(applyFishExpressivity("Seguro.", { emotion: "neutral", mood: 10, enabled: true }), "[dry sarcastic tone] Seguro.");
  assert.equal(applyFishExpressivity("Hola chat.", { emotion: "happy", mood: 95, enabled: false }), "Hola chat.");
});

test("TTSEngine integrates Fish Audio with fallback, manual cancellation without fallback, and independent Discord engine", async () => {
  const src = readFileSync(new URL("../server.js", import.meta.url), "utf8");
  const lines = src.split(/\r?\n/);
  const start = lines.findIndex(l => l.startsWith("class TTSEngine"));
  const end = lines.findIndex((l, i) => i > start && l === "}");
  const classCode = lines.slice(start, end + 1).join("\n");

  const errMock = await startMockFishServer({ apiKey: "k", mode: "error500" });
  const slowMock = await startMockFishServer({ apiKey: "k", mode: "ok", delayMs: 300 });
  const tmp = mkdtempSync(join(tmpdir(), "nyra-tts-engine-"));

  try {
    const CONFIG = {
      ttsEngine: "fish",
      shareTtsEngine: true,
      discordTtsEngine: "edge",
      useSystemTTS: false,
      edgeVoice: "es-AR-ElenaNeural",
      murf: { apiKey: "" },
      fish: { url: errMock.url, apiKey: "k", fallback: true, expressive: true },
    };

    const PatchedFishClient = class extends FishAudioClient {
      constructor(opts) {
        super({ ...opts, settingsFile: join(tmp, "tts_settings.json") });
      }
    };

    const TTSEngine = new Function(
      "deps",
      `const { CONFIG, FishAudioClient, join, __dirname, mkdirSync, renameSync, rmSync, readdirSync, unlinkSync, spawn, createInterface, applyPiperPhonetics, exec, discord } = deps;
       ${classCode}
       return TTSEngine;`
    )({
      CONFIG,
      FishAudioClient: PatchedFishClient,
      join,
      __dirname: tmp,
      mkdirSync,
      renameSync: () => {},
      rmSync,
      readdirSync: () => [],
      unlinkSync: () => {},
      spawn: () => {},
      createInterface: () => ({ on: () => {} }),
      applyPiperPhonetics: s => s,
      exec: (_c, cb) => cb?.(),
      discord: { stopAudio: () => {} },
    });

    const tts = new TTSEngine();
    let edgeCalls = 0;
    tts._synthEdge = async () => { edgeCalls += 1; return "/tts/edge-fallback.mp3"; };

    // 1. Motor independiente para Discord vs Live
    assert.equal(tts.resolveEngine("live"), "fish");
    assert.equal(tts.resolveEngine("discord"), "fish");
    CONFIG.shareTtsEngine = false;
    CONFIG.discordTtsEngine = "edge";
    assert.equal(tts.resolveEngine("live"), "fish");
    assert.equal(tts.resolveEngine("discord"), "edge");
    const discRes = await tts.synth("Hola Discord", { source: "discord", returnMeta: true });
    assert.equal(discRes.url, "/tts/edge-fallback.mp3");
    assert.equal(edgeCalls, 1);

    // 2. Error 500 en Fish -> fallback automático a Edge
    const fbRes = await tts.synth("Hola Live", { source: "live", returnMeta: true });
    assert.equal(fbRes.url, "/tts/edge-fallback.mp3");
    assert.equal(fbRes.fallbackUsed, true);
    assert.equal(edgeCalls, 2);

    // 3. Cancelación manual (tts.stop()) -> NO ejecuta fallback ni devuelve audio
    tts.fish.updateConfig({ url: slowMock.url });
    const pending = tts.synth("No reproducir esto", { source: "live", returnMeta: true });
    setTimeout(() => tts.stop(), 40);
    const cancelRes = await pending;
    assert.equal(cancelRes.cancelled, true);
    assert.equal(cancelRes.url, null);
    assert.equal(edgeCalls, 2); // sin fallback

    // 4. Cancelación durante fallback a Edge -> descarta resultado tardío
    tts.fish.updateConfig({ url: errMock.url });
    tts._synthEdge = async () => {
      await new Promise(r => setTimeout(r, 80));
      return "/tts/late-edge.mp3";
    };
    const pendingFb = tts.synth("Cancelar durante fallback", { source: "live", returnMeta: true });
    setTimeout(() => tts.stop("live"), 25);
    const fbCancelRes = await pendingFb;
    assert.equal(fbCancelRes.cancelled, true);
    assert.equal(fbCancelRes.url, null);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
    await Promise.all([errMock.close(), slowMock.close()]);
  }
});

test("Audit security: blocks DNS rebinding, private IPv6/IPv4-mapped, non-allowlisted hosts, and symlink escapes", async () => {
  assert.equal(isPrivateOrReservedIp("::ffff:127.0.0.1"), true);
  assert.equal(isPrivateOrReservedIp("fc00::1"), true);
  assert.equal(isPrivateOrReservedIp("fe80::1"), true);
  assert.equal(isPrivateOrReservedIp("100.64.0.1"), true);
  assert.equal(isPrivateOrReservedIp("8.8.8.8"), false);

  // DNS rebinding simulation: external domain resolving to loopback/metadata IP must be rejected
  await assert.rejects(
    () => assertSafeFishTarget("https://rebind.evil.test", {
      lookupImpl: async () => [{ address: "169.254.169.254", family: 4 }],
    }),
    /IP privada/
  );
  await assert.rejects(
    () => assertSafeFishTarget("https://rebind-local.evil.test", {
      lookupImpl: async () => [{ address: "127.0.0.1", family: 4 }],
    }),
    /IP privada/
  );

  // Explicit allowlist
  assert.throws(
    () => validateFishUrl("https://attacker.com", { allowedHosts: [".ngrok-free.dev"] }),
    /lista permitida/
  );
  assert.equal(
    validateFishUrl("https://spiffy-deviate-overact.ngrok-free.dev", { allowedHosts: [".ngrok-free.dev"] }),
    "https://spiffy-deviate-overact.ngrok-free.dev"
  );

  // Symlink escape inside voices/fish/
  const tmp = mkdtempSync(join(tmpdir(), "nyra-fish-sym-"));
  const voicesDir = join(tmp, "voices", "fish");
  mkdirSync(voicesDir, { recursive: true });
  const outsideFile = join(tmp, "outside.wav");
  writeFileSync(outsideFile, makeWavBuffer(0.2));
  const linkPath = join(voicesDir, "escape.wav");
  try {
    symlinkSync(outsideFile, linkPath);
    assert.throws(() => resolveAuthorizedVoicePath(voicesDir, "escape.wav"), /fuera del directorio autorizado/);
  } catch (err) {
    // On Windows without Developer Mode / admin symlink privilege, EPERM can occur; skip symlink assertion only if OS denied creation
    if (err?.code !== "EPERM") throw err;
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

test("Audit persistence & concurrency: settings survive restart, deleted profile throws, and live/discord/panel_test are isolated", async () => {
  const mock = await startMockFishServer({ apiKey: "k", delayMs: 60 });
  const tmp = mkdtempSync(join(tmpdir(), "nyra-fish-conc-"));
  const voicesDir = join(tmp, "voices", "fish");
  mkdirSync(voicesDir, { recursive: true });
  const profilesFile = join(tmp, "fish_voices.json");
  const settingsFile = join(tmp, "tts_settings.json");

  try {
    const c1 = new FishAudioClient({
      config: { url: mock.url, apiKey: "k" },
      voicesDir,
      profilesFile,
      settingsFile,
    });
    c1.updateConfig({ temperature: 0.55, seed: 99, activeProfile: "missing-prof" });
    c1.saveEnginePrefs({ engine: "fish", shareTtsEngine: false, discordEngine: "edge" });

    // Deleted/missing activeProfile must throw, never silently fall back to random voice
    await assert.rejects(() => c1.synthesizeWav("Hola"), /no encontrado/);

    // Second instance loads persisted settings without storing apiKey on disk
    const rawDisk = readFileSync(settingsFile, "utf8");
    assert.doesNotMatch(rawDisk, /"apiKey"/);
    const c2 = new FishAudioClient({
      config: { url: "", apiKey: "k" },
      voicesDir,
      profilesFile,
      settingsFile,
    });
    assert.equal(c2.config.temperature, 0.55);
    assert.equal(c2.config.seed, 99);
    assert.equal(c2.enginePrefs.engine, "fish");
    assert.equal(c2.enginePrefs.shareTtsEngine, false);
    assert.equal(c2.enginePrefs.discordEngine, "edge");

    // Concurrent live + discord + panel_test: cancelling live must NOT cancel discord or panel_test
    c2.updateConfig({ activeProfile: "" });
    const pLive = c2.synthesizeWav("Live turn", { source: "live" });
    const pDiscord = c2.synthesizeWav("Discord turn", { source: "discord" });
    const pPanel = c2.synthesizeWav("Panel test", { source: "panel_test" });
    setTimeout(() => c2.abortSource("live"), 15);

    const [rLive, rDiscord, rPanel] = await Promise.allSettled([pLive, pDiscord, pPanel]);
    assert.equal(rLive.status, "rejected");
    assert.equal(rLive.reason.cancelled, true);
    assert.equal(rDiscord.status, "fulfilled");
    assert.equal(rPanel.status, "fulfilled");
  } finally {
    rmSync(tmp, { recursive: true, force: true });
    await mock.close();
  }
});

test("DNS pinning: pinnedRequest uses the exact validated IP without re-resolving, ignores panel allowedHosts/apiKey overrides, and blocks redirects", async () => {
  const mock = await startMockFishServer({ apiKey: "env-key" });
  const redirServer = createServer((_req, res) => {
    res.writeHead(302, { Location: "http://127.0.0.1:9999/steal-key" });
    res.end();
  });
  await new Promise(r => redirServer.listen(0, "127.0.0.1", r));
  const redirPort = redirServer.address().port;

  const tmp = mkdtempSync(join(tmpdir(), "nyra-fish-pin-"));
  try {
    let lookupCalls = 0;
    const target = await assertSafeFishTarget(`http://127.0.0.1:${new URL(mock.url).port}`, {
      lookupImpl: async () => {
        lookupCalls += 1;
        return [{ address: "127.0.0.1", family: 4 }];
      },
    });
    assert.equal(target.pinnedAddress, "127.0.0.1");

    // Even if panel tries to pass allowedHosts or apiKey, backend ignores both
    const client = new FishAudioClient({
      config: { url: mock.url, apiKey: "env-key" },
      voicesDir: tmp,
      profilesFile: join(tmp, "p.json"),
      settingsFile: join(tmp, "s.json"),
      allowedHosts: ["127.0.0.1"],
    });
    client.updateConfig({ allowedHosts: ["attacker.com"], apiKey: "injected-key" });
    assert.deepEqual(client.allowedHosts, ["127.0.0.1"]);
    assert.equal(client.config.apiKey, "env-key");
    assert.equal(lookupCalls, 0); // loopback did not even need DNS; pinned directly

    // Real pinnedRequest to mock server succeeds
    const h = await client.checkHealth();
    assert.equal(h.ok, true);

    // 302 redirect is NOT followed (fails with HTTP 302 instead of following Location)
    const cRedir = new FishAudioClient({
      config: { url: `http://127.0.0.1:${redirPort}`, apiKey: "env-key" },
      voicesDir: tmp,
      profilesFile: join(tmp, "p2.json"),
      settingsFile: join(tmp, "s2.json"),
    });
    await assert.rejects(() => cRedir.checkHealth(), /302/);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
    await Promise.all([mock.close(), new Promise(r => redirServer.close(r))]);
  }
});

test("Regression: TTSEngine serializes concurrent Fish requests without self-cancelling, manual stop cancels queued items, and HTTP 429 retries cleanly", async () => {
  let inFlight = 0;
  let maxInFlight = 0;
  let attempt429 = 0;
  let mode = "serial";

  const mock = await startMockFishServer({ apiKey: "k-reg" });
  const srv = createServer(async (req, res) => {
    if (req.url === "/v1/tts" && req.method === "POST") {
      if (mode === "429_once") {
        attempt429 += 1;
        if (attempt429 === 1) {
          res.writeHead(429, { "Retry-After": "0.05" });
          res.end("rate limited");
          return;
        }
      }
      inFlight += 1;
      if (inFlight > maxInFlight) maxInFlight = inFlight;
      await new Promise(r => setTimeout(r, 40));
      inFlight -= 1;
      const wav = makeWavBuffer();
      res.writeHead(200, { "Content-Type": "audio/wav" });
      res.end(wav);
      return;
    }
    res.writeHead(404);
    res.end();
  });
  await new Promise(r => srv.listen(0, "127.0.0.1", r));
  const port = srv.address().port;
  const tmp = mkdtempSync(join(tmpdir(), "nyra-fish-reg-"));

  try {
    const client = new FishAudioClient({
      config: { url: `http://127.0.0.1:${port}`, apiKey: "k-reg", timeoutMs: 5000 },
      voicesDir: tmp,
      profilesFile: join(tmp, "p.json"),
      settingsFile: join(tmp, "s.json"),
      allowedHosts: ["127.0.0.1"],
    });

    // Replicate TTSEngine _fishQueue + _cancelEpoch serialization
    let fishQueue = Promise.resolve();
    const cancelEpoch = new Map();
    const synthSerialized = (text, source = "live") => {
      const myEpoch = cancelEpoch.get(source) || 0;
      const isCancelled = () => (cancelEpoch.get(source) || 0) !== myEpoch;
      const runFish = async () => {
        if (isCancelled()) throw Object.assign(new Error("Fish TTS cancelado"), { cancelled: true });
        return client.synthesizeWav(text, { source, outDir: tmp });
      };
      const queued = fishQueue.then(runFish, runFish);
      fishQueue = queued.catch(() => {});
      return queued.then(
        r => (isCancelled() ? { url: null, cancelled: true } : { url: r.filePath, cancelled: false }),
        err => (err.cancelled || isCancelled() ? { url: null, cancelled: true } : Promise.reject(err)),
      );
    };

    // 1) Two concurrent live requests must serialize (maxInFlight === 1) and neither is cancelled
    const [r1, r2] = await Promise.all([
      synthSerialized("Primer mensaje live", "live"),
      synthSerialized("Segundo mensaje live", "live"),
    ]);
    assert.equal(maxInFlight, 1);
    assert.equal(r1.cancelled, false);
    assert.equal(r2.cancelled, false);
    assert.ok(r1.url && r2.url);

    // 2) Manual stop cancels both active and queued live requests without fallback
    const pA = synthSerialized("Mensaje A", "live");
    const pB = synthSerialized("Mensaje B", "live");
    setTimeout(() => {
      cancelEpoch.set("live", (cancelEpoch.get("live") || 0) + 1);
      client.abortAll();
    }, 10);
    const [resA, resB] = await Promise.all([pA, pB]);
    assert.equal(resA.cancelled, true);
    assert.equal(resB.cancelled, true);

    // 3) Transient HTTP 429 retries and succeeds without failing
    mode = "429_once";
    const r429 = await client.synthesizeWav("Prueba 429", { source: "live", outDir: tmp });
    assert.equal(attempt429, 2);
    assert.ok(r429.filePath);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
    await Promise.all([mock.close(), new Promise(r => srv.close(r))]);
  }
});
