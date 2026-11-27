import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import OpusScript from "opusscript";
import { readPhrase, DiscordVoice, Whisper, LOCAL_LEGACY_DEFAULTS } from "../discord-voice.js";
import { resolveRecorderPort, createRecorderApp } from "../scripts/corpus-recorder.mjs";
import { norm, normNyra, werOps } from "../scripts/benchmark-stt.mjs";

test("readPhrase respects pauseMs parameter for intra-phrase silence gaps", async () => {
  const enc = new OpusScript(16000, 1);
  try {
    const pcmFrame = Buffer.alloc(320 * 2);
    const opus = Buffer.from(enc.encode(pcmFrame, 320));
    async function* stream() {
      const p1 = Buffer.from(opus);
      p1.ts = 1000;
      yield p1;
      const p2 = Buffer.from(opus);
      p2.ts = 1000 + 960 + 500 * 48; // 500ms gap
      yield p2;
    }
    const r400 = await readPhrase(stream(), 400);
    const r800 = await readPhrase(stream(), 800);
    assert.ok(r800.pcm.length > r400.pcm.length, "800ms pause cap should keep more silence than 400ms cap");
    assert.equal(typeof r800.segmentationMs, "number");
  } finally {
    enc.delete();
  }
});

test("Whisper.transcribe differentiates queue_full from valid silence without breaking empty text contract", async () => {
  const fakeWhisper = Object.create(Whisper.prototype);
  fakeWhisper.pending = new Map();
  fakeWhisper.id = 0;
  for (let i = 1; i <= 10; i++) fakeWhisper.pending.set(i, { resolve: () => {}, reject: () => {} });

  const res = await fakeWhisper.transcribe(Buffer.alloc(32000));
  assert.equal(res.text, "");
  assert.equal(res.status, "queue_full");
  assert.equal(res.discard_reason, "queue_full");
  assert.equal(res.pending_count, 10);
});

test("DiscordVoice._transcribePhrase distinguishes valid_silence, queue_full, and inference_error", async () => {
  const dv = new DiscordVoice({ synth: async () => null, onSpeech: async () => {} });
  dv.run = 1;
  dv.sessionId = "sess-1";
  dv.whisper = {
    info: { requestedModel: "large-v3-turbo", model: "large-v3-turbo", device: "cuda", computeType: "int8_float16" },
    transcribe: async () => ({ text: "", raw: "", ms: 120, discard_reason: "vad_no_speech", queue_ms: 1, pending_count: 0 }),
  };

  const sil = await dv._transcribePhrase(Buffer.alloc(32000), { uid: "u1", capturedAt: Date.now(), secs: 1, run: 1, fragmentId: "f1", segmentationMs: 10 });
  assert.equal(sil.text, "");
  assert.equal(sil.meta.discardReason, "valid_silence");
  assert.equal(sil.meta.effectiveModel, "large-v3-turbo");

  dv.whisper.transcribe = async () => ({ text: "", ms: 0, status: "queue_full", discard_reason: "queue_full", pending_count: 10 });
  const qf = await dv._transcribePhrase(Buffer.alloc(32000), { uid: "u1", capturedAt: Date.now(), secs: 1, run: 1, fragmentId: "f2", segmentationMs: 10 });
  assert.equal(qf.text, "");
  assert.equal(qf.meta.discardReason, "queue_full");

  dv.whisper.transcribe = async () => { throw Object.assign(new Error("boom"), { status: "worker_error" }); };
  const errRes = await dv._transcribePhrase(Buffer.alloc(32000), { uid: "u1", capturedAt: Date.now(), secs: 1, run: 1, fragmentId: "f3", segmentationMs: 10 });
  assert.equal(errRes.text, "");
  assert.equal(errRes.meta.discardReason, "worker_error");
});

test("whisper-worker.py baseline A and experimental profiles B-H resolve expected parameters and phantom filters", () => {
  const pyCode = `
import json, re
from pathlib import Path
src = Path("scripts/whisper-worker.py").read_text(encoding="utf-8")
# Extract pure functions/constants before model load
head = src.split("sys.stdout.reconfigure")[0]
ns = {"__file__": "scripts/whisper-worker.py"}
exec(head, ns)
assert ns["PROFILES"]["A"] == {"prompt_mode": "narrative", "hotwords": None, "vad_filter": True, "vad_threshold": 0.8, "beam_size": 5, "temperature": 0.0, "second_pass_mode": "always_only_name", "filter_profile": "legacy"}
pB = ns["resolve_effective_params"]({"profile": "B"})
assert pB["use_prompt"] is False and pB["hotwords"] is None and pB["vad_threshold"] == 0.8
pC = ns["resolve_effective_params"]({"profile": "C"})
assert pC["use_prompt"] is False and "Nyra" in pC["hotwords"]
pE = ns["resolve_effective_params"]({"profile": "E"})
assert pE["vad_threshold"] == 0.5 and pE["beam_size"] == 5
pH = ns["resolve_effective_params"]({"profile": "H", "vad_threshold": 0.65})
assert pH["beam_size"] == 1 and pH["vad_threshold"] == 0.65
pG = ns["resolve_effective_params"]({"profile": "G"})
assert pG["beam_size"] == 3 and pG["vad_threshold"] == 0.8
assert ns["phantom_reason"]("Charla de amigos argentinos") == "phantom_recited"
assert ns["phantom_reason"]("Gracias por ver el video.") == "phantom_ghost"
assert ns["phantom_reason"]("Che Nira vení al dragón") is None
print("PY_OK")
`;
  const out = execFileSync(".whisper/venv/Scripts/python.exe", ["-c", pyCode], { encoding: "utf8" }).trim();
  assert.equal(out, "PY_OK");
});

test("prompts-40.json has 40 scenarios, 28 tune / 12 eval, exact category distribution, and unique IDs", () => {
  const prompts = JSON.parse(readFileSync("docs/stt-corpus/prompts-40.json", "utf8"));
  assert.equal(prompts.length, 40);
  assert.equal(new Set(prompts.map(p => p.id)).size, 40);
  assert.equal(prompts.filter(p => p.split === "tune").length, 28);
  assert.equal(prompts.filter(p => p.split === "eval").length, 12);
  const byCat = {};
  for (const p of prompts) byCat[p.category] = (byCat[p.category] || 0) + 1;
  assert.deepEqual(byCat, {
    llamada_nyra: 8,
    similar_no_nyra: 6,
    charla_argentina: 6,
    videojuegos: 5,
    programacion_tech: 4,
    intervencion_corta: 4,
    pausas_naturales: 3,
    sin_habla: 4,
  });
});

test("corpus-recorder refuses port 3000 and serves /api/corpus on isolated port", async () => {
  assert.throws(() => resolveRecorderPort(["node", "corpus-recorder.mjs", "--port", "3000"]), /3000/);
  assert.equal(resolveRecorderPort(["node", "corpus-recorder.mjs"]), 3456);
  const app = createRecorderApp();
  const srv = await new Promise(resolve => {
    const s = app.listen(0, "127.0.0.1", () => resolve(s));
  });
  try {
    const { port } = srv.address();
    const res = await fetch(`http://127.0.0.1:${port}/api/corpus`).then(r => r.json());
    assert.equal(res.ok, true);
    assert.equal(res.prompts.length, 40);
  } finally {
    await new Promise(resolve => srv.close(resolve));
  }
});

test("confidence_ghost recovers high-confidence speech while still discarding low-confidence or high-no-speech segments", () => {
  const pyCode = `
from pathlib import Path
from types import SimpleNamespace
src = Path("scripts/whisper-worker.py").read_text(encoding="utf-8")
head = src.split("sys.stdout.reconfigure")[0]
ns = {"__file__": "scripts/whisper-worker.py"}
exec(head, ns)
pr = ns["phantom_reason"]

hi = [SimpleNamespace(avg_logprob=-0.25, no_speech_prob=0.05)]
lo = [SimpleNamespace(avg_logprob=-0.85, no_speech_prob=0.05)]
mixed = [SimpleNamespace(avg_logprob=-0.20, no_speech_prob=0.05), SimpleNamespace(avg_logprob=-0.75, no_speech_prob=0.10)]
nospeech = [SimpleNamespace(avg_logprob=-0.20, no_speech_prob=0.35)]

# Legacy drops all GHOST matches
assert pr("Chao.", "legacy", hi) == "phantom_ghost"
assert pr("Gracias por ver el video pibes", "legacy", hi) == "phantom_ghost"

# confidence_ghost keeps high-confidence speech
assert pr("Chao.", "confidence_ghost", hi) is None
assert pr("Gracias por ver el video pibes", "confidence_ghost", hi) is None
assert pr("Suscribite al canal.", "confidence_ghost", hi) is None

# confidence_ghost still discards weak or mixed-weak segments
assert pr("Gracias por ver el video.", "confidence_ghost", lo) == "phantom_ghost"
assert pr("Gracias por ver el video.", "confidence_ghost", mixed) == "phantom_ghost"
assert pr("Gracias.", "confidence_ghost", nospeech) == "phantom_ghost"
print("GHOST_OK")
`;
  const out = execFileSync(".whisper/venv/Scripts/python.exe", ["-c", pyCode], { encoding: "utf8" }).trim();
  assert.equal(out, "GHOST_OK");
});

test("DiscordVoice localConfig persists independently, validates ranges, and resets cleanly to Legacy", () => {
  const tmp = mkdtempSync(join(tmpdir(), "nyra-local-stt-"));
  try {
    const file = join(tmp, "whisper_local_settings.json");
    const dv = new DiscordVoice({ synth: async () => null, onSpeech: async () => {}, localSettingsFile: file });
    assert.deepEqual(dv.localConfig, LOCAL_LEGACY_DEFAULTS);

    dv.updateLocalConfig({ profile: "experimental", filterProfile: "confidence_ghost", promptMode: "minimal", secondPassMode: "none", pauseMs: 900, beamSize: 5 });
    assert.equal(dv.localConfig.profile, "experimental");
    assert.equal(dv.localConfig.filterProfile, "confidence_ghost");
    assert.equal(dv.localConfig.pauseMs, 900);

    // Reload survives restart
    const dv2 = new DiscordVoice({ synth: async () => null, onSpeech: async () => {}, localSettingsFile: file });
    assert.equal(dv2.localConfig.profile, "experimental");
    assert.equal(dv2.localConfig.pauseMs, 900);

    // Invalid pause or beam rejected
    assert.throws(() => dv2.updateLocalConfig({ pauseMs: 777 }), /Pausa Discord/);
    assert.throws(() => dv2.updateLocalConfig({ beamSize: 9 }), /Beam size/);

    // Reset to Legacy restores exact production defaults
    dv2.updateLocalConfig({ profile: "legacy" });
    assert.deepEqual(dv2.localConfig, LOCAL_LEGACY_DEFAULTS);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

test("werOps normNyra normalizes Nyra/Nira variants without hiding Mirá or Nyra duplicates", () => {
  assert.equal(werOps("Che Nira", "Che Nyra", normNyra).rate, 0);
  assert.notEqual(werOps("Mirá esto", "Nira esto", normNyra).rate, 0);
  assert.notEqual(werOps("Ey Nira", "Ey Nyra Nyra", normNyra).rate, 0);
  assert.notEqual(werOps("Che Nira", "Che Nyra", norm).rate, 0);
});
