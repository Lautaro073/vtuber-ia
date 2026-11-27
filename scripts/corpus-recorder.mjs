// Herramienta local e independiente para grabar el corpus piloto de STT.
// Uso: node scripts/corpus-recorder.mjs [--port 3456]
// Solo escucha en 127.0.0.1, nunca usa el puerto 3000 y no inicia NYRA, Discord, LLM ni TTS.

import express from "express";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PROMPTS_FILE = join(ROOT, "docs", "stt-corpus", "prompts-40.json");
const VOICES_DIR = join(ROOT, "voices", "stt");
const CLIPS_DIR = join(VOICES_DIR, "clips");
const MANIFEST_FILE = join(VOICES_DIR, "manifest.json");
const HTML_FILE = join(ROOT, "scripts", "corpus-recorder.html");

export function resolveRecorderPort(argv = process.argv) {
  const i = argv.indexOf("--port");
  const port = i >= 0 && argv[i + 1] ? Number(argv[i + 1]) : 3456;
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new Error(`Puerto inválido: ${port}`);
  }
  if (port === 3000) {
    throw new Error("El puerto 3000 está reservado para NYRA; usá otro puerto (ej. 3456)");
  }
  return port;
}

function readManifest() {
  if (!existsSync(MANIFEST_FILE)) return [];
  try {
    const parsed = JSON.parse(readFileSync(MANIFEST_FILE, "utf8"));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeManifestAtomic(items) {
  mkdirSync(VOICES_DIR, { recursive: true });
  const tmp = `${MANIFEST_FILE}.${process.pid}.${Date.now()}.tmp`;
  writeFileSync(tmp, JSON.stringify(items, null, 2));
  renameSync(tmp, MANIFEST_FILE);
}

async function convertToPcm16kWav(inputPath, outputPath) {
  mkdirSync(dirname(outputPath), { recursive: true });
  const tmpOut = `${outputPath}.tmp.wav`;
  await execFileAsync("ffmpeg", [
    "-y",
    "-i", inputPath,
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    tmpOut,
  ]);
  renameSync(tmpOut, outputPath);
}

export function createRecorderApp() {
  const app = express();
  app.use(express.json({ limit: "25mb" }));

  app.get("/", (_req, res) => {
    res.type("html").send(readFileSync(HTML_FILE, "utf8"));
  });

  app.get("/api/corpus", (_req, res) => {
    const prompts = JSON.parse(readFileSync(PROMPTS_FILE, "utf8"));
    const manifest = readManifest();
    res.json({ ok: true, prompts, manifest, voicesDir: VOICES_DIR });
  });

  app.get("/api/clip/:id", (req, res) => {
    const id = String(req.params.id || "").replace(/[^a-zA-Z0-9_-]/g, "");
    const wavPath = join(CLIPS_DIR, `${id}.wav`);
    if (!id || !existsSync(wavPath)) return res.status(404).json({ ok: false, error: "Clip no encontrado" });
    res.type("audio/wav").send(readFileSync(wavPath));
  });

  app.post("/api/save", async (req, res) => {
    const { id, ref, split, category, hasNyra, silence, speaker, audioBase64, ext = "webm", authorizedImport = false, source = "mic" } = req.body || {};
    const cleanId = String(id || "").replace(/[^a-zA-Z0-9_-]/g, "");
    if (!cleanId) return res.status(400).json({ ok: false, error: "ID inválido" });
    if (source === "discord_import" && !authorizedImport) {
      return res.status(400).json({ ok: false, error: "Se requiere autorización explícita de los participantes para importar audio de Discord" });
    }
    if (!audioBase64 || typeof audioBase64 !== "string") {
      return res.status(400).json({ ok: false, error: "Falta audioBase64" });
    }
    const cleanExt = String(ext).replace(/[^a-zA-Z0-9]/g, "") || "webm";
    mkdirSync(CLIPS_DIR, { recursive: true });
    const rawTmp = join(CLIPS_DIR, `${cleanId}.${process.pid}.${Date.now()}.${cleanExt}`);
    const wavOut = join(CLIPS_DIR, `${cleanId}.wav`);
    try {
      writeFileSync(rawTmp, Buffer.from(audioBase64, "base64"));
      await convertToPcm16kWav(rawTmp, wavOut);
    } finally {
      try { unlinkSync(rawTmp); } catch { }
    }

    const entry = {
      id: cleanId,
      file: `clips/${cleanId}.wav`,
      ref: String(ref ?? "").trim(),
      split: split === "eval" ? "eval" : "tune",
      category: String(category || "custom"),
      hasNyra: Boolean(hasNyra),
      silence: Boolean(silence),
      speaker: String(speaker || "A").trim().slice(0, 16) || "A",
      source: source === "discord_import" ? "discord_authorized" : "mic",
    };

    const manifest = readManifest().filter(m => m.id !== cleanId);
    manifest.push(entry);
    writeManifestAtomic(manifest);
    res.json({ ok: true, entry, manifest });
  });

  app.delete("/api/clip/:id", (req, res) => {
    const cleanId = String(req.params.id || "").replace(/[^a-zA-Z0-9_-]/g, "");
    if (!cleanId) return res.status(400).json({ ok: false, error: "ID inválido" });
    const wavOut = join(CLIPS_DIR, `${cleanId}.wav`);
    try { unlinkSync(wavOut); } catch { }
    const manifest = readManifest().filter(m => m.id !== cleanId);
    writeManifestAtomic(manifest);
    res.json({ ok: true, manifest });
  });

  return app;
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const port = resolveRecorderPort(process.argv);
  const app = createRecorderApp();
  app.listen(port, "127.0.0.1", () => {
    console.log(`🎙️ Grabador de corpus STT activo en http://127.0.0.1:${port}`);
    console.log(`📁 Guardará WAVs PCM 16kHz mono en: ${CLIPS_DIR}`);
  });
}
