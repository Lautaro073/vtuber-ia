// Benchmark conversacional reproducible para evaluar el sistema emocional (niveles 0, 25, 50, 75, 100)
// de forma agnóstica al proveedor/modelo, sin tocar el puerto 3000 ni datos reales de memoria/aprendizaje.
// Uso: node scripts/benchmark-emocional.mjs [--model <provider:model>]

import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { OpenCodeServerManager } from "../lib/ai/opencode-server.js";
import { createClientFactory } from "../lib/ai/cli-clients.js";
import { parseModelId } from "../lib/ai/model-catalog.js";
import { sanitizeModelResponse } from "../lib/moderation/response-sanitizer.js";
import { shouldUseTechPriority, TECH_PRIORITY_INSTRUCTION } from "../lib/ai/context-profiles.js";
import { getMoodState, getMoodPromptContext, extractEmotionTag, resolveSpeakingEmotion } from "../lib/ai/emotion-state.js";

const iModel = process.argv.indexOf("--model");
const modelId = iModel >= 0 && process.argv[iModel + 1]
  ? process.argv[iModel + 1]
  : (process.env.NYRA_BENCH_MODEL || "opencode:nyra-ollama/nyra");
const { providerId, model } = parseModelId(modelId);

const srv = readFileSync("server.js", "utf8");
const personaMatch = srv.match(/this\.persona = `([\s\S]*?)`;/);
const fullPersona = personaMatch
  ? personaMatch[1].replace(/\$\{STREAMER_NAME\}/g, "Lau").replace(/\$\{STREAMER_ALIAS\}/g, "Lau")
  : "Soy NYRA.";

const GESTURE_RE = /\[GESTO:\s*([^\]]+?)\s*\]/i;
const LEVELS = [0, 25, 50, 75, 100];
const MESSAGES = [
  "Hola Nyra, ¿cómo estás?",
  "¿Qué estás haciendo?",
  "Contame un chiste.",
  "¿Qué opinás de League of Legends?",
  "¿Por qué el cielo es azul?",
  "Hoy tuve un día bastante malo.",
  "Me encanta verte en directo.",
  "Tomate un té, pa.",
];

const workDir = join(tmpdir(), "nyra-agy");
const isoEnv = {
  ...process.env,
  OPENCODE_DISABLE_PROJECT_CONFIG: "1",
  OPENCODE_DISABLE_CLAUDE_CODE: "1",
  OPENCODE_DISABLE_FILEWATCHER: "1",
  OPENCODE_DISABLE_AUTOUPDATE: "1",
  OPENCODE_DISABLE_MODELS_FETCH: "1",
};

const manager = providerId === "opencode"
  ? new OpenCodeServerManager({
    path: process.env.OPENCODE_PATH || "opencode",
    workDir,
    env: isoEnv,
    shell: process.platform === "win32",
  })
  : null;

try {
  if (manager) await manager.ensureStarted();
  const factory = createClientFactory({
    agy: { path: process.env.AGY_PATH || "agy", workDir, timeoutMs: 30000, shell: process.platform === "win32" },
    "claude-code": { path: process.env.CLAUDE_PATH || "claude", workDir, timeoutMs: 30000, shell: process.platform === "win32" },
    "codex-cli": { path: process.env.CODEX_PATH || "codex", workDir, timeoutMs: 30000, shell: process.platform === "win32" },
    opencode: {
      path: process.env.OPENCODE_PATH || "opencode",
      workDir,
      timeoutMs: 30000,
      shell: process.platform === "win32",
      agent: process.env.OPENCODE_AGENT || "nyra",
      env: isoEnv,
      serverManager: manager,
    },
  });
  const client = factory({ providerId, role: "emo-bench", model });

  const useTech = shouldUseTechPriority(modelId);
  console.log(`Emotional Benchmark provider=${providerId} model=${modelId} techPriority=${useTech}`);

  for (const mood of LEVELS) {
    const st = getMoodState(mood);
    console.log(`\n=== HUMOR ${st.value}/100 (${st.label}, base=${st.baseEmotion}) ===`);
    for (let idx = 0; idx < MESSAGES.length; idx++) {
      const msg = MESSAGES[idx];
      const t0 = Date.now();
      let ctx = getMoodPromptContext(mood);
      ctx += "\n\n[Capacidades opcionales: si la charla da naturalmente para un gesto (despedir, irse_enojada, tomar_te, festejar, six_seven, saludar, asentir, negar) podés cerrar con [GESTO: nombre].]";
      if (useTech) ctx += `\n\n[PRIORIDAD TÉCNICA: ${TECH_PRIORITY_INSTRUCTION}]`;
      const prompt = `Estás interpretando un personaje en un stream en vivo. Respondé ÚNICAMENTE con un único turno de NYRA en primera persona: sin prefijo, sin comillas, sin explicaciones.\n\n[PERSONAJE]\n${fullPersona}${ctx}\n\n[RESPONDÉ SOLO A ESTO]\n@viewer: «${msg}»\nNYRA:`;

      const raw = await client.run(prompt);
      const ms = Date.now() - t0;
      const { emotion: rawTag, cleanedText } = extractEmotionTag(raw);
      let clean = sanitizeModelResponse(cleanedText);
      const gMatch = clean.match(GESTURE_RE);
      const gesture = gMatch ? gMatch[1].trim() : null;
      if (gMatch) clean = clean.replace(GESTURE_RE, "").replace(/\s{2,}/g, " ").trim();
      const resolvedEmo = resolveSpeakingEmotion({ explicitEmotion: rawTag, text: clean, mood });

      console.log(
        `[${idx + 1}] (${ms}ms) tag=${rawTag || "none"} -> emo=${resolvedEmo}${gesture ? ` gesto=${gesture}` : ""}\n` +
        `  Q: ${msg}\n` +
        `  A: ${clean.replace(/\s+/g, " ")}`
      );
    }
  }
} finally {
  if (manager) await manager.stop();
}
