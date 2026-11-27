// Fuente única de verdad para la escala emocional, colores y expresiones compatibles (2D y 3D).
// Importado tanto por Node (server.js / tests) como por el navegador (overlay.html / obs.html).

export const VALID_EXPRESSIONS = ["neutral", "happy", "sad", "angry", "surprised"];

export const MOOD_LEVELS = [
  {
    key: "irritada",
    min: 0,
    max: 14,
    label: "Irritada",
    baseEmotion: "angry",
    color: "#ff4d6d",
    barGradient: "linear-gradient(90deg, #ff2d55, #ff6b6b)",
    promptHint: "Muy poca paciencia: respondé más cortante, seca o filosa, manteniendo la precisión si te preguntan algo técnico o real.",
  },
  {
    key: "distante",
    min: 15,
    max: 34,
    label: "Distante",
    baseEmotion: "neutral",
    color: "#6c8cff",
    barGradient: "linear-gradient(90deg, #4b6cb7, #7b9acc)",
    promptHint: "Tono frío, reservado y con menos ganas de explayarse, conservando tu ingenio.",
  },
  {
    key: "neutral",
    min: 35,
    max: 64,
    label: "Neutral",
    baseEmotion: "neutral",
    color: "#bb77ff",
    barGradient: "linear-gradient(90deg, #5b7cff, #bb77ff)",
    promptHint: "Tono habitual de NYRA: tranquila, irónica, directa y equilibrada.",
  },
  {
    key: "contenta",
    min: 65,
    max: 84,
    label: "Contenta",
    baseEmotion: "happy",
    color: "#4ce0b3",
    barGradient: "linear-gradient(90deg, #5b7cff, #4ce0b3)",
    promptHint: "Más receptiva, ágil y de buen humor; ironía ligera y mayor disposición a charlar.",
  },
  {
    key: "entusiasmada",
    min: 85,
    max: 100,
    label: "Entusiasmada",
    baseEmotion: "happy",
    color: "#ff85c0",
    barGradient: "linear-gradient(90deg, #bb77ff, #ff4fa3)",
    promptHint: "Energía alta, calidez y entusiasmo genuino con el directo, sin caer en romanticismo ni perder tu personalidad.",
  },
];

export function clampMood(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 50;
  return Math.max(0, Math.min(100, n));
}

export function getMoodState(mood) {
  const raw = clampMood(mood);
  const value = Math.round(raw);
  const level = MOOD_LEVELS.find(l => value >= l.min && value <= l.max) || MOOD_LEVELS[2];
  return { value, raw, ...level };
}

export function computeChatMoodGain(mood, baseGain = 1) {
  const m = clampMood(mood);
  return Math.max(0, Number(baseGain || 0) * (1 - m / 100));
}

export function decayMood(mood, step = 1) {
  const m = clampMood(mood);
  if (m === 50) return 50;
  return m > 50 ? Math.max(50, m - step) : Math.min(50, m + step);
}

export function normalizeExpression(emotion) {
  if (typeof emotion !== "string") return null;
  const norm = emotion.trim().toLowerCase();
  return VALID_EXPRESSIONS.includes(norm) ? norm : null;
}

export function detectContextualEmotion(text) {
  const t = String(text || "").toLowerCase();
  if (!t) return null;
  const has = words => new RegExp(`(^|[^a-záéíóúñ])(${words})`).test(t);
  if (has("idiot|estúpid|estupid|cállate|callate|kuka|zurd|peronist|asco|odio|ridícul|ridicul|harta|insoportable")) return "angry";
  if (has("¿qué\\?|en serio|no puede ser|wow|increíble|increible|sorpresa|épico|epico") || /!\?|\?!/.test(t)) return "surprised";
  if (has("triste|lástima|lastima|qué pena|que pena|lo siento|lamento|fuerza|ánimo|animo|extraño|aburrid|bajón|bajon")) return "sad";
  if (has("fufu|ara~|ara\\b|jaja|jeje|gracias|interesante|me gusta|lindo|bien hecho|genial|feliz|alegra")) return "happy";
  return null;
}

export function resolveSpeakingEmotion({ explicitEmotion = null, text = "", mood = 50 } = {}) {
  return (
    normalizeExpression(explicitEmotion) ||
    detectContextualEmotion(text) ||
    getMoodState(mood).baseEmotion
  );
}
