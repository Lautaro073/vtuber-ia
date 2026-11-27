// Integración emocional del servidor: contexto para el LLM, extracción de [EMOCIÓN:] y dinámica de humor.

import {
  VALID_EXPRESSIONS,
  MOOD_LEVELS,
  clampMood,
  getMoodState,
  computeChatMoodGain,
  decayMood,
  normalizeExpression,
  detectContextualEmotion,
  resolveSpeakingEmotion,
} from "../../public/emotion-config.js";

export {
  VALID_EXPRESSIONS,
  MOOD_LEVELS,
  clampMood,
  getMoodState,
  computeChatMoodGain,
  decayMood,
  normalizeExpression,
  detectContextualEmotion,
  resolveSpeakingEmotion,
};

const EMOTION_TAG_RE = /\[EMOCI[OÓ]N:\s*([^\]]+?)\s*\]/gi;

export function extractEmotionTag(text) {
  const raw = String(text || "");
  let emotion = null;
  for (const match of raw.matchAll(EMOTION_TAG_RE)) {
    const valid = normalizeExpression(match[1]);
    if (valid) emotion = valid;
  }
  const cleanedText = raw.replace(EMOTION_TAG_RE, "").replace(/\s{2,}/g, " ").trim();
  return { emotion, cleanedText };
}

export function getMoodPromptContext(mood) {
  const state = getMoodState(mood);
  return (
    `\n\n[ESTADO EMOCIONAL ACTUAL: Humor ${state.value}/100 (${state.label}). ` +
    `${state.promptHint} Este humor colorea tu tono, energía, paciencia y espontaneidad gestual, pero es temporal. ` +
    `Priorizá siempre el contexto situacional cuando corresponda (ej: empatía ante algo triste, sorpresa o gratitud ante un regalo inesperado, humor ante un chiste). ` +
    `Opcionalmente podés cerrar con [EMOCIÓN: happy|sad|angry|surprised|neutral] si la escena marca una emoción puntual distinta al reposo; no se lee en voz alta.]`
  );
}

export function evaluateMoodPenalty({
  hasHostility = false,
  ignoreMins = 0,
  recentStrikes = 0,
  isCreator = false,
  isRawPrompt = false,
} = {}) {
  if (isCreator || isRawPrompt) return 0;
  if (ignoreMins > 0) return -6;
  if (!hasHostility) return 0;
  return recentStrikes >= 1 ? -6 : -4;
}
