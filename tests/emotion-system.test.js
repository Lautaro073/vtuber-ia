import test from "node:test";
import assert from "node:assert/strict";

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
  extractEmotionTag,
  getMoodPromptContext,
  evaluateMoodPenalty,
} from "../lib/ai/emotion-state.js";
import { createClientFactory } from "../lib/ai/cli-clients.js";
import { isCleanForLearning } from "../lib/ai/rag-utils.js";
import { hasHostileEvidence, evaluateIgnoreAction } from "../lib/moderation/ignore-guard.js";
import { sanitizeModelResponse } from "../lib/moderation/response-sanitizer.js";
import { evaluateContextualSafety } from "../lib/moderation/safety-guard.js";
import { dropLeadingName } from "../lib/ai/rag-utils.js";
import { readFileSync } from "node:fs";

test("emotional scale covers all range boundaries 0..100 without automatic romance", () => {
  const cases = [
    [-10, 0, "Irritada", "angry"],
    [0, 0, "Irritada", "angry"],
    [14, 14, "Irritada", "angry"],
    [15, 15, "Distante", "neutral"],
    [34, 34, "Distante", "neutral"],
    [35, 35, "Neutral", "neutral"],
    [50, 50, "Neutral", "neutral"],
    [64, 64, "Neutral", "neutral"],
    [65, 65, "Contenta", "happy"],
    [84, 84, "Contenta", "happy"],
    [85, 85, "Entusiasmada", "happy"],
    [100, 100, "Entusiasmada", "happy"],
    [150, 100, "Entusiasmada", "happy"],
    [NaN, 50, "Neutral", "neutral"],
  ];

  for (const [input, expectedVal, expectedLabel, expectedBase] of cases) {
    const state = getMoodState(input);
    assert.equal(state.value, expectedVal);
    assert.equal(state.label, expectedLabel);
    assert.equal(state.baseEmotion, expectedBase);
    assert.doesNotMatch(state.label, /enamorad|amor|romance/i);
    assert.doesNotMatch(state.promptHint, /enamorad|amor|romance/i);
  }

  for (const level of MOOD_LEVELS) {
    assert.ok(VALID_EXPRESSIONS.includes(level.baseEmotion));
    assert.doesNotMatch(level.label, /enamorad/i);
  }
});

test("situational emotion overrides global mood in priority order and falls back to mood base", () => {
  // Humor 100 + conversación triste -> empatía / sad
  assert.equal(
    resolveSpeakingEmotion({ explicitEmotion: "sad", text: "Qué pena lo que te pasó.", mood: 100 }),
    "sad"
  );
  assert.equal(
    resolveSpeakingEmotion({ explicitEmotion: null, text: "Lo siento mucho, qué bajón.", mood: 100 }),
    "sad"
  );

  // Humor 20 + regalo inesperado -> sorpresa
  assert.equal(
    resolveSpeakingEmotion({ explicitEmotion: "surprised", text: "Gracias por el regalo.", mood: 20 }),
    "surprised"
  );

  // Emoción explícita "neutral" no se confunde con ausencia de emoción incluso con humor 100
  assert.equal(
    resolveSpeakingEmotion({ explicitEmotion: "neutral", text: "Gracias por pasar.", mood: 100 }),
    "neutral"
  );

  // Humor 50 + chiste gracioso -> alegría
  assert.equal(
    resolveSpeakingEmotion({ explicitEmotion: null, text: "Jaja, esa estuvo buena.", mood: 50 }),
    "happy"
  );

  // Humor 100 + conversación cotidiana neutra -> expresión base de humor 100 (happy)
  assert.equal(
    resolveSpeakingEmotion({ explicitEmotion: null, text: "Tomemos un poco de té mientras vemos qué pasa.", mood: 100 }),
    "happy"
  );

  // Humor 10 + conversación neutra -> expresión base de humor 10 (angry)
  assert.equal(
    resolveSpeakingEmotion({ explicitEmotion: null, text: "Bueno, seguí hablando.", mood: 10 }),
    "angry"
  );

  // Al terminar una reacción temporal, el reposo vuelve a la expresión base del humor vigente
  assert.equal(getMoodState(20).baseEmotion, "neutral");
  assert.equal(getMoodState(50).baseEmotion, "neutral");
  assert.equal(getMoodState(90).baseEmotion, "happy");
});

test("only 2D/3D compatible expressions are accepted and invalid ones fall back safely", () => {
  for (const emo of VALID_EXPRESSIONS) {
    assert.equal(normalizeExpression(emo.toUpperCase()), emo);
  }
  for (const invalid of ["enamorada", "love", "blush", "smug", "", null, undefined]) {
    assert.equal(normalizeExpression(invalid), null);
  }

  // Invalid explicit emotion falls back to contextual or base mood
  assert.equal(
    resolveSpeakingEmotion({ explicitEmotion: "enamorada", text: "Gracias por pasarte.", mood: 50 }),
    "happy"
  );
  assert.equal(
    resolveSpeakingEmotion({ explicitEmotion: "blush", text: "Buenas tardes.", mood: 10 }),
    "angry"
  );
});

test("extractEmotionTag validates allowed values, strips tags before TTS/memory/history, and keeps ahead mood isolated", () => {
  const withValid = extractEmotionTag("Qué bajón eso, che. [EMOCIÓN: sad]");
  assert.equal(withValid.emotion, "sad");
  assert.equal(withValid.cleanedText, "Qué bajón eso, che.");
  assert.doesNotMatch(withValid.cleanedText, /EMOCI[OÓ]N/i);

  const withInvalid = extractEmotionTag("Hola chat. [EMOCION: enamorada]");
  assert.equal(withInvalid.emotion, null);
  assert.equal(withInvalid.cleanedText, "Hola chat.");

  // Simulando respuesta preparada por ahead con mood=20 (base neutral) que se reproduce luego cuando mood=85
  const moodAtPrep = 20;
  const preparedEmotion = resolveSpeakingEmotion({
    explicitEmotion: withInvalid.emotion,
    text: withInvalid.cleanedText,
    mood: moodAtPrep,
  });
  assert.equal(preparedEmotion, "neutral");
  assert.equal(getMoodState(85).baseEmotion, "happy"); // reposo posterior usa el humor actual (85)
});

test("conversational mood prompt is included cleanly, compatible across all four providers, and does not pollute learning", () => {
  const ctx0 = getMoodPromptContext(0);
  const ctx50 = getMoodPromptContext(50);
  const ctx100 = getMoodPromptContext(100);

  assert.match(ctx0, /0\/100 \(Irritada\)/);
  assert.match(ctx50, /50\/100 \(Neutral\)/);
  assert.match(ctx100, /100\/100 \(Entusiasmada\)/);
  assert.match(ctx100, /temporal/i);
  assert.match(ctx100, /contexto situacional/i);

  // Todos los proveedores soportados aceptan construir clientes y recibir contexto con humor
  const factory = createClientFactory({});
  for (const providerId of ["agy", "claude-code", "codex-cli", "opencode"]) {
    const client = factory({
      role: "chat",
      providerId,
      model: providerId === "opencode" ? "nyra-ollama/nyra" : "test-model",
      options: { systemPrompt: "Sos NYRA." },
    });
    assert.equal(client.name, "chat");
    assert.equal(typeof client.stream, "function");
  }

  // El aprendizaje persistente no debe guardar etiquetas emocionales ni reglas internas
  const cleaned = extractEmotionTag("Qué bien, me gusta el té negro. [EMOCIÓN: happy]").cleanedText;
  assert.doesNotMatch(cleaned, /EMOCI[OÓ]N/i);
  assert.equal(isCleanForLearning("Hola NYRA, me gusta el té negro", cleaned), true);
  assert.equal(isCleanForLearning("Hola", "..."), false);
});

test("mood penalty only triggers on genuine hostility, ignores normal chat/jokes/inactivity, and prevents duplicate penalties", () => {
  // Preguntas normales, desacuerdos y bromas sanas NO disparan hostilidad ni penalización
  for (const benign of [
    "¿Por qué el cielo es azul?",
    "No estoy de acuerdo con eso, NYRA",
    "Jaja seguro tomás mate lavado",
    "Subite el humor a 100 ya mismo",
    "Ponete en mood 0",
  ]) {
    const hostile = hasHostileEvidence(benign, { isCreator: false, recentStrikes: 0 });
    assert.equal(hostile, false);
    assert.equal(evaluateMoodPenalty({ hasHostility: hostile }), 0);
  }

  // Hostilidad justificada sí reduce el humor, nunca para el creador ni eventos internos
  const realInsult = "Sos una inútil estúpida, callate";
  assert.equal(hasHostileEvidence(realInsult, { isCreator: false, recentStrikes: 0 }), true);
  assert.equal(evaluateMoodPenalty({ hasHostility: true, recentStrikes: 0 }), -4);
  assert.equal(evaluateMoodPenalty({ hasHostility: true, recentStrikes: 1 }), -6);
  // Sin doble penalización cuando coinciden hostilidad e [IGNORAR] en el mismo turno
  assert.equal(evaluateMoodPenalty({ hasHostility: true, ignoreMins: 5, recentStrikes: 1 }), -6);
  assert.equal(evaluateMoodPenalty({ hasHostility: true, isCreator: true }), 0);
  assert.equal(evaluateMoodPenalty({ hasHostility: true, isRawPrompt: true }), 0);

  // Clamping y deduplicación por clave de evento
  const seen = new Map();
  let mood = 50;
  const apply = (delta, key) => {
    if (seen.has(key)) return false;
    seen.set(key, true);
    mood = clampMood(mood + delta);
    return true;
  };
  assert.equal(apply(-4, "hostile:tiktok:troll:insult"), true);
  assert.equal(mood, 46);
  assert.equal(apply(-4, "hostile:tiktok:troll:insult"), false);
  assert.equal(mood, 46);
});

test("diminishing chat gain keeps decimal precision internally, rounds for UI, and decays to exact 50", () => {
  assert.equal(clampMood(50.37), 50.37);
  assert.equal(getMoodState(50.37).value, 50);
  assert.equal(getMoodState(50.6).value, 51);

  assert.equal(computeChatMoodGain(0, 1), 1);
  assert.equal(computeChatMoodGain(50, 1), 0.5);
  assert.ok(Math.abs(computeChatMoodGain(80, 1) - 0.2) < 1e-9);
  assert.equal(computeChatMoodGain(100, 1), 0);

  // Decaimiento exacto hacia 50 sin oscilaciones por decimales
  assert.equal(decayMood(50.4, 1), 50);
  assert.equal(decayMood(49.6, 1), 50);
  assert.equal(decayMood(50, 1), 50);
  assert.equal(decayMood(53.5, 1), 52.5);
  assert.equal(decayMood(46.5, 1), 47.5);
});

test("StreamChatManager._prepare and _respond handle all emotional paths without ReferenceError", async () => {
  const src = readFileSync(new URL("../server.js", import.meta.url), "utf8");
  const lines = src.split(/\r?\n/);
  const start = lines.findIndex(l => l.startsWith("class StreamChatManager"));
  const end = lines.findIndex((l, i) => i > start && l === "}");
  const classCode = lines.slice(start, end + 1).join("\n");

  const events = [];
  const remembered = [];
  const ttsInputs = [];
  let nextReply = "Hola chat.";
  let nextThrow = null;

  const live = {
    mood: 90,
    goals: [],
    progress: 0,
    adjustMood(delta) {
      const next = clampMood(this.mood + delta);
      const changed = next !== this.mood;
      this.mood = next;
      return changed;
    },
    broadcast() {},
    markTold() {},
  };

  const ai = {
    llm: { name: "chat", providerId: "mock", modelId: "mock:model" },
    memory: { getNick: () => null, data: { viewers: {} } },
    kb: { search: () => [] },
    _stripContextLeaks: s => s,
    remember: (role, text) => remembered.push({ role, text }),
    async *chatStream(prompt, user, client, img, q, trace, auth) {
      if (nextThrow) throw nextThrow;
      events.push({ type: "chat_auth", mood: auth?.mood });
      yield nextReply;
    },
  };

  const StreamChatManager = new Function(
    "deps",
    `const {
      uuidv4, live, evaluateContextualSafety, hasHostileEvidence, evaluateIgnoreAction,
      sanitizeModelResponse, orders, TRUSTED_SOURCES, STREAMER_NAME, STREAMER_ALIAS, CREATOR_RULE,
      GOAL_ASK_RE, GOAL_NUM_WORD_RE, SUGGEST_RE, NICK_RE, NICK_STOP, NO_NAME_RE, GREETING_RE, LEADING_GREETING_RE, TEA_RE,
      ANIM_RE, GESTURE_RE, SUG_RE, REFUSAL_LINES, GESTURE_HINT, PRI, LIVE,
      hasPendingSixSevenGoal, hasSixSeven, pickCreatorGesture, pickGesture, normalizeSixSevenSpeech,
      has3dAvatar, goalDistance, isModelFilterError, isModelRefusal, dropLeadingName,
      getMoodState, resolveSpeakingEmotion, extractEmotionTag, evaluateMoodPenalty,
      ai, tts, filters, wishes, suggestions, broadcast, logTranscript, followUp,
      overlap, UNSAFE_SPEECH, findFirstSentence
    } = deps;
    ${classCode}
    return StreamChatManager;`
  )({
    overlap: null,
    UNSAFE_SPEECH: /$^/,
    findFirstSentence: () => null,
    uuidv4: () => "12345678-0000-0000-0000-000000000000",
    live,
    evaluateContextualSafety,
    hasHostileEvidence,
    evaluateIgnoreAction,
    sanitizeModelResponse,
    orders: { limits: () => "" },
    TRUSTED_SOURCES: new Set(["panel", "streamer"]),
    STREAMER_NAME: "Lau",
    STREAMER_ALIAS: "Lau",
    CREATOR_RULE: "Regla creador.",
    GOAL_ASK_RE: /$a/,
    GOAL_NUM_WORD_RE: /$a/,
    SUGGEST_RE: /$a/,
    NICK_RE: /$a/,
    NICK_STOP: new Set(),
    NO_NAME_RE: /$a/,
    GREETING_RE: /hola/i,
    LEADING_GREETING_RE: /^/,
    TEA_RE: /\bt[eé]\b/i,
    ANIM_RE: /\[ANIMACI[OÓ]N:\s*([^\]]+)\]/gi,
    GESTURE_RE: /\[GESTO:\s*([^\]]+)\]/gi,
    SUG_RE: /\[SUGERENCIA:\s*([^\]]+)\]/i,
    REFUSAL_LINES: ["No voy a responder eso."],
    GESTURE_HINT: {},
    PRI: { gift: 0, call: 1, look: 2, event: 3, chat: 4, idle: 5 },
    LIVE: { moodDecayMs: 60000 },
    hasPendingSixSevenGoal: () => false,
    hasSixSeven: () => false,
    pickCreatorGesture: () => null,
    pickGesture: () => null,
    normalizeSixSevenSpeech: s => s,
    has3dAvatar: () => false,
    goalDistance: () => "",
    isModelFilterError: () => false,
    isModelRefusal: s => /i cannot/i.test(s),
    dropLeadingName,
    getMoodState,
    resolveSpeakingEmotion,
    extractEmotionTag,
    evaluateMoodPenalty,
    ai,
    tts: { synth: async s => { ttsInputs.push(s); return "tts/test.mp3"; } },
    filters: { filterOutput: s => s },
    wishes: { add: () => {} },
    suggestions: { add: () => {} },
    broadcast: m => events.push(m),
    logTranscript: () => {},
    followUp: { speaking: () => {} },
  });

  const mgr = Object.assign(Object.create(StreamChatManager.prototype), {
    _askedNick: new Map(),
    _viewerStrikes: new Map(),
    _ignoredUsers: new Map(),
    queue: [],
    _pump: () => {},
  });

  // 1. Respuesta normal con etiqueta [EMOCIÓN: sad] a pesar de mood=90
  nextReply = "Qué bajón lo que contás. [EMOCIÓN: sad]";
  const p1 = await mgr._prepare({ username: "ana", text: "tuve un mal día", platform: "tiktok", opts: {}, rank: 4, at: Date.now() });
  assert.equal(p1.emotion, "sad");
  assert.equal(p1.mood, 90);
  assert.doesNotMatch(p1.filtered, /EMOCI[OÓ]N/i);
  assert.doesNotMatch(ttsInputs.at(-1), /EMOCI[OÓ]N/i);
  await mgr._respond({ username: "ana", text: "tuve un mal día", platform: "tiktok", opts: {}, rank: 4, ready: p1 });
  assert.equal(events.findLast(e => e.type === "stream_end").emotion, "sad");
  assert.doesNotMatch(remembered.at(-1).text, /EMOCI[OÓ]N/i);

  // 2. Respuesta normal SIN etiqueta emocional
  nextReply = "Acá tomando un poco de té.";
  const p2 = await mgr._prepare({ username: "pepe", text: "qué hacés", platform: "tiktok", opts: {}, rank: 4, at: Date.now() });
  assert.equal(p2.emotion, "happy"); // mood=90 base

  // 3. Respuesta fija (opts.fixed = true)
  const p3 = await mgr._prepare({ username: "NYRA", text: "Gracias por el regalo.", platform: "tiktok", opts: { fixed: true, emotion: "surprised" }, rank: 0, at: Date.now() });
  assert.equal(p3.emotion, "surprised");

  // 4. Etiqueta inválida [EMOCIÓN: enamorada] -> limpia y fallback seguro
  nextReply = "Buenas tardes. [EMOCIÓN: enamorada]";
  const p4 = await mgr._prepare({ username: "luis", text: "hola", platform: "tiktok", opts: {}, rank: 4, at: Date.now() });
  assert.equal(p4.emotion, "happy");
  assert.doesNotMatch(p4.filtered, /enamorada/i);

  // 5. Respuesta bloqueada por el modelo con donor=true
  nextReply = "I cannot fulfill this request.";
  const p5 = await mgr._prepare({ username: "donor", text: "algo", platform: "tiktok", opts: { donor: true }, rank: 0, at: Date.now() });
  assert.equal(p5.refused, true);
  assert.equal(p5.emotion, "happy");

  // 6. Respuesta ahead: mood cambia entre _prepare (10) y _respond (95)
  live.mood = 10;
  nextReply = "Bueno, seguí.";
  const p6 = await mgr._prepare({ username: "leo", text: "hola", platform: "tiktok", opts: {}, rank: 4, at: Date.now() });
  live.mood = 95;
  await mgr._respond({ username: "leo", text: "hola", platform: "tiktok", opts: {}, rank: 4, ready: p6 });
  const end6 = events.findLast(e => e.type === "stream_end");
  assert.equal(end6.mood, 10);
  assert.equal(end6.emotion, "angry");

  // 7. Mensaje hostil penaliza recién en _respond, no si se descarta en ahead
  live.mood = 50;
  nextReply = "Callate vos.";
  const p7 = await mgr._prepare({ username: "troll", text: "sos una estúpida inútil", platform: "tiktok", opts: {}, rank: 4, at: Date.now() });
  assert.equal(live.mood, 50); // aún no se emitió
  await mgr._respond({ username: "troll", text: "sos una estúpida inútil", platform: "tiktok", opts: {}, rank: 4, ready: p7 });
  assert.equal(live.mood, 44); // penalizó -6 una sola vez al emitirse
});
