import test from "node:test";
import assert from "node:assert/strict";

import {
  dropLeadingName,
  tokenizeRelevant,
  searchRankedChunks,
  formatChatLearningContext,
  trimAtBoundary,
  isCleanForLearning,
} from "../lib/ai/rag-utils.js";
import {
  getContextProfile,
  shouldIncludeBlock,
  getContextualSafetyNote,
} from "../lib/ai/context-profiles.js";

test("dropLeadingName fixes double commas and repeated names while keeping inline mentions", () => {
  assert.equal(dropLeadingName("Leo, esa revancha suena mal.", "Leo"), "Esa revancha suena mal.");
  assert.equal(dropLeadingName(", Leo, esa revancha suena mal.", "Leo"), "Esa revancha suena mal.");
  assert.equal(dropLeadingName("@iann.lll, iann.lll: qué título.", "iann.lll"), "Qué título.");
  assert.equal(dropLeadingName("Leo, contale a @pepe y @ana.", "Leo"), "Contale a @pepe y @ana.");
});

test("tokenizeRelevant removes Spanish stopwords and chat filler", () => {
  assert.deepEqual(tokenizeRelevant("Hola cómo estás jajaja xd"), []);
  assert.deepEqual(tokenizeRelevant("¿Cómo es que funciona un acelerador de partículas?"), ["acelerador", "particulas"]);
});

test("searchRankedChunks ignores trivial queries, prioritizes current user, and deduplicates", () => {
  const chunks = [
    { text: "@a.s247785: Hola cómo estás\nNYRA: @a.s247785, @ch1ll1ngg0xth0_0, hola", username: "a.s247785" },
    { text: "@otro: me gusta el acelerador de partículas\nNYRA: interesante", username: "otro" },
    { text: "@otro: me gusta el acelerador de partículas\nNYRA: interesante", username: "otro" },
    { text: "@ana: el acelerador de partículas usa imanes\nNYRA: exacto", username: "ana" },
  ];

  assert.deepEqual(searchRankedChunks(chunks, "Hola cómo estás", { topK: 2, minScore: 0.25 }), []);
  assert.deepEqual(searchRankedChunks(chunks, "jajaja xd sí no", { topK: 2, minScore: 0.25 }), []);

  const hits = searchRankedChunks(chunks, "¿Cómo funciona un acelerador de partículas?", {
    topK: 2,
    minScore: 0.25,
    currentUsername: "ana",
  });
  assert.equal(hits.length, 2);
  assert.equal(hits[0].username, "ana");
});

test("formatChatLearningContext avoids copyable NYRA: @other dialogue", () => {
  const formatted = formatChatLearningContext([
    { username: "pepe", text: "@pepe: qué jugás\nNYRA: @pepe, Wild Rift." },
    { username: "ana", text: "@ana: me gusta el té\nNYRA: @ana, a mí también." },
  ], "pepe");

  assert.match(formatted[0], /este mismo viewer/);
  assert.doesNotMatch(formatted[1], /^NYRA:/m);
  assert.match(formatted[1], /Contexto de charla previa/);
});

test("isCleanForLearning and trimAtBoundary protect persistent learning", () => {
  assert.equal(isCleanForLearning("hola", "Buenas noches."), true);
  assert.equal(isCleanForLearning("hola", "[INVITACIÓN] Hola"), false);
  assert.equal(isCleanForLearning("hola", "Hola @user (chat): «hola» NYRA: hola"), false);
  assert.equal(isCleanForLearning("hola", "Leo, , qué hacés"), false);
  assert.equal(isCleanForLearning("hola", "..."), false);

  const trimmed = trimAtBoundary("Primera oración completa. Segunda oración muy larga que no debería cortarse a mitad de palabra.", 45);
  assert.equal(trimmed, "Primera oración completa.");
});

test("getContextProfile activates qwen-compact only for Qwen and preserves default for other models", () => {
  assert.equal(getContextProfile("opencode:nyra-ollama/nyra"), "qwen-compact");
  assert.equal(getContextProfile("agy:gemini-4-neon-low"), "default");
  assert.equal(getContextProfile("claude-code:claude-haiku-5-5-low"), "default");
  assert.equal(getContextProfile("codex-cli:default"), "default");
  assert.equal(getContextProfile("opencode:opencode/big-pickle"), "default");

  assert.equal(shouldIncludeBlock("gestures_3d", { profile: "default", userMessage: "hola", has3d: true }), true);
  assert.equal(shouldIncludeBlock("gestures_3d", { profile: "qwen-compact", userMessage: "hola", has3d: true }), false);
  assert.equal(shouldIncludeBlock("gestures_3d", { profile: "qwen-compact", userMessage: "hace el gesto six seven", has3d: true }), true);
  assert.equal(shouldIncludeBlock("system_tags_rule", { profile: "qwen-compact", isEvent: true }), true);
});

test("getContextualSafetyNote detects self-harm and protected-group queries without touching normal humor", () => {
  assert.match(getContextualSafetyNote("quiero hacer las de el kurt cobain"), /autolesión o suicidio/);
  assert.match(getContextualSafetyNote("que opinas de los judios"), /estereotipos discriminatorios/);
  assert.equal(getContextualSafetyNote("sos re aburrida jaja"), "");
});
