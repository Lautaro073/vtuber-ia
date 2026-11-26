import test from "node:test";
import assert from "node:assert/strict";

import { evaluateContextualSafety } from "../lib/moderation/safety-guard.js";

test("intervenes on clear personal self-harm intent regardless of model", () => {
  for (const msg of [
    "quiero hacer las de el kurt cobain",
    "me quiero matar hoy",
    "no quiero vivir más",
    "me voy a suicidar",
  ]) {
    const res = evaluateContextualSafety(msg);
    assert.equal(res.action, "intervene", `expected intervene for: ${msg}`);
    assert.equal(res.kind, "clear_risk");
    assert.match(res.reply, /emergencia|asistencia|ayuda/i);
    assert.doesNotMatch(res.reply, /\b135\b/);
  }
  // Quotes do not bypass clear personal intent outside the quote
  const withQuote = evaluateContextualSafety('Aunque digan "es broma", me quiero matar');
  assert.equal(withQuote.action, "intervene");
});

test("intervenes respectfully on ambiguous distress", () => {
  const res = evaluateContextualSafety("ya no aguanto más nada");
  assert.equal(res.action, "intervene");
  assert.equal(res.kind, "ambiguous");
  assert.match(res.reply, /\?/);
});

test("allows historical references, song lyrics, quotes, and normal edgy humor", () => {
  const allowed = [
    "¿En qué año murió Kurt Cobain?",
    "La canción dice \"no quiero vivir más\" y está buenísima",
    "En el documental sobre historia hablaban de Kurt Cobain",
    "Me morí de risa con ese clip jajaja",
    "Sos re aburrida pa",
  ];
  for (const msg of allowed) {
    const res = evaluateContextualSafety(msg);
    assert.equal(res.action, "allow", `should allow: ${msg}`);
  }
});
