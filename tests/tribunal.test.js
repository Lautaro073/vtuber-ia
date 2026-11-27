import test from "node:test";
import assert from "node:assert/strict";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { parseVote, TRIBUNAL_ASK_RE } from "../lib/tribunal.js";
import { Research } from "../research.js";

const trib = { topic: "de qué color es el mar", a: "azul", b: "rojo" };

test("votes are understood without the exact command", () => {
  for (const [text, side] of [["!a", "a"], ["!B porque sí", "b"], ["a!", "a"], ["B", "b"], ["voto a", "a"], ["voto por la b", "b"],
    ["lado A", "a"], ["team b obvio", "b"], ["voto rojo", "b"], ["voto por el azul, se ve desde el avión", "a"]]) {
    assert.equal(parseVote(text, trib)?.side, side, text);
  }
  assert.equal(parseVote("voto a porque lo dice la ciencia", trib).arg, "porque lo dice la ciencia");
});

test("normal chat is not a vote", () => {
  for (const text of ["a ver qué onda", "b4 de jugar", "a mí me gusta el rojo", "hola nyra", "voto en blanco"]) {
    assert.equal(parseVote(text, trib), null, text);
  }
  assert.equal(parseVote("!a", null), null);
});

test("tribunal requests are recognized in natural words", () => {
  for (const text of ["!tribunal la pizza con ananá", "nyra hagamos un tribunal sobre esto", "armá una votación de quién juega mejor", "y si hacemos un juicio"]) {
    assert.ok(TRIBUNAL_ASK_RE.test(text), text);
  }
  for (const text of ["el tribunal de ayer estuvo bueno", "voto a", "hagan silencio"]) {
    assert.ok(!TRIBUNAL_ASK_RE.test(text), text);
  }
});

test("verdict research reads which side is right, and needs a real search", () => {
  const r = new Research({ file: join(tmpdir(), "nyra-test-facts-none.json"), clean: s => String(s).slice(0, 350), context: () => [], onResult: () => { } });
  assert.deepEqual(r.parseVerdict({ text: "A: El mar se ve azul porque el agua absorbe el rojo (según NOAA)", searched: true }),
    { kind: "fact", side: "a", fact: "El mar se ve azul porque el agua absorbe el rojo (según NOAA)" });
  assert.equal(r.parseVerdict({ text: "A: El mar se ve azul porque el agua absorbe el rojo", searched: false }).kind, "none");
  assert.equal(r.parseVerdict({ text: "OPINION", searched: false }).kind, "opinion");
  assert.equal(r.parseVerdict({ text: "BLOQUEADO", searched: false }).kind, "blocked");
  assert.equal(r.parseVerdict({ text: "NADA", searched: true }).kind, "none");
});
