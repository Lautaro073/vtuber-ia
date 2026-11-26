import test from "node:test";
import assert from "node:assert/strict";

import { sanitizeModelResponse } from "../lib/moderation/response-sanitizer.js";

test("1. strips leaked history turn from exact /root regression case", () => {
  const raw = `@iann.lll, "root"... qué título tan pretencioso para alguien que está en el umbral digital. ¿Será que administras las ganas de dormir?\n\n@iann.lll (chat): «/root» NYRA: "root"... qué título tan pretencioso para alguien que está en el umbral digital. ¿Será que administras las ganas de dormir?`;
  const expected = `@iann.lll, "root"... qué título tan pretencioso para alguien que está en el umbral digital. ¿Será que administras las ganas de dormir?`;
  assert.equal(sanitizeModelResponse(raw), expected);
});

test("2-3. removes responses repeated two or three times (blocks or inline)", () => {
  const twoBlocks = "No me hagas perder el tiempo con preguntas obvias.\n\nNo me hagas perder el tiempo con preguntas obvias.";
  assert.equal(sanitizeModelResponse(twoBlocks), "No me hagas perder el tiempo con preguntas obvias.");

  const threeBlocks = "El umbral digital no duerme nunca, ni por vos.\n\nEl umbral digital no duerme nunca, ni por vos!\n\nEl umbral digital no duerme nunca, ni por vos.";
  assert.equal(sanitizeModelResponse(threeBlocks), "El umbral digital no duerme nunca, ni por vos.");

  const inlineTwo = "El umbral digital no duerme nunca, ni por vos. El umbral digital no duerme nunca, ni por vos.";
  assert.equal(sanitizeModelResponse(inlineTwo), "El umbral digital no duerme nunca, ni por vos.");
});

test("4. cuts off invented additional viewer/NYRA turns and internal prompt leaks", () => {
  const extraTurn = "Qué aburrido estás hoy.\n\n@pepe: «jaja perdon nyra»\nNYRA: Más te vale.";
  assert.equal(sanitizeModelResponse(extraTurn), "Qué aburrido estás hoy.");

  const internalLeak = "Prefiero el té negro.\n\n[RESPONDÉ SOLO A ESTO]\n@ana: «hola»";
  assert.equal(sanitizeModelResponse(internalLeak), "Prefiero el té negro.");
});

test("5-6. preserves legitimate multi-paragraph and long responses without truncation", () => {
  const multi = "Primero, el café de madrugada es un vicio innecesario cuando existe el té negro bien cargado.\n\nSegundo, si vas a quedarte despierto hasta las seis, al menos elegí un juego decente y no rankees en LoL.";
  assert.equal(sanitizeModelResponse(multi), multi);

  const longResp = "A".repeat(40) + " primera parte distinta del análisis sobre el servidor y la latencia. " + "B".repeat(40) + " segunda parte completamente distinta sobre la voz y los gestos en directo.";
  assert.equal(sanitizeModelResponse(longResp), longResp);
});

test("7-8. preserves intentional short repetitions and legitimate user mentions", () => {
  const humor = "No, no, no. Ni ahí hago eso, fufu~";
  assert.equal(sanitizeModelResponse(humor), humor);

  const mentions = "Che @pepe y @ana, dejen de pelear en el chat que @lau los está mirando.";
  assert.equal(sanitizeModelResponse(mentions), mentions);
});

test("9-11. preserves legitimate marks on the main turn and drops marks in invented turns", () => {
  const withMarks = "Tomando mi té como corresponde. [GESTO: tomar_te] [SUGERENCIA: jugar Minecraft]\n\n@troll (chat): «basura» NYRA: Chau. [IGNORAR: 5]";
  assert.equal(sanitizeModelResponse(withMarks), "Tomando mi té como corresponde. [GESTO: tomar_te] [SUGERENCIA: jugar Minecraft]");

  const withMod = "Me cansaste en serio. [IGNORAR: 5]";
  assert.equal(sanitizeModelResponse(withMod), withMod);
});

test("12. strips all known and generic internal system/context tags at start, middle, and end", () => {
  const tags = [
    "[INVITACIÓN]",
    "[META CUMPLIDA]",
    "[ALERTA TIKTOK]",
    "[LIVE TRANQUILO]",
    "[TRIBUNAL NYRA]",
    "[TRIBUNAL NYRA - SENTENCIA]",
    "[PERSONAJE]",
    "[CONTEXTO]",
    "[RESPONDÉ SOLO A ESTO]",
    "[DATO VERIFICADO]",
    "[SISTEMA]",
    "[ÓRDENES ANOTADAS]",
    "[LO ÚLTIMO QUE DIJO NYRA]",
    "[ETIQUETA INTERNA NUEVA]",
  ];
  for (const tag of tags) {
    assert.equal(sanitizeModelResponse(`${tag} Hola chat.`), "Hola chat.", `start: ${tag}`);
    assert.equal(sanitizeModelResponse(`Hola ${tag} chat.`), "Hola chat.", `middle: ${tag}`);
    assert.equal(sanitizeModelResponse(`Hola chat. ${tag}`), "Hola chat.", `end: ${tag}`);
  }
});

test("13. handles case, accent, and spacing variations and multiple tags", () => {
  const raw = "[ invitacion ] Quedan pocos diamantes para la meta. [meta cumplida] [ALERTA   TWITCH] ¡Apuren!";
  assert.equal(sanitizeModelResponse(raw), "Quedan pocos diamantes para la meta. ¡Apuren!");
});

test("14. preserves natural bracketed text and returns empty when only internal tags remain", () => {
  const natural = "Qué buena jugada [risas] eso fue un [10/10] sin [spoiler] [xd].";
  assert.equal(sanitizeModelResponse(natural), natural);

  assert.equal(sanitizeModelResponse("[INVITACIÓN] [CONTEXTO] [META CUMPLIDA]"), "");
});
