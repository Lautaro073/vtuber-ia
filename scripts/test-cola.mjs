// Prueba de prioridades de la cola (sin IA, sin red): node scripts/test-cola.mjs
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";

const src = readFileSync(new URL("../server.js", import.meta.url), "utf8");
const line = name => src.match(new RegExp(`^const ${name} = .*$`, "m"))[0];
const { nextUp, WAKE_RE, PRI } = new Function(`${line("PRI")}\n${line("WAKE_RE")}\n${line("SLOW_MS")}
${src.match(/^function nextUp[\s\S]*?^}/m)[0]}
return { nextUp, WAKE_RE, PRI };`)();

const now = 100000;
const it = (id, rank, ready = true, more = {}) => ({ id, rank, ready, at: now - 10000, ...more });
const next = (queue, streak = 0) => nextUp(queue, now, streak)?.id ?? null;

assert.equal(next([it("chat", PRI.chat), it("voz", PRI.call), it("regalo", PRI.gift)]), "regalo");
assert.equal(next([it("meta", PRI.gift), it("gracias", PRI.gift)]), "meta", "regalos por orden de llegada");
assert.equal(next([it("regalo", PRI.gift, false), it("chat", PRI.chat)]), null, "un regalo no se saltea");
assert.equal(next([it("voz", PRI.call), it("regalo", PRI.gift)], 5), "regalo", "el turno del chat no pisa regalos");
assert.equal(next([it("chat", PRI.chat), it("voz", PRI.call)]), "voz");
assert.equal(next([it("chat", PRI.chat), it("juego", PRI.look)]), "juego");
assert.equal(next([it("juego", PRI.look), it("voz", PRI.call)]), "voz");
assert.equal(next([it("voz", PRI.call, false, { at: now - 1000 }), it("chat", PRI.chat)]), null, "Haiku todavía llega");
assert.equal(next([it("voz", PRI.call, false), it("chat", PRI.chat)]), "chat", "Haiku tarda: pasa chat");
assert.equal(next([it("voz", PRI.call, false, { fillers: 1 }), it("chat", PRI.chat)]), "chat", "pasa un segundo");
assert.equal(next([it("voz", PRI.call, false, { fillers: 2 }), it("chat", PRI.chat)]), null, "después de dos, espera");
assert.equal(next([it("voz", PRI.call), it("chat", PRI.chat)], 2), "chat", "cada dos de voz, una de chat");
assert.equal(next([it("voz", PRI.call)], 2), "voz", "sin chat listo sigue la voz");
assert.equal(next([it("a", PRI.chat, false), it("b", PRI.chat)]), "b", "sale el que ya esté listo");
assert.equal(next([it("saludo", PRI.idle), it("chat", PRI.chat)]), "chat");
assert.equal(next([it("retenido", PRI.chat, true, { holdUntil: now + 1000 })]), null);

for (const s of ["Nira mirá", "che nyra, ¿qué opinás?", "decime ñira", "¿Nira?", "NYRA!", "Chenira, opiná", "¿Chénira?", "escucha, Menira"]) assert.ok(WAKE_RE.test(s), s);
for (const s of ["mira el mapa", "mirá ahí", "Nara va mid", "admira eso", "niranja", "che, mira esto", "es mentira"]) assert.ok(!WAKE_RE.test(s), s);

// El panel tiene su copia: tiene que despertar con lo mismo
const panel = readFileSync(new URL("../public/panel.html", import.meta.url), "utf8").match(/const wakeWords = (\/.*\/iu);/)[1];
assert.equal(panel, WAKE_RE.toString(), "regex del panel distinta a la del server");

// Apodos por voz (frases reales de la primera llamada)
const voiceNick = new Function(`${line("WAKE_RE")}\n${line("NICK_STOP")}\n${line("VOICE_NICK_RE")}
${src.match(/^function voiceNick[\s\S]*?^}/m)[0]}
return voiceNick;`)();
for (const [said, nick] of [
  ["Nira, por favor cámbiame el apodo y decime el Jesus, por favor.", "el Jesus"],
  ["Nira, decime el Jesus.", "el Jesus"],
  ["Nira, pero es que mi apodo es el Jesus con Y.", "el Jesus"],
  ["Nira, mi apodo es Charly", "Charly"],
  ["Decime El Profe, Nira.", "el Profe"],
  ["Nira, decime quién va ganando.", null],
  ["Nira, decime qué me armo contra ese tanque.", null],
  ["Nira, decime Dragón o barón.", null],
  ["Decime, Nira, ¿voy o no?", null],
]) assert.equal(voiceNick(said), nick, said);

console.log("cola y activación: ok");
