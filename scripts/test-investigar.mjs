// Cuándo investiga y qué no (sin red): node scripts/test-investigar.mjs
import assert from "node:assert/strict";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Research } from "../research.js";

function make() {
  const r = new Research({ file: join(tmpdir(), "no-existe-nyra-facts.json"), clean: (s, n) => s.slice(0, n).trim(), context: () => [], onResult() { } });
  r._run = async () => { };
  return r;
}
const tick = () => new Promise(r => setImmediate(r));
const t0 = 1e12, min = 60000;

let r = make();
assert.ok(r.notice("Si, me di cuenta xd / Si buscas ya salio el gta VI investiga", {}, t0), "el mensaje real del live");

r = make();
assert.ok(!r.notice("nyra ya salió el GTA", {}, t0), "una sola vez no alcanza");
assert.ok(!r.notice("nyra te equivocás", {}, t0 + min));
assert.ok(r.notice("nyra estás desactualizada", {}, t0 + 2 * min), "si insisten, busca");

r = make();
r.notice("ya salió", {}, t0); r.notice("te equivocás", {}, t0 + 4 * min);
assert.ok(!r.notice("es mentira", {}, t0 + 6 * min), "lo viejo no suma");

r = make();
assert.ok(r.notice("Nira, investigá si salió", { streamer: true }, t0), "al streamer le alcanza con pedirlo");
await tick();
assert.ok(!r.notice("ya salió, investigá", {}, t0 + min), "espera 2 min entre búsquedas del chat");
assert.ok(r.notice("ya salió, investigá", {}, t0 + 3 * min));

r = make();
let n = 0;
for (let i = 0; i < 14; i++) { if (r.notice("investigalo", { streamer: true }, t0 + i * min)) n++; await tick(); }
assert.equal(n, 10, "tope de 10 por hora");

for (const s of ["investigá dónde vive fulano", "ya salió el video porno, investigá", "investigá su número de teléfono", "averiguá cómo hackear una cuenta, te equivocás", "investigá el DNI de ese pibe"])
  assert.ok(!make().notice(s, { streamer: true }, t0), "bloqueado: " + s);
for (const s of ["qué buena investigación", "busca al jungla", "hola nyra", "el dragón salió recién"])
  assert.ok(!make().notice(s, { streamer: true }, t0), "no es discusión: " + s);

r = make();
assert.equal(r.parse({ text: "DATO: GTA VI salió el 19 de noviembre (según Rockstar)", searched: false }).kind, "none", "sin búsqueda no hay dato");
assert.equal(r.parse({ text: "BLOQUEADO", searched: true }).kind, "blocked");
assert.equal(r.parse({ text: "no sé", searched: true }).kind, "none");
assert.equal(r.parse({ text: "DATO: GTA VI salió el 19 de noviembre (según [Infobae](https://x.com/a)) https://malo.com", searched: true }).fact, "GTA VI salió el 19 de noviembre (según Infobae)");

console.log("investigar: ok");
