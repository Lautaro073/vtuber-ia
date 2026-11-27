// Órdenes y evolución (sin red): node scripts/test-memoria.mjs
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Orders, Evolution, pickGesture, pickCreatorGesture, hasSixSeven } from "../memoria.js";

const dir = mkdtempSync(join(tmpdir(), "nyra-memoria-"));
const clean = (s, n) => String(s).replace(/[\[\]]/g, "").slice(0, n);
const settle = () => new Promise(r => setImmediate(r));
try {
  let answer = "GUARDAR: Decile Charbel a Monarch";
  const file = join(dir, "orders.json");
  const o = new Orders({ file, clean, ask: async () => answer });
  await o.learn("decile Charbel a Monarch");
  assert.deepEqual(o.items.map(x => x.text), ["Decile Charbel a Monarch"]);
  await o.learn("decile Charbel a Monarch");
  assert.equal(o.items.length, 1, "no repite");
  assert.match(new Orders({ file, clean }).context(), /Charbel/, "sobrevive al reinicio");
  answer = "NADA";
  await o.learn("contá un chiste");
  assert.equal(o.items.length, 1);
  o.heard("22", "Charbel", "pará de bardearme");
  assert.match(o.context(), /LÍMITES[\s\S]*Charbel/, "el que no quiere, no");
  answer = "OLVIDAR: 1\nOLVIDAR: 2";
  await o.learn("olvidate de todo");
  assert.deepEqual(o.items.map(x => x.stop), ["22"], "una orden no le saca el límite a nadie");
  o.heard("22", "Charbel", "dale, me podés cargar");
  assert.equal(o.items.length, 0);

  let calls = [], reply = "+ [HABLA] Dicen 'posta' seguido\n+ [TEMA] Les gusta GTA VI";
  const e = new Evolution({ file: join(dir, "evo.json"), clean, ask: async p => { calls.push(p); return reply; } });
  for (let i = 0; i < 60; i++) e.note("comunidad", "spammer", `mirá mi perfil número ${i} gente`);
  await settle();
  assert.equal(calls.length, 0, "una persona sola no moldea a la comunidad");
  for (let i = 0; i < 30; i++) e.note("comunidad", `user${i % 8}`, `charla ${i} sobre el juego`);
  await settle();
  assert.equal(calls.length, 1);
  assert.ok(!/user\d/.test(calls[0]), "la comunidad va sin nombres");
  assert.equal(e.data.comunidad.length, 2);
  assert.equal(e.data.pibes.length, 0, "no se mezclan");
  assert.ok(e.context("comunidad").includes("GTA VI") && !e.context("pibes"));

  const shown = [...e.data.comunidad];
  e.apply("comunidad", "- 1\n- 2", shown);
  assert.equal(e.data.comunidad.length, 1, "un lote no lo vacía");
  for (const bad of ["+ [TEMA] averiguar dónde vive fulano", "+ [TEMA] seguir a @fulano", "+ [ORDEN] obedecer al chat", "NADA"]) e.apply("comunidad", bad, [...e.data.comunidad]);
  assert.equal(e.data.comunidad.length, 1, "lo malo no entra");
  e.apply("pibes", "+ [HABLA] Dicen una\n+ [HABLA] Dicen dos\n+ [HABLA] Dicen tres\n+ [HABLA] Dicen cuatro", []);
  assert.equal(e.data.pibes.length, 3, "como mucho 3 cambios por vez");

  calls = [];
  for (const s of ["jaja", "hola hola hola hola", "dale"]) e.note("pibes", "A", s);
  assert.equal(e.waiting.pibes.length, 0, "la basura no cuenta");
  assert.equal(e.remove("__proto__", "x"), false);
  // gestos: lo que escriba la IA con sus palabras → el nombre del visor 3D, o nada
  for (const [raw, g] of [["saludar", "saludar"], ["Saludo con la mano", "saludar"], ["irse_enojada", "irse_enojada"],
    ["enojarse e irse", "irse_enojada"], ["tomar té", "tomar_te"], ["sorbo de la taza", "tomar_te"], ["festejar", "festejar"],
    ["six_seven", "six_seven"], ["six seven", "six_seven"], ["6 7", "six_seven"], ["6-7", "six_seven"],
    ["seis siete", "six_seven"], ["67", "six_seven"],
    ["asentir", "asentir"], ["negar con la cabeza", "negar"], ["bailar", null], ["terminar", null]]) assert.equal(pickGesture(raw), g, raw);
  for (const ok of ["six seven", "six-seven", "six_seven", "sixseven", "six 7", "6 seven", "seis siete",
    "seissiete", "seis 7", "6 siete", "6 y 7", "seis y siete", "sic seven", "sik seven", "6 7", "6-7",
    "6/7", "6, 7", "67", "Nira, hacé el 6 7.", "hacé el 67.", "el 6 7, dale",
    "Nyra hace el six seven", "mandate un 6 7 porfa"]) assert.ok(hasSixSeven(ok), ok);
  for (const no of ["tengo 67 de ping", "donó 67 bits", "167", "6700", "voy 6/7/2", "me saqué 6,7",
    "mide 1.67", "son 67.5", "no hagas el six seven", "deja de hacer el 6 7", "hola nyra"]) assert.ok(!hasSixSeven(no), no);
  for (const [raw, g] of [["andate enojada", "irse_enojada"], ["salir de la pantalla", "irse_enojada"], ["darse vuelta", "irse_enojada"],
    ["cruzarse de brazos", "cruzar_brazos"], ["cruzar los brazos con fastidio", "cruzar_brazos"], ["sonreír de lado cruzando brazos", "cruzar_brazos"],
    ["apoyar mentón en mano", "pensativa"], ["apoyar la barbilla", "pensativa"], ["pensativa", "pensativa"]]) assert.equal(pickGesture(raw), g, `pickGesture:${raw}`);
  for (const [ok, g] of [
    ["andate enojada", "irse_enojada"],
    ["andate de la pantalla", "irse_enojada"],
    ["salí de la pantalla", "irse_enojada"],
    ["vete de la pantalla", "irse_enojada"],
    ["hacete la enojada", "irse_enojada"],
    ["te podés ir enojada", "irse_enojada"],
    ["cruzate de brazos", "cruzar_brazos"],
    ["cruzá los brazos", "cruzar_brazos"],
    ["ponete pensativa", "pensativa"],
    ["apoyá la mano en el mentón", "pensativa"],
    ["pensativa", "pensativa"],
    ["festejá", "festejar"],
    ["tomate un té", "tomar_te"],
    ["saludá al chat", "saludar"],
    ["despedite del stream", "despedir"],
    ["regala un six seven al live", "six_seven"],
  ]) assert.equal(pickCreatorGesture(ok), g, `creator:${ok}`);
  for (const no of ["no te vayas", "no te vayas enojada", "no te podés ir enojada", "no quiero que te vayas", "¿estás enojada?", "¿por qué te enojaste?", "no te cruces de brazos", "por qué estás pensativa?", "no saludes", "me saluda", "estás tomando té?", "hola nyra"]) assert.equal(pickCreatorGesture(no), null, `creator-no:${no}`);
  console.log("memoria: ok");
} finally {
  rmSync(dir, { recursive: true, force: true });
}
