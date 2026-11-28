import test from "node:test";
import assert from "node:assert/strict";

import { matchChamp, MATCH_OVER_RE } from "../wildrift.js";

test("the player's champion is read from what vision described", () => {
  for (const [text, champ] of [["El jugador juega Lux en la línea del medio", "Lux"], ["Itami, con Kayn, queda cerca del Blue", "Kayn"],
    ["El jugador juega Kai'Sa y farmea", "Kai'Sa"], ["juega Lee Sin en jungla", "Lee Sin"], ["maneja a Dr. Mundo", "Dr. Mundo"]]) {
    assert.equal(matchChamp(text), champ, text);
  }
  for (const text of ["se ve el minimapa", "pelea contra Ashe", "juega Luxor"]) assert.equal(matchChamp(text), null, text);
});

test("the locked champion is released when the match ends", () => {
  assert.ok(MATCH_OVER_RE.test("La partida terminó en derrota: 9 a 25"));
  assert.ok(MATCH_OVER_RE.test("Itami está en el menú principal de Wild Rift"));
  assert.ok(!MATCH_OVER_RE.test("perdió una pelea en el río"));
});
