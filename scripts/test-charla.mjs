// Seguir la charla sin el nombre + el audio de Discord por marca de tiempo (sin red): node scripts/test-charla.mjs
import assert from "node:assert/strict";
import { Readable } from "node:stream";
import OpusScript from "opusscript";
import * as voice from "@discordjs/voice";
import { FollowUp } from "../follow-up.js";
import { readPhrase, keepTiming } from "../discord-voice.js";

let asked = 0, answer = "SI";
const f = new FollowUp({ ask: async () => { asked++; return answer; } });
const t0 = 1e12;
assert.equal(await f.isReply("A", "sí, dale", [], t0), false, "sin que ella haya hablado no hay ventana");
f.speaking("¿Van al dragón o al barón?");
assert.equal(await f.isReply("A", "al dragón obvio", [], t0), true, "le pueden contestar mientras habla");
f.finished(t0);
assert.equal(await f.isReply("A", "al dragón", [], t0 + 29000), true);
asked = 0;
assert.equal(await f.isReply("A", "al dragón", [], t0 + 31000), false, "pasados 30 s hace falta el nombre");
assert.equal(asked, 0, "fuera de la ventana no gasta Haiku");
assert.equal(await f.isReply("A", "van al dragón o al barón", [], t0 + 1000), false, "su propio eco");
assert.equal(asked, 0);
for (const no of ["NO", "No.", "Sinceramente no", ""]) { answer = no; assert.equal(await f.isReply("A", "obvio", [], t0 + 1000), false, no); }
f.ask = async () => { throw new Error("caído"); };
assert.equal(await f.isReply("A", "obvio", [], t0 + 1000), false, "si falla se calla");
f.ask = () => new Promise(r => setTimeout(() => r("SI"), 50));
assert.deepEqual(await Promise.all(["uno", "dos", "tres"].map(s => f.isReply("A", s, [], t0 + 1000))), [true, true, false], "como mucho dos decisiones a la vez");

// audio
const enc = new OpusScript(16000, 1);
const frame = Buffer.from(enc.encode(Buffer.alloc(640, 9), 320));
enc.delete();
const packets = ts => ts.map(t => { const p = Buffer.from(frame); p.ts = (t * 48) >>> 0; return p; });
const ms = async ts => (await readPhrase(Readable.from(packets(ts)))).pcm.length / 32;
assert.equal(await ms([0, 20, 40, 60]), 80, "seguidos, sin silencio inventado");
assert.equal(await ms([0, 20, 420, 440]), 460, "la pausa vuelve exacta aunque llegue en ráfaga");
assert.equal(await ms([0, 20, 60]), 80, "un paquete perdido deja su lugar");
assert.equal(await ms([0, 20, 20, 40]), 60, "uno repetido se ignora");
assert.equal(await ms([90000, 90020, 5, 25]), 80, "si el reloj se reinicia sigue oyendo");

const receiver = new voice.VoiceReceiver({});
receiver.parsePacket = msg => { if (msg[0] === 0) throw new Error("no descifra"); return Buffer.from(frame); };
keepTiming(receiver);
assert.equal(receiver.parsePacket(Buffer.alloc(16)), null, "uno que no se descifra no tira la frase");
const good = Buffer.alloc(16, 1); good.writeUInt32BE(4800, 4);
assert.equal(receiver.parsePacket(good).ts, 4800);

console.log("charla y audio: ok");
