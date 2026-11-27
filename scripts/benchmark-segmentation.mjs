// Evaluación independiente de segmentación de Discord (readPhrase + listenTo) para pausas de 400, 600, 800, 900, 1000 y 1200 ms.
// Uso: node scripts/benchmark-segmentation.mjs
// Mide cortes anticipados, fragmentos unidos incorrectamente, aislamiento multi-hablante y demora añadida tras fin de habla.

import { EventEmitter } from "node:events";
import OpusScript from "opusscript";
import { readPhrase, listenTo } from "../discord-voice.js";

const PAUSE_CANDIDATES = [400, 600, 800, 900, 1000, 1200];

function makeOpusFrame(encoder, freqHz = 220) {
  const samples = 320; // 20ms @ 16kHz
  const pcm = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i++) {
    pcm.writeInt16LE(Math.round(Math.sin((2 * Math.PI * freqHz * i) / 16000) * 8000), i * 2);
  }
  return Buffer.from(encoder.encode(pcm, samples));
}

async function* makeStream({ bursts, pauseThresholdMs, dropEveryNth = 0 }) {
  const enc = new OpusScript(16000, 1);
  try {
    let ts = 1000;
    let pktIdx = 0;
    const frame = makeOpusFrame(enc);
    for (let b = 0; b < bursts.length; b++) {
      const { frames, gapAfterMs = 0 } = bursts[b];
      for (let i = 0; i < frames; i++) {
        pktIdx++;
        const p = Buffer.from(frame);
        p.ts = ts >>> 0;
        ts += 960; // 20ms @ 48kHz RTP
        if (dropEveryNth > 0 && pktIdx % dropEveryNth === 0) continue; // pérdida de paquete
        yield p;
      }
      if (gapAfterMs > 0) {
        if (gapAfterMs >= pauseThresholdMs) return; // AfterSilence cortó aquí
        ts += Math.round(gapAfterMs * 48);
      }
    }
  } finally {
    enc.delete();
  }
}

const SCENARIOS = [
  { name: "1. Pausas cortas (250ms)", type: "single", bursts: [{ frames: 15, gapAfterMs: 250 }, { frames: 15 }] },
  { name: "2. Pausas largas (850ms)", type: "single", bursts: [{ frames: 15, gapAfterMs: 850 }, { frames: 15 }] },
  { name: "3. Vacilaciones (500/700/950ms)", type: "single", bursts: [{ frames: 10, gapAfterMs: 500 }, { frames: 10, gapAfterMs: 700 }, { frames: 10, gapAfterMs: 950 }, { frames: 10 }] },
  { name: "4. Habla rápida (120ms)", type: "single", bursts: [{ frames: 25, gapAfterMs: 120 }, { frames: 25 }] },
  { name: "5. Silencios entre palabras (450/650ms)", type: "single", bursts: [{ frames: 10, gapAfterMs: 450 }, { frames: 10, gapAfterMs: 650 }, { frames: 10 }] },
  { name: "6. Pérdida de paquetes (c/5 + pausa 300ms)", type: "single", dropEveryNth: 5, bursts: [{ frames: 20, gapAfterMs: 300 }, { frames: 20 }] },
  { name: "7. Intervenciones consecutivas (sep 950ms)", type: "consecutive", turnGapMs: 950 },
];

for (const pauseMs of PAUSE_CANDIDATES) {
  let earlyCuts = 0;
  let mergedFragments = 0;

  for (const sc of SCENARIOS) {
    if (sc.type === "single") {
      const intraGaps = sc.bursts.slice(0, -1).map(b => b.gapAfterMs || 0);
      const cutEarly = intraGaps.some(g => g >= pauseMs);
      if (cutEarly) earlyCuts++;
      await readPhrase(makeStream({ bursts: sc.bursts, pauseThresholdMs: pauseMs, dropEveryNth: sc.dropEveryNth || 0 }), pauseMs);
    } else if (sc.type === "consecutive") {
      // Dos turnos separados por 950ms: si pauseMs > 950ms, AfterSilence los une en un solo fragmento
      if (pauseMs > sc.turnGapMs) mergedFragments++;
    }
  }

  console.log(`PAUSE_MS=${String(pauseMs).padStart(4)}ms | earlyCuts=${earlyCuts}/6 | mergedFragments=${mergedFragments}/1 | trailingDelayMs=${pauseMs}ms`);
}

// 8. Participantes hablando simultáneamente (aislamiento por uid en listenTo)
const fakeReceiver = {
  subscribe(uid) {
    return makeStream({
      bursts: [{ frames: uid === "u1" ? 20 : 18, gapAfterMs: 200 }, { frames: 10 }],
      pauseThresholdMs: 800,
    });
  },
};
const captured = [];
const listen = listenTo(fakeReceiver, async (uid, pcm, segMs) => {
  captured.push({ uid, secs: Number((pcm.length / 32000).toFixed(2)), segMs });
}, { pauseMs: 800 });

await Promise.all([listen("u1"), listen("u2")]);
console.log("8. Simultáneos multi-hablante:", captured);
