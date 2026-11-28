// Benchmark OFFLINE de la Fase 3 (superposición LLM → TTS). Determinista.
//   node scripts/sim-llm-tts-overlap.mjs              → respuestas tipo (sintéticas)
//   node scripts/sim-llm-tts-overlap.mjs --from-logs  → además, largos reales de respuestas de NYRA leídos de logs/
//                                                       (solo se usan cantidades de caracteres en memoria; no se guarda texto)
// TIEMPOS DEL LLM: SINTÉTICOS (1er token a 1.5 s, texto emitido en forma lineal hasta T_llm).
// MODELO DEL TTS: calibrado con las trazas REALES de SGLang (tests/fixtures/sglang-arrival-traces.json): no es una medición nueva.
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { AdaptiveStartPolicy, WholeFilePolicy, simulatePlayback } from "../public/tts-stream/buffer-policy.js";
import { findFirstSentence } from "../lib/tts/llm-tts-overlap.js";

const fixture = JSON.parse(readFileSync(new URL("../tests/fixtures/sglang-arrival-traces.json", import.meta.url), "utf8"));
const med = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1) + 0.5))]; };

// ── Calibración del modelo TTS con trazas SGLang (streaming, mediciones, perfil compile/R20) ──
const cal = fixture.traces.filter(t => t.engine === "sglang-omni" && t.streaming && t.kind === "measure" && t.trace.length >= 3);
export const TTS_MODEL = {
  ttfa: med(cal.map(t => t.trace[0][0])),
  firstBlock: med(cal.map(t => t.trace[0][1])),
  block: med(cal.flatMap(t => t.trace.slice(1, -1).map((p, i) => p[1] - t.trace[i][1]))),
  interval: med(cal.flatMap(t => t.trace.slice(1, -1).map((p, i) => p[0] - t.trace[i][0]))),
  // el último bloque (parcial) llega antes que uno completo: proporción observada tiempo/audio respecto de un bloque entero
  finalRatio: med(cal.map(t => { const n = t.trace.length; const dt = t.trace[n - 1][0] - t.trace[n - 2][0]; const da = t.trace[n - 1][1] - t.trace[n - 2][1]; return (dt / da) / (med([3.17]) / 2.09); })),
};

export function ttsArrivals(start, audio, { slow = 1 } = {}) {
  const m = TTS_MODEL;
  const ev = [];
  let t = start + m.ttfa * slow, a = Math.min(audio, m.firstBlock);
  ev.push([t, a]);
  while (a < audio - 1e-9) {
    const b = Math.min(m.block, audio - a);
    t += (b >= m.block - 1e-9 ? m.interval : Math.max(0.15, b * (m.interval / m.block) * m.finalRatio)) * slow;
    a += b;
    ev.push([t, a]);
  }
  return ev;
}

const PROD_CPS = 14;   // estimador de producción (Fase 2): conservador
const TRUE_CPS = 16;   // velocidad de habla de las trazas (mediana ~15–18 cps)

// Una respuesta: D s de audio, la 1.ª oración es la fracción f del texto (null = una sola oración). LLM: T0 → Tllm.
// Incluye las guardas reales del controlador: delaySec = 0.25 s antes de lanzar y minHeadStartSec = 1.0 s mínimo.
export function simulateTurn({ D, f, Tllm, T0 = 1.5, slow = 1, delaySec = 0.25, minHeadStartSec = 1.0 }) {
  const est = D * TRUE_CPS / PROD_CPS;
  const gate = (start, seg = {}) => { const p = new AdaptiveStartPolicy({ estDurationSec: est, ...seg }); return { onArrival: (t, a) => p.onArrival(t, a), onEnd: (t, a) => p.onEnd(t, a), onUnderrun: () => p.onUnderrun(), shouldPlay: (t, x) => t >= start && p.shouldPlay(t, x) }; };
  const full = ttsArrivals(Tllm, D, { slow });
  const wav = simulatePlayback(full, new WholeFilePolicy());
  const stream = simulatePlayback(full, gate(Tllm));
  const out = {
    wav: { start: wav.firstAudibleSec, end: wav.endSec, requests: 1, cuts: 0, cutSec: 0, gpu: full.at(-1)[0] - Tllm },
    stream: { start: stream.firstAudibleSec, end: stream.endSec, requests: 1, cuts: stream.underruns, cutSec: stream.underrunSec, gpu: full.at(-1)[0] - Tllm },
  };
  // "Anticipada de respuesta completa": el texto completo recién existe al terminar el LLM → igual que stream
  out.fullAhead = { ...out.stream, note: "el texto completo no existe antes del fin del LLM" };
  if (f === null || f >= 1) { out.firstSentence = { ...out.stream, requests: 1, note: "una sola oración: no hay qué superponer" }; return out; }
  const tSentence = T0 + f * (Tllm - T0);
  const t1 = tSentence + delaySec;
  if (t1 >= Tllm) {
    out.firstSentence = { ...out.stream, requests: 1, note: "texto completo antes del retardo de 250ms" };
    return out;
  }
  const a1 = ttsArrivals(t1, D * f, { slow });
  if (Tllm - t1 < minHeadStartSec) {
    out.firstSentence = { ...out.stream, requests: 1, wastedIfDiscarded: Math.min(a1.at(-1)[0], Tllm) - t1, note: "ventaja < 1 s: descartada" };
    return out;
  }
  const t2 = Math.max(a1.at(-1)[0], Tllm);
  const a2 = ttsArrivals(t2, D * (1 - f), { slow }).map(([t, a]) => [t, a + D * f]);
  const comb = [...a1, ...a2];
  const gpu = (a1.at(-1)[0] - t1) + (a2.at(-1)[0] - t2);
  const naive = simulatePlayback(comb, gate(Tllm));
  out.firstSentenceNaive = { start: naive.firstAudibleSec, end: naive.endSec, requests: 2, cuts: naive.underruns, cutSec: naive.underrunSec, gpu };
  // la política conoce el límite entre pedidos y mide el TTFA del 2.º pedido desde que realmente arranca (t2)
  const seg = simulatePlayback(comb, gate(Tllm, { segmentBoundarySec: D * f, segmentGapSec: TTS_MODEL.ttfa * slow, segmentStartedAt: t2 }));
  out.firstSentence = { start: seg.firstAudibleSec, end: seg.endSec, requests: 2, cuts: seg.underruns, cutSec: seg.underrunSec, gpu, wastedIfDiscarded: a1.at(-1)[0] - t1 };
  return out;
}

export const TYPES = [
  { name: "corta (1 oración, 2.8 s)", D: 2.8, f: null },
  { name: "mediana (5.2 s, 1.ª = 50 %)", D: 5.2, f: 0.5 },
  { name: "emocional (4.8 s, 1.ª = 50 %)", D: 4.8, f: 0.5 },
  { name: "típica NYRA (8.6 s, 1.ª = 45 %)", D: 8.6, f: 0.45 },   // 137 car. (mediana de logs) / 16 cps
  { name: "larga (16.9 s, 1.ª = 25 %)", D: 16.9, f: 0.25 },
];

function realLengths() {
  const dir = fileURLToPath(new URL("../logs/discord/", import.meta.url));
  if (!existsSync(dir)) return [];
  const out = [];
  for (const f of readdirSync(dir).filter(x => x.endsWith(".jsonl"))) {
    for (const line of readFileSync(join(dir, f), "utf8").split("\n")) {
      try {
        const o = JSON.parse(line);
        if (o.who !== "NYRA" || !o.meta?.latencyMs || !o.text) continue;
        const fs = findFirstSentence(o.text);
        out.push({ D: o.text.length / TRUE_CPS, f: fs ? fs.segment.length / o.text.length : null }); // solo números
      } catch { }
    }
  }
  return out;
}

const isMain = process.argv[1]?.replace(/\\/g, "/").endsWith("scripts/sim-llm-tts-overlap.mjs");
if (isMain) {
  const r2 = v => Number(v).toFixed(2);
  console.log("Modelo TTS calibrado con trazas SGLang:", Object.fromEntries(Object.entries(TTS_MODEL).map(([k, v]) => [k, r2(v)])));
  console.log("LLM: SINTÉTICO (1er token 1.5 s, texto lineal). Inicio = segundos desde que llega el mensaje hasta el audio.\n");
  for (const slow of [1, 1.3]) {
    for (const Tllm of [3, 7, 12]) {
      console.log(`### LLM ${Tllm} s${slow > 1 ? `, TTS 30 % más lento` : ""}`);
      console.log("| Respuesta | WAV inicio/fin | Fase 2 inicio/fin | 1.ª or. sin límite: inicio (cortes) | 1.ª oración con límite inicio/fin | cortes | pedidos | GPU extra | GPU perdida si se descarta |");
      console.log("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
      for (const ty of TYPES) {
        const r = simulateTurn({ D: ty.D, f: ty.f, Tllm, slow });
        const s = r.firstSentence, nv = r.firstSentenceNaive;
        console.log(`| ${ty.name} | ${r2(r.wav.start)} / ${r2(r.wav.end)} | ${r2(r.stream.start)} / ${r2(r.stream.end)} | ${nv ? `${r2(nv.start)} (${nv.cuts}, ${r2(nv.cutSec)} s)` : "—"} | ${r2(s.start)} / ${r2(s.end)} | ${s.cuts} (${r2(s.cutSec)} s) | ${s.requests} | ${r2(s.gpu - r.stream.gpu)} s | ${s.wastedIfDiscarded ? r2(s.wastedIfDiscarded) + " s" : "—"} |`);
      }
      console.log("");
    }
  }
  if (process.argv.includes("--from-logs")) {
    const lens = realLengths();
    console.log(`### Largos REALES de respuestas (logs, ${lens.length} respuestas; ${lens.filter(x => x.f !== null).length} con 1.ª oración segmentable) × LLM sintético`);
    console.log("| LLM | Fase 2 inicio p50/p95 | 1.ª oración inicio p50/p95 | Ahorro p50 | % con cortes | Fin p50 (Fase 2 → 1.ª or.) |");
    console.log("|---|---:|---:|---:|---:|---:|");
    for (const Tllm of [3, 7, 12]) {
      const rs = lens.map(l => simulateTurn({ D: l.D, f: l.f, Tllm }));
      const a = rs.map(r => r.stream.start), b = rs.map(r => r.firstSentence.start);
      const sv = rs.map(r => r.stream.start - r.firstSentence.start);
      const cut = rs.filter(r => r.firstSentence.cuts > 0).length;
      console.log(`| ${Tllm} s | ${r2(pct(a, .5))} / ${r2(pct(a, .95))} | ${r2(pct(b, .5))} / ${r2(pct(b, .95))} | ${r2(pct(sv, .5))} s | ${Math.round(100 * cut / rs.length)} % | ${r2(pct(rs.map(r => r.stream.end), .5))} → ${r2(pct(rs.map(r => r.firstSentence.end), .5))} |`);
    }
  }
}
