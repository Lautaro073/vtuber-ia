// Compara políticas de reproducción sobre trazas REALES de llegada PCM (tests/fixtures/sglang-arrival-traces.json).
// Uso: node scripts/sim-buffer-policies.mjs            → evaluación causal (como en producción)
//      node scripts/sim-buffer-policies.mjs --oracle   → además, referencias que usan la duración real (NO causales)
// Simulación offline determinista: reproduce tiempos registrados (L4, loopback); no mide GPU, red, gateway ni OBS.
import { readFileSync } from "node:fs";
import { AdaptiveStartPolicy, FixedStartPolicy, WholeFilePolicy, simulatePlayback, oracleStartSec } from "../public/tts-stream/buffer-policy.js";

export const fixture = JSON.parse(readFileSync(new URL("../tests/fixtures/sglang-arrival-traces.json", import.meta.url), "utf8"));
// Solo SGLang-Omni (no el servidor original de Fish), streaming, sin warmups, sin WAV de un único punto ni la demo de split
export const evalPool = fixture.traces.filter(t => t.engine === "sglang-omni" && t.streaming === true && t.kind === "measure");
export const withChars = evalPool.filter(t => Number.isFinite(t.chars) && t.chars > 0);

const PRIOR_CPS = 14;   // el mismo valor inicial que SglangStreamManager
const learn = (cps, chars, audioSec) => Math.max(8, Math.min(25, 0.7 * cps + 0.3 * (chars / audioSec))); // idem servidor
const audioOf = t => t.trace[t.trace.length - 1][1];
const pct = (arr, p) => { if (!arr.length) return null; const a = [...arr].sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * (a.length - 1) + 0.5))]; };

// Estimadores causales: la estimación de una sesión solo usa su texto y las sesiones ANTERIORES de la secuencia
export const ESTIMATORS = {
  "14 cps fijo": () => ({ estimate: t => t.chars / PRIOR_CPS, observe() {} }),
  "aprendido conservador ≤14 cps (PRODUCCIÓN)": () => { let cps = PRIOR_CPS; return { estimate: t => t.chars / Math.min(cps, PRIOR_CPS), observe: t => { cps = learn(cps, t.chars, audioOf(t)); } }; },
  "aprendido libre (sin historial)": () => { let cps = PRIOR_CPS; return { estimate: t => t.chars / cps, observe: t => { cps = learn(cps, t.chars, audioOf(t)); } }; },
  "aprendido, error −30 %": () => { let cps = PRIOR_CPS; return { estimate: t => 0.7 * t.chars / cps, observe: t => { cps = learn(cps, t.chars, audioOf(t)); } }; },
  "aprendido, error +30 %": () => { let cps = PRIOR_CPS; return { estimate: t => 1.3 * t.chars / cps, observe: t => { cps = learn(cps, t.chars, audioOf(t)); } }; },
};

// Recorre la secuencia en el orden del fixture (orden de ejecución de los benchmarks). Las primeras `calib` sesiones
// solo alimentan al estimador y no se cuentan (separación calibración / evaluación).
export function runCausal(list, estimatorName, { calibFraction = 0.25 } = {}) {
  const est = ESTIMATORS[estimatorName]();
  const calib = Math.floor(list.length * calibFraction);
  const out = [];
  list.forEach((t, i) => {
    const r = simulatePlayback(t.trace, new AdaptiveStartPolicy({ estDurationSec: est.estimate(t) }));
    if (i >= calib) out.push({ t, r });
    est.observe(t); // recién después de terminar la sesión se aprende de ella
  });
  return out;
}

export function summarize(rows) {
  const wavStart = new Map(rows.map(({ t }) => [t, simulatePlayback(t.trace, new WholeFilePolicy()).firstAudibleSec]));
  const starts = rows.map(x => x.r.firstAudibleSec);
  const savings = rows.map(x => wavStart.get(x.t) - x.r.firstAudibleSec);
  const withCut = rows.filter(x => x.r.underruns > 0).length;
  return {
    n: rows.length,
    p50: pct(starts, 0.5), p95: pct(starts, 0.95),
    savingP50: pct(savings, 0.5), wavP50: pct([...wavStart.values()], 0.5),
    pctWithCut: rows.length ? Math.round(100 * withCut / rows.length) : 0,
    cuts: rows.reduce((s, x) => s + x.r.underruns, 0),
    cutSec: Number(rows.reduce((s, x) => s + x.r.underrunSec, 0).toFixed(2)),
  };
}

function baseline(list, make) {
  return list.map(t => ({ t, r: simulatePlayback(t.trace, make(t)) }));
}

const fmt = v => (v === null || v === undefined ? "—" : typeof v === "number" ? v.toFixed(2) : v);
function printRow(name, s) {
  console.log(`| ${name} | ${s.n} | ${fmt(s.p50)} s | ${fmt(s.p95)} s | ${fmt(s.savingP50)} s | ${s.pctWithCut} % | ${s.cuts} | ${fmt(s.cutSec)} s |`);
}

const isMain = process.argv[1]?.replace(/\\/g, "/").endsWith("scripts/sim-buffer-policies.mjs");
if (isMain) {
  const oracle = process.argv.includes("--oracle");
  console.log(`Fixture: ${fixture.dedup}`);
  console.log(`Evaluables (streaming, sin warmup): ${evalPool.length}; con largo de texto registrado: ${withChars.length}. Calibración: primer 25 % de cada caso (no contado).\n`);
  const cases = ["short", "medium", "emotion", "long"];
  for (const kase of [...cases, "TODOS"]) {
    const list = kase === "TODOS" ? withChars : withChars.filter(t => t.case === kase);
    if (!list.length) continue;
    console.log(`## ${kase} (${list.length} trazas, audio medio ${(list.reduce((s, t) => s + audioOf(t), 0) / list.length).toFixed(1)} s)`);
    console.log("| Política | N eval | Inicio p50 | Inicio p95 | Ahorro vs WAV (p50) | % con corte | Cortes | Tiempo cortado |");
    console.log("|---|---:|---:|---:|---:|---:|---:|---:|");
    const calib = Math.floor(list.length * 0.25);
    const evalList = list.slice(calib);
    printRow("WAV completo", summarize(baseline(evalList, () => new WholeFilePolicy())));
    printRow("Inmediato", summarize(baseline(evalList, () => new FixedStartPolicy({ bufferSec: 0.01 }))));
    printRow("Fijo 0.5 s", summarize(baseline(evalList, () => new FixedStartPolicy({ bufferSec: 0.5 }))));
    printRow("Fijo 1 s", summarize(baseline(evalList, () => new FixedStartPolicy({ bufferSec: 1 }))));
    printRow("Fijo 2 s", summarize(baseline(evalList, () => new FixedStartPolicy({ bufferSec: 2 }))));
    for (const name of Object.keys(ESTIMATORS)) printRow(`Adaptativa — ${name}`, summarize(runCausal(list, name)));
    if (oracle) {
      printRow("*Ref. NO causal: duración real exacta*", summarize(baseline(evalList, t => new AdaptiveStartPolicy({ estDurationSec: audioOf(t) }))));
      const o = evalList.map(t => oracleStartSec(t.trace));
      console.log(`| *Óptimo retrospectivo (inalcanzable)* | ${evalList.length} | ${fmt(pct(o, 0.5))} s | ${fmt(pct(o, 0.95))} s | — | 0 % | 0 | 0 s |`);
    }
    console.log("");
  }
}
