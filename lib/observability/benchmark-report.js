// Fase 4A: agregación estadística, exportación (JSON/JSONL/CSV/Markdown/HTML) y visualización de líneas temporales.
// Principios estadísticos:
//  - Nunca mezcla categorías heterogéneas (mock / simulation / historical_replay / isolated_browser / real_gateway / real_obs).
//  - Separa warmups (cold start) de mediciones calientes, y nunca oculta fallos, parciales, fallbacks ni cancelaciones.
//  - Documenta tamaño muestral, método de percentiles (linear_interpolation_R7) e inestabilidad de p95 con N < 20.
//  - Distingue explícitamente diferencia de medianas vs. mediana de diferencias pareadas (solo en pares equivalentes).
import { mkdirSync, writeFileSync } from "node:fs";
import { isAbsolute, join, relative, resolve } from "node:path";
import { TRACE_SCHEMA_VERSION, VALID_CATEGORIES, sanitizeValue } from "./turn-tracer.js";

// Política conservadora para endpoints de diagnóstico (/api/diagnostics/traces*).
// No altera la autenticación general del panel: cuando PANEL_PASSWORD está definida, /api ya validó x-panel-key;
// cuando PANEL_PASSWORD NO está definida, solo permite acceso local loopback verificado y sin cabeceras de proxy.
export function isLoopbackAddress(rawAddr) {
  const s = String(rawAddr || "").trim().toLowerCase();
  return s === "127.0.0.1" || s === "::1" || s === "::ffff:127.0.0.1";
}

export function checkDiagnosticsAccess({ panelPassword = "", remoteAddress = "", forwarded = "" } = {}) {
  if (panelPassword) return { ok: true, mode: "panel_password" };
  if (forwarded) {
    return { ok: false, status: 403, error: "Diagnóstico bloqueado tras proxy sin PANEL_PASSWORD configurada" };
  }
  if (!isLoopbackAddress(remoteAddress)) {
    return { ok: false, status: 403, error: "Diagnóstico restringido a loopback local cuando PANEL_PASSWORD no está configurada" };
  }
  return { ok: true, mode: "loopback_only" };
}

// Garantiza que una exportación siempre escriba estrictamente dentro del directorio base autorizado (sin traversal)
export function resolveAuthorizedExportDir(baseDir, runId) {
  const safeId = String(runId || "").trim();
  if (!/^[a-zA-Z0-9._-]{1,96}$/.test(safeId) || safeId.includes("..")) {
    throw new Error("Identificador de exportación inválido");
  }
  const absBase = resolve(baseDir);
  const absTarget = resolve(absBase, safeId);
  const rel = relative(absBase, absTarget);
  if (!rel || rel.startsWith("..") || isAbsolute(rel)) {
    throw new Error("Ruta de exportación fuera del directorio autorizado");
  }
  return absTarget;
}

export const PERCENTILE_METHOD = "linear_interpolation_R7";

export function percentile(values, p) {
  const arr = (values || []).filter(v => Number.isFinite(v)).sort((a, b) => a - b);
  if (!arr.length) return null;
  if (arr.length === 1) return Number(arr[0].toFixed(2));
  const clamped = Math.max(0, Math.min(1, p));
  const idx = clamped * (arr.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const frac = idx - lo;
  const v = arr[lo] + frac * (arr[hi] - arr[lo]);
  return Number(v.toFixed(2));
}

export function summarizeSeries(values) {
  const arr = (values || []).filter(v => Number.isFinite(v)).sort((a, b) => a - b);
  const n = arr.length;
  if (!n) return { n: 0, min: null, max: null, mean: null, median: null, p90: null, p95: null, p95Unstable: true };
  const mean = arr.reduce((s, x) => s + x, 0) / n;
  return {
    n,
    min: Number(arr[0].toFixed(2)),
    max: Number(arr[n - 1].toFixed(2)),
    mean: Number(mean.toFixed(2)),
    median: percentile(arr, 0.5),
    p90: percentile(arr, 0.9),
    p95: percentile(arr, 0.95),
    p95Unstable: n < 20,
  };
}

// Exige categoría homogénea para evitar mezclar mocks con mediciones reales o replays
export function assertSingleCategory(turns) {
  const cats = new Set((turns || []).map(t => t.category || "unknown"));
  if (cats.size > 1) {
    throw new Error(`No se pueden mezclar categorías distintas en un mismo agregado: ${[...cats].join(", ")}`);
  }
  const cat = [...cats][0] || null;
  if (cat && !VALID_CATEGORIES.has(cat)) {
    throw new Error(`Categoría inválida: ${cat}`);
  }
  return cat;
}

export function aggregateTurns(turns, { includeWarmups = false } = {}) {
  const list = Array.isArray(turns) ? turns : [];
  const category = assertSingleCategory(list);
  const warmups = list.filter(t => t.warmup === true);
  const measured = includeWarmups ? list : list.filter(t => !t.warmup);

  const succeeded = measured.filter(t => t.state === "completed" && !t.metrics?.partialFailure);
  const failed = measured.filter(t => t.state === "failed");
  const partialFailures = measured.filter(t => t.state === "partial_failure" || t.metrics?.partialFailure);
  const fallbacks = measured.filter(t => t.metrics?.fallbackUsed);
  const cancelled = measured.filter(t => t.state === "cancelled");
  const expired = measured.filter(t => t.state === "expired");
  const abandoned = measured.filter(t => t.state === "abandoned");

  const pick = (pool, fn) => pool.map(fn).filter(v => Number.isFinite(v));
  // Métricas de latencia se calculan sobre las ejecuciones con dato válido (exitosas + parciales con inicio),
  // pero el informe expone siempre los contadores completos de fallos/cancelaciones/fallbacks sin ocultarlos.
  const withPlayback = measured.filter(t => Number.isFinite(t.metrics?.estimatedPlaybackStartRelMs));

  const metricStats = {
    queueWaitMs: summarizeSeries(pick(measured, t => t.metrics?.queueWaitMs)),
    llmFirstTextMs: summarizeSeries(pick(measured, t => t.metrics?.llmFirstTextMs)),
    llmCompleteMs: summarizeSeries(pick(measured, t => t.metrics?.llmCompleteMs)),
    validationsMs: summarizeSeries(pick(measured, t => t.metrics?.validationsMs)),
    firstByteRelMs: summarizeSeries(pick(measured, t => t.metrics?.firstByteRelMs)),
    firstPcmRelMs: summarizeSeries(pick(measured, t => t.metrics?.firstPcmRelMs)),
    estimatedPlaybackStartRelMs: summarizeSeries(pick(withPlayback, t => t.metrics?.estimatedPlaybackStartRelMs)),
    playbackEndRelMs: summarizeSeries(pick(measured, t => t.metrics?.playbackEndRelMs)),
    turnTotalMs: summarizeSeries(pick(measured, t => t.metrics?.turnTotalMs)),
    audioDurationSec: summarizeSeries(pick(measured, t => t.metrics?.audioDurationSec)),
    httpTotalRtf: summarizeSeries(pick(measured, t => t.metrics?.httpTotalRtf)),
    ttsRequestsPerTurn: summarizeSeries(pick(measured, t => t.metrics?.ttsRequestsPerTurn)),
  };

  const turnsWithUnderruns = measured.filter(t => (t.metrics?.underruns || 0) > 0).length;
  const underrunCount = measured.reduce((s, t) => s + (t.metrics?.underruns || 0), 0);
  const underrunTotalMs = Number(measured.reduce((s, t) => s + (t.metrics?.underrunTotalMs || 0), 0).toFixed(2));
  const speculationsStarted = measured.filter(t => t.metrics?.speculationStarted).length;
  const speculationsUsed = measured.filter(t => t.metrics?.speculationUsed).length;
  const speculationsDiscarded = measured.filter(t => t.metrics?.speculationDiscarded).length;

  // Identificar cuello de botella dominante por mediana de etapa individual
  const stageMedians = [
    ["cola", metricStats.queueWaitMs.median],
    ["llm", metricStats.llmCompleteMs.median],
    ["validaciones", metricStats.validationsMs.median],
    ["tts_hasta_inicio_reproduccion", metricStats.estimatedPlaybackStartRelMs.median !== null && metricStats.llmCompleteMs.median !== null
      ? Math.max(0, Number((metricStats.estimatedPlaybackStartRelMs.median - metricStats.llmCompleteMs.median - (metricStats.queueWaitMs.median || 0)).toFixed(2)))
      : metricStats.estimatedPlaybackStartRelMs.median],
  ].filter(([, v]) => Number.isFinite(v));
  stageMedians.sort((a, b) => b[1] - a[1]);

  return {
    category,
    percentileMethod: PERCENTILE_METHOD,
    p95Unstable: measured.length < 20,
    counts: {
      totalRecorded: list.length,
      warmupsExcluded: includeWarmups ? 0 : warmups.length,
      measured: measured.length,
      succeeded: succeeded.length,
      failed: failed.length,
      partialFailures: partialFailures.length,
      fallbacks: fallbacks.length,
      cancelled: cancelled.length,
      expired: expired.length,
      abandoned: abandoned.length,
      turnsWithUnderruns,
      underrunCount,
      underrunTotalMs,
      speculationsStarted,
      speculationsUsed,
      speculationsDiscarded,
    },
    dominantBottleneck: stageMedians[0]?.[0] || null,
    metrics: metricStats,
    warmupSummary: warmups.length ? {
      n: warmups.length,
      estimatedPlaybackStartRelMs: summarizeSeries(pick(warmups, t => t.metrics?.estimatedPlaybackStartRelMs)),
    } : null,
  };
}

// Comparación rigurosa entre dos perfiles (baseline vs candidate).
// Distingue explícitamente diferencia entre medianas de mediana de diferencias pareadas (solo si pairedKey coincide).
export function compareProfiles(baselineTurns, candidateTurns, metricGetter = t => t.metrics?.estimatedPlaybackStartRelMs, { pairKey = null } = {}) {
  const baseCat = assertSingleCategory(baselineTurns);
  const candCat = assertSingleCategory(candidateTurns);
  if (baseCat && candCat && baseCat !== candCat) {
    throw new Error(`No se pueden comparar categorías distintas (${baseCat} vs ${candCat})`);
  }
  const baseMeas = baselineTurns.filter(t => !t.warmup);
  const candMeas = candidateTurns.filter(t => !t.warmup);
  const baseVals = baseMeas.map(metricGetter).filter(Number.isFinite);
  const candVals = candMeas.map(metricGetter).filter(Number.isFinite);
  const baseMedian = percentile(baseVals, 0.5);
  const candMedian = percentile(candVals, 0.5);
  const diffOfMedians = baseMedian !== null && candMedian !== null ? Number((baseMedian - candMedian).toFixed(2)) : null;

  let paired = null;
  if (typeof pairKey === "function") {
    const baseMap = new Map();
    for (const t of baseMeas) {
      const k = pairKey(t);
      const v = metricGetter(t);
      if (k != null && Number.isFinite(v)) baseMap.set(k, v);
    }
    const diffs = [];
    for (const t of candMeas) {
      const k = pairKey(t);
      const v = metricGetter(t);
      if (k != null && Number.isFinite(v) && baseMap.has(k)) {
        diffs.push(Number((baseMap.get(k) - v).toFixed(2))); // positivo = candidate ahorró tiempo
      }
    }
    if (diffs.length) {
      paired = {
        n: diffs.length,
        medianOfPairedDiffs: percentile(diffs, 0.5),
        p95OfPairedDiffs: percentile(diffs, 0.95),
        minDiff: Math.min(...diffs),
        maxDiff: Math.max(...diffs),
        p95Unstable: diffs.length < 20,
      };
    }
  }
  return {
    category: baseCat || candCat,
    baselineN: baseVals.length,
    candidateN: candVals.length,
    baselineMedian: baseMedian,
    candidateMedian: candMedian,
    diffOfMedians, // positivo = candidato más rápido por diferencia de medianas
    paired,        // solo presente si hubo emparejamiento explícito equivalente
  };
}

// Renderiza una línea temporal ASCII que respeta etapas paralelas (nunca las suma como consecutivas)
export function formatOverlappingTimeline(trace, { width = 56 } = {}) {
  if (!trace || !Array.isArray(trace.timeline) || !trace.timeline.length) return "(sin intervalos de línea temporal)";
  const maxMs = Math.max(1, ...trace.timeline.map(s => s.endRelMs ?? s.startRelMs ?? 0), trace.metrics?.turnTotalMs || 0);
  const lines = [
    `Turno ${trace.ids?.turnId || "?"} [${trace.category}] total=${maxMs}ms (eje horizontal 0..${Math.round(maxMs)}ms; etapas superpuestas en paralelo):`,
  ];
  for (const s of trace.timeline) {
    const a = Math.max(0, s.startRelMs ?? 0);
    const b = Math.max(a, s.endRelMs ?? a);
    const i0 = Math.min(width - 1, Math.floor((a / maxMs) * width));
    const i1 = Math.max(i0 + 1, Math.min(width, Math.ceil((b / maxMs) * width)));
    const bar = ".".repeat(i0) + "#".repeat(i1 - i0) + ".".repeat(Math.max(0, width - i1));
    const label = String(s.stage).padEnd(16);
    lines.push(`  ${label} |${bar}| ${Math.round(a)}..${Math.round(b)}ms (${Math.round(b - a)}ms)`);
  }
  return lines.join("\n");
}

function csvCell(v) {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(rows, columns = null) {
  const list = Array.isArray(rows) ? rows : [];
  if (!list.length) return "";
  const cols = columns || [...new Set(list.flatMap(r => Object.keys(r)))];
  const header = cols.map(csvCell).join(",");
  const body = list.map(r => cols.map(c => csvCell(r[c])).join(","));
  return [header, ...body].join("\n") + "\n";
}

export function turnsToCsvRows(turns, { profile = null } = {}) {
  return (turns || []).map(t => ({
    turnId: t.ids?.turnId || "",
    messageId: t.ids?.messageId || "",
    profile: profile || t.profile || "",
    category: t.category || "",
    warmup: Boolean(t.warmup),
    state: t.state || "",
    source: t.source || "",
    platform: t.platform || "",
    priority: t.priority ?? "",
    providerEffective: t.effectiveProvider || "",
    engineRequested: t.requestedEngine || "",
    engineEffective: t.effectiveEngine || "",
    playbackMode: t.playbackMode || "",
    queueWaitMs: t.metrics?.queueWaitMs ?? "",
    llmFirstTextMs: t.metrics?.llmFirstTextMs ?? "",
    llmCompleteMs: t.metrics?.llmCompleteMs ?? "",
    validationsMs: t.metrics?.validationsMs ?? "",
    firstByteRelMs: t.metrics?.firstByteRelMs ?? "",
    firstPcmRelMs: t.metrics?.firstPcmRelMs ?? "",
    estimatedPlaybackStartRelMs: t.metrics?.estimatedPlaybackStartRelMs ?? "",
    playbackEndRelMs: t.metrics?.playbackEndRelMs ?? "",
    audibleEndDiffFromHttpEndMs: t.metrics?.audibleEndDiffFromHttpEndMs ?? "",
    audioDurationSec: t.metrics?.audioDurationSec ?? "",
    httpTotalRtf: t.metrics?.httpTotalRtf ?? "",
    ttsRequestsPerTurn: t.metrics?.ttsRequestsPerTurn ?? "",
    retries429: t.metrics?.retries429 ?? 0,
    underruns: t.metrics?.underruns ?? 0,
    underrunTotalMs: t.metrics?.underrunTotalMs ?? 0,
    fallbackUsed: Boolean(t.metrics?.fallbackUsed),
    partialFailure: Boolean(t.metrics?.partialFailure),
    speculationUsed: Boolean(t.metrics?.speculationUsed),
    speculationDiscarded: Boolean(t.metrics?.speculationDiscarded),
  }));
}

export function buildRunDocument({
  runId,
  commit = "unknown",
  date = new Date().toISOString(),
  profile = "multi",
  effectiveProvider = null,
  config = {},
  environment = {},
  category = "mock",
  synthetic = null,
  dataOrigin = null,
  levels = {},
  aggregates = {},
  comparisons = [],
  traces = [],
  errors = [],
  limitations = [],
} = {}) {
  if (!VALID_CATEGORIES.has(category)) throw new Error(`Categoría inválida: ${category}`);
  const isSynthetic = synthetic !== null ? Boolean(synthetic) : (category === "mock" || category === "simulation");
  const resolvedOrigin = dataOrigin || (isSynthetic ? "synthetic_test" : "runtime_capture");
  return {
    schemaVersion: TRACE_SCHEMA_VERSION,
    runId: String(runId || `run_${Date.now()}`),
    commit: String(commit),
    date,
    profile: String(profile),
    effectiveProvider: effectiveProvider || null,
    config: sanitizeValue(config),
    environment: sanitizeValue({
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      ...environment,
    }),
    category,
    synthetic: isSynthetic,
    dataOrigin: String(resolvedOrigin),
    percentileMethod: PERCENTILE_METHOD,
    levels,
    aggregates,
    comparisons,
    traces,
    errors: (errors || []).map(e => sanitizeValue(e)),
    limitations: Array.isArray(limitations) ? limitations.map(String) : [],
  };
}

export function renderMarkdownReport(doc) {
  const lines = [
    `# NYRA — Reporte de Benchmark de Latencia (${doc.category})`,
    "",
    `- **Run ID:** \`${doc.runId}\``,
    `- **Commit:** \`${doc.commit}\``,
    `- **Fecha:** ${doc.date}`,
    `- **Perfil:** \`${doc.profile}\``,
    `- **Categoría:** \`${doc.category}\` (no mezclada con otras categorías)`,
    `- **Método de percentiles:** \`${doc.percentileMethod}\``,
    "",
  ];
  const cfgEntries = Object.entries(doc.config || {});
  if (cfgEntries.length) {
    lines.push("## Parámetros efectivos de ejecución", "");
    lines.push("| Parámetro | Valor |");
    lines.push("|---|---|");
    for (const [k, v] of cfgEntries) {
      lines.push(`| ${k} | \`${typeof v === "object" && v !== null ? JSON.stringify(v) : String(v)}\` |`);
    }
    lines.push("");
  }
  const entries = Object.entries(doc.aggregates || {});
  if (entries.length) {
    lines.push("## Comparación de perfiles", "");
    lines.push("| Perfil | N eval (warmups) | Éxitos / Fallos / Parcial / FB / Canc | Inicio est. p50 / p95 (ms) | Fin reprod. p50 (ms) | Cortes (turnos / ms) | Pedidos TTS/turno | Cuello de botella |");
    lines.push("|---|---:|---:|---:|---:|---:|---:|---|");
    for (const [name, agg] of entries) {
      const c = agg.counts || {};
      const m = agg.metrics || {};
      const p50 = m.estimatedPlaybackStartRelMs?.median ?? "—";
      const p95 = m.estimatedPlaybackStartRelMs?.p95 ?? "—";
      const p95Tag = m.estimatedPlaybackStartRelMs?.p95Unstable ? "*" : "";
      const endP50 = m.playbackEndRelMs?.median ?? "—";
      const reqMed = m.ttsRequestsPerTurn?.median ?? "—";
      lines.push(`| ${name} | ${c.measured ?? 0} (${c.warmupsExcluded ?? 0}) | ${c.succeeded ?? 0}/${c.failed ?? 0}/${c.partialFailures ?? 0}/${c.fallbacks ?? 0}/${c.cancelled ?? 0} | ${p50} / ${p95}${p95Tag} | ${endP50} | ${c.underrunCount ?? 0} (${c.turnsWithUnderruns ?? 0} / ${c.underrunTotalMs ?? 0}ms) | ${reqMed} | ${agg.dominantBottleneck || "—"} |`);
    }
    lines.push("", "_Nota: `*` indica N < 20 (p95 con baja estabilidad estadística)._", "");
  }
  if (doc.traces?.length) {
    const sample = doc.traces.find(t => !t.warmup && t.timeline?.length) || doc.traces[0];
    if (sample) {
      lines.push("## Ejemplo de línea temporal (etapas superpuestas)", "", "```text", formatOverlappingTimeline(sample), "```", "");
    }
  }
  if (doc.limitations?.length) {
    lines.push("## Limitaciones", "");
    for (const lim of doc.limitations) lines.push(`- ${lim}`);
    lines.push("");
  }
  return lines.join("\n");
}

function escapeHtml(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

export function renderHtmlReport(doc) {
  const rows = Object.entries(doc.aggregates || {}).map(([name, agg]) => {
    const c = agg.counts || {};
    const m = agg.metrics || {};
    return `<tr>
      <td><strong>${escapeHtml(name)}</strong></td>
      <td>${c.measured ?? 0} (${c.warmupsExcluded ?? 0})</td>
      <td>${c.succeeded ?? 0} / ${c.failed ?? 0} / ${c.partialFailures ?? 0} / ${c.fallbacks ?? 0} / ${c.cancelled ?? 0}</td>
      <td>${escapeHtml(m.estimatedPlaybackStartRelMs?.median ?? "—")} / ${escapeHtml(m.estimatedPlaybackStartRelMs?.p95 ?? "—")}${m.estimatedPlaybackStartRelMs?.p95Unstable ? "*" : ""}</td>
      <td>${escapeHtml(m.playbackEndRelMs?.median ?? "—")}</td>
      <td>${c.underrunCount ?? 0} (${c.underrunTotalMs ?? 0} ms)</td>
      <td>${escapeHtml(agg.dominantBottleneck || "—")}</td>
    </tr>`;
  }).join("\n");

  const sample = (doc.traces || []).find(t => !t.warmup && t.timeline?.length) || doc.traces?.[0];
  const maxMs = sample ? Math.max(1, ...sample.timeline.map(s => s.endRelMs ?? s.startRelMs ?? 0), sample.metrics?.turnTotalMs || 0) : 1;
  const gantt = sample ? sample.timeline.map(s => {
    const left = (((s.startRelMs ?? 0) / maxMs) * 100).toFixed(2);
    const width = (Math.max(1, ((s.endRelMs ?? s.startRelMs ?? 0) - (s.startRelMs ?? 0))) / maxMs * 100).toFixed(2);
    return `<div class="row"><span class="lbl">${escapeHtml(s.stage)}</span><div class="track"><div class="bar" style="left:${left}%;width:${width}%"></div></div><span class="val">${Math.round(s.startRelMs ?? 0)}..${Math.round(s.endRelMs ?? 0)} ms</span></div>`;
  }).join("\n") : "<p>Sin trazas de ejemplo.</p>";

  const cfgRows = Object.entries(doc.config || {}).map(([k, v]) =>
    `<tr><td><code>${escapeHtml(k)}</code></td><td><code>${escapeHtml(typeof v === "object" && v !== null ? JSON.stringify(v) : String(v))}</code></td></tr>`
  ).join("\n");

  return `<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>NYRA Benchmark — ${escapeHtml(doc.runId)}</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; background: #0f1218; color: #e6edf3; margin: 24px; }
  h1, h2 { color: #9ecbff; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }
  th, td { border: 1px solid #2b3240; padding: 8px 10px; text-align: left; }
  th { background: #181d27; }
  .meta { color: #9aa7b8; font-size: 13px; margin-bottom: 16px; }
  .row { display: grid; grid-template-columns: 150px 1fr 130px; gap: 10px; align-items: center; margin: 6px 0; font-size: 12px; }
  .track { position: relative; height: 14px; background: #1b2230; border-radius: 4px; overflow: hidden; }
  .bar { position: absolute; top: 0; bottom: 0; background: linear-gradient(90deg, #3b82f6, #22d3ee); border-radius: 4px; }
  .val { text-align: right; color: #9aa7b8; }
</style>
</head>
<body>
  <h1>NYRA — Benchmark de Latencia (${escapeHtml(doc.category)})</h1>
  <div class="meta">Run: <code>${escapeHtml(doc.runId)}</code> · Commit: <code>${escapeHtml(doc.commit)}</code> · Perfil: <code>${escapeHtml(doc.profile)}</code> · Fecha: ${escapeHtml(doc.date)} · Percentiles: <code>${escapeHtml(doc.percentileMethod)}</code></div>
  ${cfgRows ? `<h2>Parámetros efectivos</h2><table><thead><tr><th>Parámetro</th><th>Valor</th></tr></thead><tbody>${cfgRows}</tbody></table>` : ""}
  <h2>Perfiles</h2>
  <table>
    <thead><tr><th>Perfil</th><th>N (warmups)</th><th>OK / Fail / Parcial / FB / Canc</th><th>Inicio est. p50 / p95 (ms)</th><th>Fin p50 (ms)</th><th>Underruns</th><th>Cuello dominante</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <h2>Línea temporal de ejemplo (etapas superpuestas)</h2>
  ${gantt}
  <h2>Limitaciones</h2>
  <ul>${(doc.limitations || []).map(l => `<li>${escapeHtml(l)}</li>`).join("")}</ul>
</body>
</html>`;
}

export function exportBenchmarkRun(outDir, doc) {
  mkdirSync(outDir, { recursive: true });
  const jsonPath = join(outDir, "report.json");
  const jsonlPath = join(outDir, "traces.jsonl");
  const csvPath = join(outDir, "turns.csv");
  const mdPath = join(outDir, "summary.md");
  const htmlPath = join(outDir, "report.html");

  writeFileSync(jsonPath, JSON.stringify(doc, null, 2));
  writeFileSync(jsonlPath, (doc.traces || []).map(t => JSON.stringify(t)).join("\n") + (doc.traces?.length ? "\n" : ""));
  writeFileSync(csvPath, toCsv(turnsToCsvRows(doc.traces || [])));
  writeFileSync(mdPath, renderMarkdownReport(doc));
  writeFileSync(htmlPath, renderHtmlReport(doc));
  return { jsonPath, jsonlPath, csvPath, mdPath, htmlPath };
}

// H06: exporta trazas ya capturadas (sesión real de Fase 4B o sintética) sin pasar por el runner mock.
// Separa estrictamente por categoría y deduce perfil efectivo desde las propias trazas/configuración no sensible.
export function exportCapturedTraces(traces, {
  outDir,
  authorizedBaseDir = null,
  runId = `capture_${new Date().toISOString().replace(/[:.]/g, "-")}`,
  commit = "unknown",
  category = null,
  synthetic = false,
  dataOrigin = null,
  profile = null,
  effectiveProvider = null,
  config = {},
  limitations = [],
} = {}) {
  const list = Array.isArray(traces) ? traces : [];
  const inferredCat = category || list[0]?.category || "real_gateway";
  if (!VALID_CATEGORIES.has(inferredCat)) throw new Error(`Categoría inválida para exportación: ${inferredCat}`);
  const filtered = list.filter(t => (t.category || inferredCat) === inferredCat);
  if (filtered.length !== list.length) {
    throw new Error("No se pueden exportar juntas trazas de categorías distintas (ej. mock + real_gateway)");
  }
  const targetDir = authorizedBaseDir ? resolveAuthorizedExportDir(authorizedBaseDir, runId) : outDir;

  // Agrupar automáticamente por perfil efectivo (motor + modo + overlap) sin editar valores a mano
  const groups = new Map();
  for (const t of filtered) {
    const key = t.profile || `${t.effectiveEngine || t.requestedEngine || "unknown"}_${t.playbackMode || "wav"}${t.overlap?.enabled ? "_overlap" : ""}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(t);
  }
  const aggregates = {};
  for (const [k, arr] of groups.entries()) {
    aggregates[k] = aggregateTurns(arr);
  }
  const isSynthetic = Boolean(synthetic || inferredCat === "mock" || inferredCat === "simulation");
  const origin = dataOrigin || (isSynthetic ? "synthetic_test" : "runtime_capture");
  const doc = buildRunDocument({
    runId,
    commit,
    profile: profile || ([...groups.keys()].join(",") || "empty"),
    effectiveProvider: effectiveProvider || filtered.at(-1)?.effectiveProvider || null,
    config,
    category: inferredCat,
    synthetic: isSynthetic,
    dataOrigin: origin,
    aggregates,
    traces: filtered,
    limitations: limitations.length ? limitations : [
      ...(isSynthetic ? ["PRUEBA SINTÉTICA: los datos fueron generados sintéticamente para validar el flujo de exportación; no representan mediciones reales."] : []),
      "Las marcas temporales de recepción corresponden al reloj monotónico local de NYRA, no al envío en TikTok o Discord.",
      "El ACK de inicio de reproducción incluye retraso de retorno unidireccional navegador→servidor y refleja el primer renderizado efectivo (no salida acústica física).",
    ],
  });
  const files = exportBenchmarkRun(targetDir, doc);
  return { doc, files };
}
