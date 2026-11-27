// Benchmark reproducible de STT: compara perfiles de Whisper local y/o Whisper remoto.
// Uso:
//   node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --profiles A --filter-profile confidence_ghost --split tune
//   node scripts/benchmark-stt.mjs --manifest voices/stt/manifest.json --engine local --profiles P0,P1,P2,P3,P4 --split tune
// No activa LLM, TTS, memoria ni aprendizaje.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { execSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { Whisper } from "../discord-voice.js";
import { WhisperRemoteClient } from "../lib/stt/whisper-remote-client.js";

const WAKE_RE = /(?<!\p{L})(nyra|(?:ch[eé]|me)?nira|ñira|naira|lyra|neira|nila)(?!\p{L})/iu;

function argVal(flag, def = null) {
  const i = process.argv.indexOf(flag);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}

function levenshteinOps(a, b) {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) => Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  let i = m, j = n, sub = 0, del = 0, ins = 0;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) { i--; j--; }
    else if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) { sub++; i--; j--; }
    else if (i > 0 && dp[i][j] === dp[i - 1][j] + 1) { del++; i--; }
    else { ins++; j--; }
  }
  return { dist: dp[m][n], sub, del, ins };
}

export const norm = s => String(s || "").toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, "").replace(/\s+/g, " ").trim();
// Normaliza únicamente variantes ortográficas inequívocas del personaje (Nyra/Nira/Ñira); nunca toca Mirá/Lira ni colapsa duplicados ("Nyra, Nyra").
export const normNyra = s => norm(s).replace(/(?<!\p{L})(nyra|nira|ñira)(?!\p{L})/gu, "nira");

export const werOps = (ref, hyp, normalizer = norm) => {
  const r = normalizer(ref).split(" ").filter(Boolean);
  const h = normalizer(hyp).split(" ").filter(Boolean);
  const ops = levenshteinOps(r, h);
  return { rate: r.length ? ops.dist / r.length : (h.length ? 1 : 0), ...ops, exact: normalizer(ref) === normalizer(hyp) };
};
const cer = (ref, hyp) => {
  const r = [...norm(ref)];
  const h = [...norm(hyp)];
  return r.length ? levenshteinOps(r, h).dist / r.length : (h.length ? 1 : 0);
};
const pct = (arr, p) => {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
};
const avg = arr => arr.length ? Number((arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(4)) : null;

function readPcmFromWav(filePath) {
  const buf = readFileSync(filePath);
  const idx = buf.indexOf(Buffer.from("data", "ascii"));
  if (idx < 0) throw new Error(`WAV sin chunk data: ${filePath}`);
  return buf.subarray(idx + 8);
}

function sampleGpu() {
  try {
    const out = execSync("nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
    const [usedMiB, freeMiB, utilPct] = out.split(",").map(s => Number(s.trim()));
    return { usedMiB, freeMiB, utilPct };
  } catch {
    return null;
  }
}

const isMain = process.argv[1] && resolve(process.argv[1]).endsWith("benchmark-stt.mjs");
if (isMain) {
  const manifestPath = resolve(argVal("--manifest", "voices/stt/manifest.json"));
  const engineMode = argVal("--engine", "local").toLowerCase();
  const stage = argVal("--stage", null)?.toUpperCase() || null;
  const stageProfiles = { A: "A,B", B: "B,C", C: "C,D,E,F", D: "C,G,H", E: "A", F: "A", G: "A,C" };
  const defaultProfiles = stage && stageProfiles[stage] ? stageProfiles[stage] : "A,B,C,D,E,F";
  const profiles = argVal("--profiles", defaultProfiles).split(",").map(s => s.trim().toUpperCase()).filter(Boolean);
  const split = argVal("--split", stage === "G" ? "eval" : "tune").toLowerCase();
  const repeat = Math.max(1, Number(argVal("--repeat", "1")) || 1);
  const vadOverride = argVal("--vad-threshold", null);
  const parsedVad = vadOverride !== null ? Number(vadOverride) : null;
  const secondPassMode = argVal("--second-pass", null);
  const filterProfile = argVal("--filter-profile", null);
  const promptMode = argVal("--prompt-mode", null);
  const hotwordsOverride = argVal("--hotwords", null);
  const beamOverride = argVal("--beam-size", null);
  const remoteUrl = argVal("--remote-url", process.env.WHISPER_REMOTE_URL || "");
  const remoteApiKey = argVal("--remote-api-key", process.env.WHISPER_REMOTE_API_KEY || "");
  const remoteModels = argVal("--remote-models", "medium,large-v3-turbo,large-v3").split(",").map(s => s.trim()).filter(Boolean);
  const jsonOut = argVal("--json", null);
  const csvOut = argVal("--csv", null);

  const allItems = JSON.parse(readFileSync(manifestPath, "utf8"));
  const items = split === "all"
    ? allItems
    : allItems.filter(it => !it.split || String(it.split).toLowerCase() === split);

  if (!items.length) {
    console.error(`No hay clips en ${manifestPath} para split=${split}`);
    process.exit(1);
  }

  const voicesDir = dirname(manifestPath);
  const summary = [];
  const rows = [];
  let coldInitMs = null;
  let localInfo = null;
  let gpuAfterInit = null;

  async function evaluateSuite(label, optsMeta, runClipFn) {
    const wers = [], wersNyra = [], cers = [], rawWers = [], latencies = [], rtfs = [];
    let silenceHalluc = 0, silenceTotal = 0;
    let validSpeechTotal = 0, validSpeechDiscarded = 0, exactMatches = 0;
    let wordSub = 0, wordDel = 0, wordIns = 0;
    let wakeTp = 0, wakeFp = 0, wakeFn = 0, wakeTn = 0;
    let errors = 0, secondPasses = 0, secondPassChanges = 0;
    let effectiveParams = null;

    for (let rep = 0; rep < repeat; rep++) {
      for (const item of items) {
        const pcm = readPcmFromWav(join(voicesDir, item.file));
        let res;
        try {
          res = await runClipFn(pcm, item);
        } catch (err) {
          errors++;
          rows.push({ suite: label, rep, file: item.file, error: err.message });
          continue;
        }
        effectiveParams = res.effectiveParams || effectiveParams;
        latencies.push(res.ms);
        if (typeof res.rtf === "number") rtfs.push(res.rtf);
        if (res.second_pass_triggered) secondPasses++;
        if (res.second_pass_changed_text) secondPassChanges++;

        const textNorm = norm(res.text);
        let itemWer = null, itemWerNyra = null, itemCer = null;
        if (!item.silence) {
          const wo = werOps(item.ref, res.text, norm);
          const won = werOps(item.ref, res.text, normNyra);
          itemWer = Number(wo.rate.toFixed(4));
          itemWerNyra = Number(won.rate.toFixed(4));
          itemCer = Number(cer(item.ref, res.text).toFixed(4));
          // Métricas de precisión estrictamente sobre audios únicos (rep === 0)
          if (rep === 0) {
            validSpeechTotal++;
            if (!textNorm) validSpeechDiscarded++;
            wers.push(wo.rate);
            wersNyra.push(won.rate);
            wordSub += wo.sub;
            wordDel += wo.del;
            wordIns += wo.ins;
            if (wo.exact) exactMatches++;
            rawWers.push(werOps(item.ref, res.raw ?? res.text, norm).rate);
            cers.push(itemCer);
          }
        } else if (rep === 0) {
          silenceTotal++;
          if (textNorm) silenceHalluc++;
        }

        const wakePred = WAKE_RE.test(res.text || "");
        const wakeTrue = Boolean(item.hasNyra);
        if (rep === 0) {
          if (wakeTrue && wakePred) wakeTp++;
          else if (!wakeTrue && wakePred) wakeFp++;
          else if (wakeTrue && !wakePred) wakeFn++;
          else wakeTn++;
        }

        rows.push({
          suite: label,
          rep,
          id: item.id || item.file,
          file: item.file,
          split: item.split || "unspecified",
          category: item.category || "",
          silence: Boolean(item.silence),
          hasNyra: wakeTrue,
          wakePred,
          wer: itemWer,
          werNyraNorm: itemWerNyra,
          cer: itemCer,
          ref: item.ref || "",
          raw: res.raw ?? res.text ?? "",
          text: res.text || "",
          ms: res.ms,
          inferenceMs: res.inference_ms ?? res.ms,
          rtf: res.rtf ?? null,
          passes: res.passes ?? 1,
          avgLogprob: res.avg_logprob ?? null,
          noSpeechProb: res.no_speech_prob ?? null,
          discardReason: res.discard_reason || "",
        });
      }
    }

    const wakePrec = wakeTp + wakeFp > 0 ? Number((wakeTp / (wakeTp + wakeFp)).toFixed(3)) : null;
    const wakeRec = wakeTp + wakeFn > 0 ? Number((wakeTp / (wakeTp + wakeFn)).toFixed(3)) : null;
    const wakeF1 = wakePrec !== null && wakeRec !== null && (wakePrec + wakeRec) > 0
      ? Number(((2 * wakePrec * wakeRec) / (wakePrec + wakeRec)).toFixed(3))
      : null;
    const entry = {
      suite: label,
      ...optsMeta,
      split,
      uniqueClips: items.length,
      totalExecutions: items.length * repeat,
      repeat,
      errors,
      effectiveParams,
      wer: avg(wers),
      werNyraNorm: avg(wersNyra),
      rawWer: avg(rawWers),
      cer: avg(cers),
      wordOps: { sub: wordSub, del: wordDel, ins: wordIns },
      exactMatches: `${exactMatches}/${validSpeechTotal}`,
      validDiscardRate: validSpeechTotal ? Number((validSpeechDiscarded / validSpeechTotal).toFixed(3)) : null,
      silenceHallucinations: `${silenceHalluc}/${silenceTotal}`,
      wake: { tp: wakeTp, fp: wakeFp, fn: wakeFn, tn: wakeTn, precision: wakePrec, recall: wakeRec, f1: wakeF1 },
      secondPasses,
      secondPassChanges,
      latencyMs: { avg: avg(latencies), p50: pct(latencies, 50), p90: pct(latencies, 90), p95: pct(latencies, 95), max: latencies.length ? Math.max(...latencies) : 0 },
      rtfAvg: avg(rtfs),
    };
    summary.push(entry);
    console.log(
      `${label}: WER=${entry.wer ?? "n/a"} (NyraNorm=${entry.werNyraNorm ?? "n/a"}, raw=${entry.rawWer ?? "n/a"}) CER=${entry.cer ?? "n/a"} ` +
      `Discard=${validSpeechDiscarded}/${validSpeechTotal} SilHalluc=${entry.silenceHallucinations} ` +
      `Wake(TP=${wakeTp}/${wakeTp + wakeFn},FP=${wakeFp},F1=${wakeF1 ?? "n/a"}) ` +
      `p50=${entry.latencyMs.p50}ms p95=${entry.latencyMs.p95}ms err=${errors}`
    );
  }

  if (engineMode === "local" || engineMode === "both") {
    const tInit0 = Date.now();
    const localWhisper = new Whisper();
    await localWhisper.ready;
    coldInitMs = Date.now() - tInit0;
    localInfo = localWhisper.info;
    gpuAfterInit = sampleGpu();
    console.log(`Local Whisper listo en ${coldInitMs}ms:`, localInfo, gpuAfterInit || "");

    try {
      for (const profile of profiles) {
        const opts = { profile };
        if (parsedVad !== null && Number.isFinite(parsedVad) && (profile === "G" || profile === "H" || profiles.length === 1)) {
          opts.vadThreshold = parsedVad;
        }
        if (secondPassMode) opts.secondPassMode = secondPassMode;
        if (filterProfile) opts.filterProfile = filterProfile;
        if (promptMode) opts.promptMode = promptMode;
        if (hotwordsOverride !== null) opts.hotwords = hotwordsOverride;
        if (beamOverride !== null && Number.isFinite(Number(beamOverride))) opts.beamSize = Number(beamOverride);
        const tag = filterProfile ? `local:${profile}:${filterProfile}` : `local:${profile}`;
        await evaluateSuite(tag, { engine: "local", profile, filterProfile: filterProfile || null }, async pcm => {
          const res = await localWhisper.transcribe(pcm, opts);
          return {
            ...res,
            effectiveParams: {
              model: res.model,
              device: res.device,
              computeType: res.compute_type,
              beamSize: res.beam_size,
              vadFilter: res.vad_filter,
              vadThreshold: res.vad_threshold,
              promptMode: res.prompt_mode,
              hotwords: res.hotwords,
              filterProfile: res.filter_profile,
              secondPassMode: secondPassMode || null,
            },
          };
        });
      }
    } finally {
      localWhisper.stop();
    }
  }

  if (engineMode === "remote" || engineMode === "both") {
    if (!remoteUrl || !remoteApiKey) {
      console.log("Remoto omitido: falta WHISPER_REMOTE_URL / WHISPER_REMOTE_API_KEY (--remote-url / --remote-api-key).");
    } else {
      for (const model of remoteModels) {
        const rem = new WhisperRemoteClient({
          config: { url: remoteUrl, apiKey: remoteApiKey, model, fallbackLocal: false },
          voicesDir,
          settingsFile: join(voicesDir, ".bench-stt.json"),
        });
        await evaluateSuite(`remote:${model}`, { engine: "remote", model }, async (pcm, item) => {
          const res = await rem.transcribePcm(pcm, { speakerId: item.speaker || "bench" });
          return {
            text: res.text,
            raw: res.text,
            ms: res.ms,
            inference_ms: res.inferenceMs ?? res.ms,
            rtf: res.rtf ?? null,
            effectiveParams: { model: res.model || model, device: res.device || "cpu", quantization: res.quantization || "int8" },
          };
        });
      }
    }
  }

  if (jsonOut) {
    mkdirSync(dirname(resolve(jsonOut)), { recursive: true });
    writeFileSync(resolve(jsonOut), JSON.stringify({ engineMode, stage, split, coldInitMs, localInfo, gpuAfterInit, summary, rows }, null, 2));
  }

  if (csvOut) {
    mkdirSync(dirname(resolve(csvOut)), { recursive: true });
    const header = "suite,rep,id,file,split,category,silence,hasNyra,wakePred,wer,werNyraNorm,cer,ms,inferenceMs,rtf,passes,avgLogprob,noSpeechProb,discardReason,ref,raw,text\n";
    const esc = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const body = rows.map(r => [r.suite, r.rep, r.id, r.file, r.split, r.category, r.silence, r.hasNyra, r.wakePred, r.wer, r.werNyraNorm, r.cer, r.ms, r.inferenceMs, r.rtf, r.passes, r.avgLogprob, r.noSpeechProb, r.discardReason, r.ref, r.raw, r.text].map(esc).join(",")).join("\n");
    writeFileSync(resolve(csvOut), header + body + "\n");
  }
}
