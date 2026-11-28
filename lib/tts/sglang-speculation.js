// Fase 3: síntesis anticipada en SGLang para TTSEngine (se instala sobre la instancia; no cambia los otros motores).
// Garantías (revisadas para no bloquear el backend):
//  - Solo se especula con el backend LIBRE y hay a lo sumo una especulación viva por vez.
//  - La 1.ª oración es un trabajo más de la cola única (nunca inferencias paralelas). Al terminar de generarse, el
//    backend queda libre: el audio espera en la sesión (dentro del presupuesto de memoria de Fase 2), no la cola.
//  - Si llega un turno de prioridad SUPERIOR mientras la especulación no está confirmada, se descarta enseguida (y se
//    aborta si estaba generando). Con igual o menor prioridad se respeta el orden FIFO de siempre.
//  - Al confirmar, el resto va al frente de la cola para salir pegado; si el backend está ocupado con OTRO trabajo, no
//    se puede garantizar continuidad: se descarta y la respuesta sigue el camino normal (sin cortes por espera ajena).
//  - Nada suena hasta commitSpeculation(), que solo se llama tras las validaciones de siempre. Stop y TTL la invalidan;
//    cancelar nunca dispara fallback.
const SPEC_SOURCE = "spec";
const UNDECIDED_MS = 60000;   // una especulación que nadie confirma ni descarta se libera (memoria), sin tocar la cola
export const SEGMENT_GAP_SEC = 3.8; // TTFA típico de un pedido SGLang (mediana de las trazas L4)

export function installSpeculation(tts, { applyPhonetics = s => s, resolveLiveEngine = () => tts.resolveEngine("live") } = {}) {
  tts._spec = null;
  const sched = tts._sglang;

  tts.ttsTextOf = text => applyPhonetics(tts.clean(String(text || "")));

  // null si se puede especular para esta fuente; si no, el motivo (se muestra en el panel)
  tts.overlapReady = (source = "live") => {
    if (source !== "live") return "solo en el directo (Discord sigue igual)";
    if (resolveLiveEngine() !== "sglang") return "el motor del directo no es SGLang";
    if (tts.sglang.config.playbackMode !== "stream") return "requiere Reproducción: Streaming PCM";
    if (!tts.sglang.isConfigured()) return "SGLang sin configurar";
    if (!tts.streams.capableClients().length) return "ninguna vista /obs compatible conectada";
    return null;
  };

  // Un turno de prioridad superior que necesita el backend desplaza a la especulación no confirmada
  const prevHook = sched.onEnqueue;
  sched.onEnqueue = job => {
    prevHook?.(job);
    const h = tts._spec;
    if (h && !h.committed && job.kind === "normal" && job.rank < h.rank) {
      h.displacedByPriority = true;
      if (h.turnId) tts.tracer?.recordOverlap?.(h.turnId, { displacedByPriority: true });
      tts.discardSpeculation(h, "llegó un turno de prioridad superior");
    }
  };

  tts.speculateStart = ({ ttsText, emotion, mood = 50, expiresAt = null, rank = 4, turnId = null }) => {
    if (tts._spec) return null;
    if (sched.busy) return null; // con trabajo en curso o esperando, la anticipación no ganaría nada y estorbaría
    const epoch = tts._cancelEpoch.get("live") || 0;
    const st = tts.streams.create({ source: SPEC_SOURCE, text: ttsText, turnId });
    if (turnId) tts.tracer?.linkStream?.(turnId, st.id, st.url);
    const h = {
      st, ttsText, emotion, mood, rank, epoch, expiresAt, turnId, url: tts.sglang.config.url, expressive: tts.sglang.config.expressive,
      cancelled: false, committed: false, displacedByPriority: false, gpuMs: 0, runningSince: null, t0: performance.now(),
    };
    h.isCancelled = () => h.cancelled || (tts._cancelEpoch.get("live") || 0) !== epoch;
    h.synth = async (text, segmentIndex = 0) => {
      const tr = tts.tracer;
      const jobIds = h.turnId ? tr?.startTtsJob?.(h.turnId, {
        requestedEngine: "sglang", effectiveEngine: "sglang", playbackMode: "stream",
        segmentIndex, speculative: segmentIndex === 0, streamId: st.id,
      }) : null;
      const segStart = performance.now();
      h.runningSince = segStart;
      if (h.turnId && jobIds) tr?.updateTtsJob?.(h.turnId, jobIds.ttsJobId, { requestStart: true, queueWaitMs: 0 });
      let firstPcmSeg = false;
      try {
        const m = await tts.sglang.synthesizeStream(text, {
          emotion, mood, source: SPEC_SOURCE, expiresAt, shouldCancel: h.isCancelled, streamId: st.id,
          onHeaders: () => {
            if (h.turnId && jobIds) tr?.updateTtsJob?.(h.turnId, jobIds.ttsJobId, { headers: true });
            tts.streams.onHeaders(st);
          },
          onPcm: (b, i) => {
            if (!firstPcmSeg) {
              firstPcmSeg = true;
              if (h.turnId && jobIds) tr?.updateTtsJob?.(h.turnId, jobIds.ttsJobId, { firstValidPcm: true });
              if (segmentIndex > 0 && h.seg1DoneAt && h.turnId) {
                tr?.recordOverlap?.(h.turnId, { interSegmentGapMs: Math.max(0, performance.now() - h.seg1DoneAt) });
              }
            }
            tts.streams.onPcm(st, b, i);
          },
        });
        if (h.turnId && jobIds) {
          tr?.updateTtsJob?.(h.turnId, jobIds.ttsJobId, {
            httpBodyEnd: true, httpTotalMs: m.totalMs, bytesReceived: m.bytes || 0, samplesReceived: m.samples || 0,
            chunkCount: m.chunks || 0, retries429: m.retries || 0, backoffMs: m.backoffMs || 0,
            audioDurationSec: m.audioDurationSec, result: "ok",
          });
        }
        return m;
      } catch (err) {
        if (h.turnId && jobIds) {
          tr?.updateTtsJob?.(h.turnId, jobIds.ttsJobId, {
            cancelled: Boolean(err.cancelled || h.isCancelled()),
            partialFailure: Boolean(err.partial || segmentIndex > 0),
            errorCode: err.code || (err.status ? `http_${err.status}` : "spec_error"),
          });
        }
        throw err;
      } finally {
        h.gpuMs += performance.now() - h.runningSince;
        h.runningSince = null;
      }
    };
    const run = async () => {
      if (h.isCancelled()) throw Object.assign(new Error("especulación cancelada"), { cancelled: true });
      if (expiresAt && Date.now() > expiresAt) throw Object.assign(new Error("especulación vencida"), { expired: true });
      return h.synth(ttsText, 0);
    };
    h.seg1 = sched.enqueue(run, { rank, kind: "spec", tag: h });
    h.seg1.then(() => { h.seg1Done = true; h.seg1DoneAt = performance.now(); }, err => { h.seg1Failed = !(err.cancelled || err.expired); h.seg1Error = err; });
    h.timer = setTimeout(() => tts.discardSpeculation(h, "nadie la confirmó a tiempo"), UNDECIDED_MS);
    h.timer.unref?.();
    tts._spec = h;
    return h;
  };

  // La respuesta pasó las validaciones y empieza exactamente con lo sintetizado. remainder: texto efectivo restante.
  // → { url } (sesión PCM normal de Fase 2) | null si ya no sirve (el llamador sigue el camino normal)
  tts.commitSpeculation = async (h, remainder, estDurationSec = null) => {
    if (!h || h !== tts._spec || h.isCancelled() || h.seg1Failed) return null;
    if (h.expiresAt && Date.now() > h.expiresAt) return null;
    if (h.url !== tts.sglang.config.url) return null;                         // cambió el destino/voz del gateway
    if (h.expressive !== tts.sglang.config.expressive) return null;           // cambió la etiqueta emocional
    if (!tts.streams.get(h.st.url)) return null;
    // Continuidad: el resto debe poder arrancar apenas termine la 1.ª oración. Si el backend está con OTRO trabajo, no.
    if (remainder && sched.active && sched.active.tag !== h) return null;
    const st = h.st;
    h.committed = true;
    tts._spec = null;
    clearTimeout(h.timer);
    st.estDurationSec = estDurationSec ?? tts.streams.estimateDurationSec(`${h.ttsText} ${remainder || ""}`);
    if (!remainder) {
      h.seg1.then(() => (h.isCancelled() ? tts.streams.cancel(st, "stop") : tts.streams.onEnd(st, {})),
        err => (err.cancelled || err.expired || h.isCancelled() ? tts.streams.cancel(st, "stop") : tts.streams.onFail(st, Object.assign(err, { partial: st.samples > 0 }))));
      return { url: st.url };
    }
    if (h.seg1Done) tts.streams.markSegmentBoundary(st, SEGMENT_GAP_SEC);
    const runRest = async () => {
      try { await h.seg1; } catch (err) {
        if (err.cancelled || err.expired || h.isCancelled()) tts.streams.cancel(st, "stop");
        else tts.streams.onFail(st, Object.assign(err, { partial: st.samples > 0 }));
        return;
      }
      if (h.isCancelled()) { tts.streams.cancel(st, "stop"); return; }
      // Límite exacto: lo recibido del 1.er pedido. El resto arranca ahora y trae su TTFA.
      tts.streams.markSegmentBoundary(st, SEGMENT_GAP_SEC);
      try {
        const m = await h.synth(remainder, 1);
        if (h.isCancelled()) tts.streams.cancel(st, "stop");
        else tts.streams.onEnd(st, m);
      } catch (err) {
        if (h.isCancelled() || err.cancelled) tts.streams.cancel(st, "stop");
        else {
          if (h.turnId) tts.tracer?.recordOverlap?.(h.turnId, { secondSegmentFailed: true, partialFailure: true });
          tts.streams.onFail(st, Object.assign(err, { partial: true })); // ya hay audio del primer segmento
        }
      }
    };
    sched.enqueue(runRest, { rank: h.rank, kind: "spec-rest", front: true, tag: h }).catch(() => {});
    return { url: st.url };
  };

  // Descartar: saca el pedido de la espera o lo aborta si está corriendo, y libera la sesión. → ms de GPU gastados
  tts.discardSpeculation = (h, reason = "descartada") => {
    if (!h || h.committed || h.cancelled) return 0;
    h.cancelled = true;
    clearTimeout(h.timer);
    if (tts._spec === h) tts._spec = null;
    sched.drop(h, Object.assign(new Error(`especulación descartada: ${reason}`), { cancelled: true }));
    if (h.runningSince !== null) tts.sglang.abortSource(SPEC_SOURCE);
    tts.streams.discard(h.st, "speculative_discarded");
    return Math.round(h.gpuMs + (h.runningSince !== null ? performance.now() - h.runningSince : 0));
  };

  // Para TTSEngine.stop(): Stop global o del directo invalida la especulación y la sesión ya confirmada
  tts.stopSpeculation = (source = null) => {
    if (source && source !== "live") return;
    if (tts._spec) tts.discardSpeculation(tts._spec, "stop");
    tts.sglang.abortSource(SPEC_SOURCE);
    tts.streams.cancelSource(SPEC_SOURCE);
  };
  return tts;
}
