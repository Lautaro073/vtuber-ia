// Políticas de arranque/recuperación para la reproducción progresiva de PCM (SGLang, Fase 2).
// Módulo puro (sin DOM ni Web Audio): lo usan el navegador (obs.html), los tests y el simulador.
// Todos los tiempos en segundos de UN solo reloj monotónico (quien llama no mezcla relojes de procesos distintos).
//
// Idea (causal): la GPU genera más lento que tiempo real (RTF ~1.6 → r ≈ 0.63 s de audio por segundo) y en bloques
// (~1.85 s el primero, ~2.09 s los siguientes, cada ~3.2 s). Si arrancamos con A segundos recibidos y faltan R por
// generar, el peor momento es justo antes de cada bloque. Para el bloque j (de n restantes) hace falta
//   A ≥ (tiempo hasta el próximo bloque) + (j−1)·(p − b)   con p = b / r
// y el máximo es el último: A ≥ T_next + (R − b)·(1 − r)/r. Se suma un margen de seguridad.
// R sale de una estimación de la duración total por texto (incierta): se infla por `estUncertainty` y nunca se
// asume que falta menos de un bloque mientras la transmisión no terminó.

export const PCM_RATE = 44100;

export class AdaptiveStartPolicy {
  constructor({
    estDurationSec = null,   // estimación por texto (servidor); null = desconocida
    priorRate = 0.63,        // s de audio generados por s de reloj (RTF ~1.6 en L4)
    priorBlockSec = 2.09,    // bloque típico después del primero
    safetySec = 0.35,
    estUncertainty = 0.2,    // la duración real puede ser +20 % de lo estimado
    mergeWindowSec = 0.15,   // lecturas TCP muy juntas = un mismo evento de generación
    // Fase 3 (opcional): la respuesta llega en DOS pedidos; el segundo arranca con su propio TTFA. Sin esto la política
    // trata ese hueco como un bloque normal y corta en el límite. null = una sola síntesis (comportamiento de Fase 2).
    segmentBoundarySec = null,
    segmentGapSec = 3.8,
    segmentStartedAt = null, // instante (en el mismo reloj que t) en que arrancó el 2.º pedido
  } = {}) {
    Object.assign(this, { estDurationSec, priorRate, priorBlockSec, safetySec, estUncertainty, mergeWindowSec, segmentBoundarySec, segmentGapSec, segmentStartedAt });
    this.events = [];        // [{t, a}] a = audio acumulado (s)
    this.ended = false;
    this.underruns = 0;
  }

  get received() { return this.events.length ? this.events[this.events.length - 1].a : 0; }

  onArrival(t, cumAudioSec) {
    const last = this.events[this.events.length - 1];
    if (last && t - last.t <= this.mergeWindowSec) last.a = cumAudioSec; // misma ráfaga: conserva el tiempo de su inicio
    else this.events.push({ t, a: cumAudioSec });
  }

  onEnd(t, totalSec) {
    if (totalSec > this.received) this.onArrival(t, totalSec);
    this.ended = true;
  }

  // Un corte demuestra que la estimación fue optimista: la próxima espera es más conservadora
  onUnderrun() {
    this.underruns += 1;
    this.estUncertainty += 0.25;
  }

  rate() {
    const ev = this.events;
    if (ev.length < 2) return this.priorRate;
    // Velocidad observada sin contar el primer bloque (llega de golpe tras el TTFA)
    const span = ev[ev.length - 1].t - ev[0].t;
    const gen = ev[ev.length - 1].a - ev[0].a;
    if (span < 0.5 || gen <= 0) return this.priorRate;
    const intervals = ev.length - 1;
    const obs = Math.min(1.5, Math.max(0.2, gen / span));
    return (intervals * obs + 2 * this.priorRate) / (intervals + 2); // el prior pesa como 2 intervalos
  }

  blockSec() {
    const ev = this.events;
    if (ev.length < 2) return this.priorBlockSec;
    let sum = 0;
    for (let i = 1; i < ev.length; i++) sum += ev[i].a - ev[i - 1].a;
    const n = ev.length - 1;
    return (sum + 2 * this.priorBlockSec) / (n + 2);
  }

  // Segundos de audio que deberían estar en el búfer para (re)arrancar en el instante t
  requiredBufferSec(t, playedSec = 0) {
    if (this.ended || !this.events.length) return 0;
    const A = this.received;
    // Con dos pedidos, el primero termina con un bloque corto que llega rápido y la velocidad observada sale optimista:
    // hasta cruzar el límite no se confía en una velocidad mayor que la previa.
    const beforeBoundary = this.segmentBoundarySec !== null && A < this.segmentBoundarySec + 0.02;
    const r = beforeBoundary ? Math.min(this.rate(), this.priorRate) : this.rate();
    const b = this.blockSec();
    const p = b / r;
    const est = this.estDurationSec ? this.estDurationSec * (1 + this.estUncertainty) : A + 2 * b;
    // Si ya se recibió más de lo estimado, la estimación quedó corta: se asume al menos un bloque más
    const R = est - A > 0.05 ? est - A : b;
    const next = Math.min(b, R); // el último bloque suele ser más corto
    const sinceLast = t - this.events[this.events.length - 1].t;
    let tNext = Math.max(0, Math.min(p, next / r) - sinceLast);
    const deficit = r < 1 ? Math.max(0, R - next) * (1 - r) / r : 0;
    let gap = 0;
    if (this.segmentBoundarySec !== null) {
      if (A >= this.segmentBoundarySec - 0.02 && A < this.segmentBoundarySec + 0.02) {
        const sinceSegStart = this.segmentStartedAt !== null ? Math.max(0, t - this.segmentStartedAt) : sinceLast;
        tNext = Math.max(tNext, Math.max(0, this.segmentGapSec - sinceSegStart)); // primer segmento completo: TTFA desde que arrancó el 2.º pedido
      } else if (A < this.segmentBoundarySec) {
        gap = Math.max(0, this.segmentGapSec - p); // todavía falta cruzar el límite: ese hueco extra hay que cubrirlo
      }
    }
    return tNext + deficit + gap + this.safetySec;
  }

  shouldPlay(t, playedSec = 0) {
    const buffered = this.received - playedSec;
    if (buffered <= 0) return false;
    if (this.ended) return true; // todo el audio ya está: arrancar ya
    return buffered >= this.requiredBufferSec(t, playedSec);
  }
}

// Búfer fijo (referencia): arranca/reanuda con X s en búfer o al terminar la transmisión
export class FixedStartPolicy {
  constructor({ bufferSec = 0.5 } = {}) {
    this.bufferSec = bufferSec;
    this.received = 0;
    this.ended = false;
  }
  onArrival(_t, cum) { this.received = cum; }
  onEnd(_t, total) { this.received = Math.max(this.received, total); this.ended = true; }
  onUnderrun() {}
  shouldPlay(_t, played = 0) {
    const buffered = this.received - played;
    return buffered > 0 && (this.ended || buffered >= this.bufferSec);
  }
}

// Esperar el audio completo (equivalente al modo WAV de Fase 1, sin contar escritura/descarga del archivo)
export class WholeFilePolicy {
  constructor() { this.received = 0; this.ended = false; }
  onArrival(_t, cum) { this.received = cum; }
  onEnd(_t, total) { this.received = Math.max(this.received, total); this.ended = true; }
  onUnderrun() {}
  shouldPlay() { return this.ended && this.received > 0; }
}

// Cota retrospectiva (NO causal: usa la traza completa). Solo sirve de referencia en las comparaciones.
export function oracleStartSec(trace) {
  let best = trace[0][0];
  for (let k = 1; k < trace.length; k++) best = Math.max(best, trace[k][0] - trace[k - 1][1]);
  return best;
}

// Simulación determinista con reloj discreto: trace = [[t_s, audio_acumulado_s], ...] (el último punto = fin).
export function simulatePlayback(trace, policy, { stepSec = 0.01, endAtSec = null } = {}) {
  const arrivals = trace.map(([t, a]) => ({ t, a }));
  const total = arrivals[arrivals.length - 1].a;
  const tEnd = endAtSec ?? arrivals[arrivals.length - 1].t;
  let i = 0, t = 0, played = 0, playing = false, firstAudible = null;
  let underruns = 0, underrunSec = 0, inUnderrun = false;
  const maxT = tEnd + total + 60;
  while (played < total - 1e-9 && t < maxT) {
    while (i < arrivals.length && arrivals[i].t <= t + 1e-9) {
      if (i === arrivals.length - 1) policy.onEnd(arrivals[i].t, arrivals[i].a);
      else policy.onArrival(arrivals[i].t, arrivals[i].a);
      i++;
    }
    const received = Math.min(total, i ? arrivals[i - 1].a : 0);
    if (!playing && policy.shouldPlay(t, played)) {
      playing = true;
      if (firstAudible === null) firstAudible = t;
      inUnderrun = false;
    }
    if (playing) {
      const avail = received - played;
      if (avail >= stepSec - 1e-9) played += stepSec;
      else {
        played += Math.max(0, avail);
        if (played < total - 1e-9) { // se quedó sin audio antes del final: corte
          playing = false; underruns += 1; inUnderrun = true; policy.onUnderrun();
        }
      }
    } else if (inUnderrun) underrunSec += stepSec;
    t += stepSec;
  }
  return {
    firstAudibleSec: firstAudible === null ? null : Number(firstAudible.toFixed(3)),
    endSec: Number(t.toFixed(3)),
    underruns,
    underrunSec: Number(underrunSec.toFixed(3)),
    playedSec: Number(played.toFixed(3)),
    totalSec: total,
  };
}
