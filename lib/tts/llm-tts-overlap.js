// Fase 3 (experimental): superposición LLM → TTS dentro de UNA respuesta.
// Mientras el LLM sigue escribiendo, si ya hay una primera oración completa se sintetiza en SGLang (sin reproducirla).
// Al terminar, la respuesta pasa por TODAS las validaciones de siempre; recién entonces se compara el texto EFECTIVO que
// iría al TTS (con todas sus transformaciones) con lo sintetizado: si empieza exactamente igual (en un límite de oración)
// y con la misma emoción, se aprovecha y el resto se sintetiza a continuación en la misma sesión PCM. Si no, se descarta.
// Nada de esto toca memoria, acciones, moderación ni prompts: solo decide si un audio preparado se puede usar.

// Proveedores cuyo cliente entrega texto INCREMENTAL de la respuesta (append-only) según su implementación:
//   claude-code: stream_event/text_delta (--include-partial-messages)   agy: step_update agent_response text_delta
// codex-cli entrega el mensaje completo de una vez (item.completed) y opencode emite partes que pueden ser snapshots:
// sin texto parcial confiable → no se especula con ellos.
export const OVERLAP_PROVIDERS = new Set(["claude-code", "agy"]);

const ABBREV = /(?:^|[\s(«"'])(?:sr|sra|srta|dr|dra|lic|ing|prof|etc|ej|aprox|vs|ud|uds|pág|pag|núm|num|av|nro|tel|min|seg|hs|km|cm|kg|mr|mrs|st)\.$/iu;
// Marcas, markdown, JSON y links se transforman o se descartan después: una oración que los tenga no se especula.
// (Se aplica DESPUÉS de sacar la etiqueta [EMOCIÓN: …], que sí se interpreta de forma segura.)
export const UNSAFE_SPEECH = /[[\]*<>`#{}|\\]|https?:|www\./i;
const MIN_CHARS = 25;   // más corto no vale un pedido aparte (TTFA ~3.6 s por pedido)
const MAX_CHARS = 260;

// Primera oración COMPLETA de un texto que sigue creciendo (append-only). Conservador: solo . ! ? (no comas, no "..."),
// con espacio y el comienzo de otra oración después (así el signo ya no puede cambiar), sin abreviaturas, decimales,
// links ni marcas. Devuelve { segment, rest } o null.
export function findFirstSentence(text, { minChars = MIN_CHARS, maxChars = MAX_CHARS } = {}) {
  const s = String(text || "");
  const re = /[.!?]+(?=\s+["«¿¡(]?[\p{Lu}\p{N}])/gu;
  let m;
  while ((m = re.exec(s))) {
    const end = m.index + m[0].length;
    if (end > maxChars) return null;
    if (m[0] === "." && ABBREV.test(s.slice(0, end))) continue;
    if (/\.{2,}/.test(m[0]) || s[m.index - 1] === "…") continue; // pausas, no fin de oración
    const segment = s.slice(0, end).trim();
    if (segment.length < minChars) continue;
    const count = re2 => (segment.match(re2) || []).length;
    if (count(/«/g) !== count(/»/g) || count(/\(/g) !== count(/\)/g) || count(/"/g) % 2) continue; // comillas/paréntesis abiertos
    return { segment, rest: s.slice(end).trimStart() };
  }
  return null;
}

// ¿El texto final efectivo empieza EXACTAMENTE con lo sintetizado, en un límite de palabra? → resto | null
export function prefixRemainder(finalText, specText) {
  const f = String(finalText || ""), p = String(specText || "");
  if (!p || !f.startsWith(p)) return null;
  if (f.length > p.length && !/\s/.test(f[p.length])) return null;
  return f.slice(p.length).trim();
}

export class OverlapController {
  // deps: { tts, enabled(): bool, now(): ms, delayMs, minHeadStartMs }. tts debe exponer: overlapReady(source) → motivo|null,
  //   ttsTextOf(text), speculateStart({ttsText, emotion, mood, expiresAt, rank}) → handle|null,
  //   commitSpeculation(handle, remainder, est) → {url}|null, discardSpeculation(handle, reason)
  constructor({ tts, enabled, now = () => performance.now(), log = () => {}, delayMs = 250, minHeadStartMs = 1000 }) {
    Object.assign(this, { tts, enabled, now, log, delayMs, minHeadStartMs });
    this.stats = { started: 0, used: 0, discarded: 0, notTriggered: 0, skipped: {}, discardReasons: {}, savedMsTotal: 0, wastedGpuMsTotal: 0, last: null };
  }

  _skip(reason) { this.stats.skipped[reason] = (this.stats.skipped[reason] || 0) + 1; return null; }

  // Una especulación por respuesta. build(sentence) → { ttsText, emotion } o null si la oración no se puede predecir.
  begin({ providerId, source, eligibleText = true, mood = 50, expiresAt = null, rank = 4, build }) {
    if (!this.enabled()) return null;
    if (!OVERLAP_PROVIDERS.has(providerId)) return this._skip(`proveedor sin texto parcial confiable (${providerId || "ninguno"})`);
    if (!eligibleText) return this._skip("respuesta fija o evento sin LLM");
    const why = this.tts.overlapReady(source);
    if (why) return this._skip(why);
    return { state: "watching", providerId, source, mood, expiresAt, rank, build, t0: this.now(), handle: null, spec: null, committed: false, delayTimer: null, pendingBuilt: null };
  }

  _launch(s) {
    if (!s || (s.state !== "delayed" && s.state !== "watching")) return;
    const built = s.pendingBuilt;
    s.delayTimer = null;
    if (!built?.ttsText || built.ttsText.length < MIN_CHARS) { s.state = "unsafe"; return; }
    const why = this.tts.overlapReady(s.source);
    if (why) { s.state = "busy"; this._skip(why); return; }
    const handle = this.tts.speculateStart({ ...built, mood: s.mood, expiresAt: s.expiresAt, rank: s.rank });
    if (!handle) { s.state = "busy"; this._skip("backend ocupado o ya hay una síntesis anticipada en curso"); return; }
    s.state = "speculating";
    s.handle = handle;
    s.spec = built;
    s.expressive = this.tts.sglang?.config?.expressive;
    s.tSpec = this.now();
    this.stats.started += 1;
  }

  // Se llama con el texto acumulado cada vez que llega un fragmento. Dispara a lo sumo una síntesis anticipada.
  feed(s, bodySoFar) {
    if (!s || s.state !== "watching") return;
    const found = findFirstSentence(bodySoFar);
    if (!found) return;
    const built = s.build(found.segment); // build aplica UNSAFE_SPEECH y las mismas transformaciones puras que el final
    if (!built?.ttsText || built.ttsText.length < MIN_CHARS) { s.state = "unsafe"; return; }
    s.pendingBuilt = built;
    if (this.delayMs > 0) {
      s.state = "delayed";
      s.delayTimer = setTimeout(() => this._launch(s), this.delayMs);
      s.delayTimer.unref?.();
    } else {
      this._launch(s);
    }
  }

  // Después de TODAS las validaciones: ¿se puede usar lo preparado? → { url, cancelled:false, fallbackUsed:false } | null
  async finish(s, { filtered, emotion, estDurationSec = null }) {
    if (!s) return null;
    if (s.delayTimer) { clearTimeout(s.delayTimer); s.delayTimer = null; }
    if (s.state === "delayed") {
      s.state = "aborted";
      this._skip("texto completo antes de disparar especulación");
      return null;
    }
    if (s.state !== "speculating") {
      if (s.state === "watching") this.stats.notTriggered += 1;
      return null;
    }
    const finalTts = this.tts.ttsTextOf(filtered);
    const remainder = prefixRemainder(finalTts, s.spec.ttsText);
    const headStartMs = Math.max(0, Math.round(this.now() - s.tSpec));
    let reason = null;
    if (remainder === null) reason = "el texto final no empieza con la oración sintetizada";
    else if (emotion !== s.spec.emotion) reason = "cambió la emoción";
    else if (s.expressive !== undefined && s.expressive !== this.tts.sglang?.config?.expressive) reason = "cambió la expresividad";
    else if (this.tts.overlapReady(s.source)) reason = `cambió la configuración (${this.tts.overlapReady(s.source)})`;
    else if (remainder && this.minHeadStartMs > 0 && headStartMs < this.minHeadStartMs) reason = "ventaja insuficiente (< 1 s)";
    if (reason) { this.abort(s, reason); return null; }
    const res = await this.tts.commitSpeculation(s.handle, remainder, estDurationSec);
    if (!res) { this.abort(s, "la síntesis anticipada falló, fue desplazada o el backend está ocupado"); return null; }
    s.state = "committed";
    s.committed = true;
    this.stats.used += 1;
    // Ahorro potencial: lo que la síntesis ya llevaba hecha cuando terminó la validación (estimado, no medido en audio)
    this.stats.savedMsTotal += headStartMs;
    this.stats.last = { result: "used", headStartMs, segmentChars: s.spec.ttsText.length, remainderChars: remainder.length };
    return { url: res.url, cancelled: false, fallbackUsed: false, overlap: true };
  }

  // Descarta lo especulado si no se usó (respuesta rechazada, vacía, vencida, error...). Idempotente.
  abort(s, reason = "respuesta descartada") {
    if (!s || s.committed || s.state === "aborted") return;
    if (s.delayTimer) { clearTimeout(s.delayTimer); s.delayTimer = null; }
    const wasSpeculating = s.state === "speculating";
    s.state = "aborted";
    if (!wasSpeculating) return;
    const wasted = this.tts.discardSpeculation(s.handle, reason);
    this.stats.discarded += 1;
    this.stats.discardReasons[reason] = (this.stats.discardReasons[reason] || 0) + 1;
    this.stats.wastedGpuMsTotal += wasted || 0;
    this.stats.last = { result: "discarded", reason };
    this.log(`🧪 Síntesis anticipada descartada: ${reason}`);
  }

  publicStats() {
    return JSON.parse(JSON.stringify(this.stats));
  }
}
