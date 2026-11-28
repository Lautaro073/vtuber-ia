// Núcleo de reproducción PCM progresiva (sin Web Audio): lo usa el AudioWorklet y se prueba en Node.
// - Entrada: Int16 mono a inRate (44100). Salida: Float32 a outRate (la frecuencia real del AudioContext).
// - Remuestreo lineal CONTINUO: la posición fraccionaria se conserva entre bloques y entre chunks (no se reinicia),
//   así no hay saltos de fase ni cambios de velocidad/tono. Con outRate === inRate es copia exacta muestra a muestra.
// - Corte por falta de datos (underrun): se baja el volumen en los últimos ~4 ms disponibles y se pausa sin perder
//   ni repetir muestras; al reanudar se sube en ~4 ms. Al final de la transmisión no se atenúa nada.
export class PcmPlaybackCore {
  constructor(outRate, inRate = 44100, { fadeMs = 4 } = {}) {
    this.step = inRate / outRate;
    this.chunks = [];
    this.ci = 0;          // chunk actual
    this.off = 0;         // muestra entera actual dentro del chunk
    this.frac = 0;        // posición fraccionaria entre la muestra actual y la siguiente
    this.available = 0;   // muestras de entrada desde la actual (incluida) que todavía no se consumieron
    this.consumed = 0;    // muestras de entrada ya reproducidas
    this.pushed = 0;
    this.ended = false;
    this.playing = false;
    this.state = "buffering";  // buffering | playing | underrun | drained | stopped
    this.gain = 0;
    this.target = 1;
    this.fadeLen = Math.max(1, Math.round(outRate * fadeMs / 1000));
    this.fadeStep = 1 / this.fadeLen;
    this.events = [];          // "started" | "underrun" | "drained" (los consume quien llama)
    this.started = false;
  }

  push(int16) {
    if (this.state === "stopped" || this.state === "drained" || !int16?.length) return;
    this.chunks.push(int16);
    this.available += int16.length;
    this.pushed += int16.length;
  }

  end() { this.ended = true; }

  play() {
    if (this.state !== "buffering" && this.state !== "underrun") return;
    if (this.available <= 0 && !this.ended) return;
    this.playing = true;
    this.state = "playing";
    this.target = 1;
  }

  stop() {
    this.playing = false;
    this.state = "stopped";
    this.chunks = [];
    this.available = 0;
  }

  _sample(k) {
    let ci = this.ci, off = this.off + k;
    while (ci < this.chunks.length && off >= this.chunks[ci].length) { off -= this.chunks[ci].length; ci++; }
    return ci < this.chunks.length ? this.chunks[ci][off] / 32768 : null;
  }

  _advance(n) {
    if (n <= 0) return;
    this.off += n;
    this.consumed += n;
    this.available -= n;
    while (this.ci < this.chunks.length && this.off >= this.chunks[this.ci].length) { this.off -= this.chunks[this.ci].length; this.ci++; }
    if (this.ci > 8) { this.chunks.splice(0, this.ci); this.ci = 0; } // soltar memoria de lo ya sonado
  }

  // Llena `out` (Float32Array). Devuelve el nivel RMS de lo que efectivamente sonó (0 si está en pausa o en corte).
  render(out) {
    let sumSq = 0;
    for (let i = 0; i < out.length; i++) {
      if (!this.playing) { out[i] = 0; continue; }
      if (this.available <= 0 || (this.available === 1 && !this.ended)) {
        // sin la muestra siguiente no se puede interpolar: fin o corte
        if (this.ended && this.available <= 0) { this.playing = false; this.state = "drained"; this.events.push("drained"); }
        else { this.playing = false; this.state = "underrun"; this.gain = 0; this.events.push("underrun"); }
        out[i] = 0;
        continue;
      }
      if (!this.started) { this.started = true; this.events.push("started"); }
      // ¿Quedan menos de fadeLen muestras de salida con los datos actuales y la transmisión sigue? → atenuar
      if (!this.ended && (this.available - 1 - this.frac) / this.step <= this.fadeLen) this.target = 0;
      else this.target = 1;
      if (this.gain < this.target) this.gain = Math.min(this.target, this.gain + this.fadeStep);
      else if (this.gain > this.target) this.gain = Math.max(this.target, this.gain - this.fadeStep);
      const s0 = this._sample(0);
      const s1n = this._sample(1);
      const s1 = s1n === null ? s0 : s1n;
      const y = (s0 + (s1 - s0) * this.frac) * this.gain;
      out[i] = y;
      sumSq += y * y;
      this.frac += this.step;
      const n = Math.floor(this.frac);
      this.frac -= n;
      // al final de todo, la última muestra se consume aunque no haya siguiente
      this._advance(Math.min(n, this.available));
    }
    return Math.sqrt(sumSq / out.length);
  }
}
