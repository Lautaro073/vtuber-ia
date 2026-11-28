// AudioWorkletProcessor de una sesión PCM progresiva. Corre en el hilo de audio; no decide CUÁNDO arrancar
// (eso lo hace stream-player.js con la política adaptativa): solo reproduce, detecta cortes y reporta.
// Sin SharedArrayBuffer: el PCM llega por port.postMessage (Int16Array transferido, sin copias).
import { PcmPlaybackCore } from "./pcm-core.js";

class NyraPcmPlayer extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // sampleRate (global del worklet) = frecuencia REAL del AudioContext: el núcleo remuestrea si no es 44100
    this.core = new PcmPlaybackCore(sampleRate, options?.processorOptions?.inRate || 44100);
    this.alive = true;
    this.reportEvery = Math.round(sampleRate * 0.05);
    this.sinceReport = 0;
    this.lastLevel = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === "push") this.core.push(data.pcm);
      else if (data.type === "end") this.core.end();
      else if (data.type === "play") this.core.play();
      else if (data.type === "stop") { this.core.stop(); this.alive = false; }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0]?.[0];
    if (!out) return this.alive;
    const level = this.core.render(out);
    for (let ch = 1; ch < outputs[0].length; ch++) outputs[0][ch].set(out);
    this.lastLevel = level;
    for (const ev of this.core.events.splice(0)) {
      this.port.postMessage({ type: ev, time: currentTime, consumed: this.core.consumed });
      if (ev === "drained") this.alive = false;
    }
    this.sinceReport += out.length;
    if (this.sinceReport >= this.reportEvery) {
      this.sinceReport = 0;
      this.port.postMessage({ type: "tick", consumed: this.core.consumed, level, state: this.core.state, buffered: this.core.available });
    }
    return this.alive;
  }
}

registerProcessor("nyra-pcm-player", NyraPcmPlayer);
