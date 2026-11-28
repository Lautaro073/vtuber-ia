// Cola única del backend SGLang (1 worker). Mismo orden que la cadena de promesas que reemplaza: FIFO, un trabajo a la
// vez, el siguiente arranca cuando el anterior termina (bien o mal). Agregados de Fase 3, sin tocar las prioridades del
// directo:
//  - onEnqueue(job): aviso antes de encolar (la especulación se descarta si llega un turno más prioritario).
//  - front: solo para el RESTO de una respuesta ya confirmada, para que salga pegado a su primera oración.
export class SglangScheduler {
  constructor() {
    this.active = null;
    this.pending = [];
    this.onEnqueue = null;
  }

  get busy() { return Boolean(this.active) || this.pending.length > 0; }

  // rank: prioridad del turno (PRI: 0 = regalo … 5 = idle; menor = más urgente). kind: "normal" | "spec" | "spec-rest"
  enqueue(run, { rank = 4, kind = "normal", front = false, tag = null } = {}) {
    let resolve, reject;
    const promise = new Promise((a, b) => { resolve = a; reject = b; });
    const job = { run, rank, kind, tag, resolve, reject };
    try { this.onEnqueue?.(job); } catch { }
    if (front) this.pending.unshift(job); else this.pending.push(job);
    this._next();
    return promise;
  }

  // Saca de la espera los trabajos con esa etiqueta (los que ya corren se cancelan por su AbortController)
  drop(tag, error) {
    const keep = [];
    for (const job of this.pending) (job.tag === tag ? job.reject(error) : keep.push(job));
    this.pending = keep;
  }

  _next() {
    if (this.active || !this.pending.length) return;
    const job = this.pending.shift();
    this.active = job;
    Promise.resolve()
      .then(job.run)
      .then(
        v => { this.active = null; this._next(); job.resolve(v); },
        e => { this.active = null; this._next(); job.reject(e); },
      );
  }
}
