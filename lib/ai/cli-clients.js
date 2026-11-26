import { mkdirSync } from "node:fs";
import { spawn } from "node:child_process";

const EFFORT_RE = /-(low|medium|high|xhigh|max)$/;
export const withEffort = model => EFFORT_RE.test(model) ? model : `${model}-low`;

export class AgyClient {
  constructor(name, model, config = {}) {
    this.name = name;
    this.model = model;
    this.config = {
      path: config.path,
      argsPrefix: config.argsPrefix || [],
      workDir: config.workDir,
      timeoutMs: config.timeoutMs ?? 60000,
      coldMs: config.coldMs ?? 90000,
      turnsPerProcess: config.turnsPerProcess ?? 12,
      env: config.env || process.env,
    };
    this.proc = null;
    this.next = null;
    this.turns = 0;
    this.listener = null;
    this.queue = Promise.resolve();
    this.children = new Set();
  }

  _start() {
    this.proc = this._spawn();
    this.turns = 0;
  }

  _refresh() {
    if (this.next) return;
    const process = this._spawn();
    this.next = process;
    const done = error => {
      clearTimeout(timer);
      process.warming = null;
      if (this.next !== process) return;
      if (!error) { process.warm = true; return; }
      this.next = null;
      process.kill();
    };
    const timer = setTimeout(() => done(new Error("timeout")), this.config.timeoutMs + this._coldMs());
    process.warming = event => this._onEvent(event, () => {}, done);
    this._send("Respondé solo: ok", process);
  }

  _coldMs() { return this.config.coldMs; }

  _spawn() {
    mkdirSync(this.config.workDir, { recursive: true });
    const [command, args, env] = this._cmd();
    const process = spawn(command, args, {
      cwd: this.config.workDir,
      windowsHide: true,
      env,
    });
    this.children.add(process);
    process.forModel = this.model;
    process.stdin.on("error", () => {});
    let buffer = "";
    process.stdout.on("data", data => {
      buffer += data;
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        let event;
        try { event = JSON.parse(line); } catch { continue; }
        (process === this.proc ? this.listener : process.warming)?.(event);
      }
    });
    process.stderr.on("data", data => {
      const message = String(data).trim();
      if (message) console.warn(`${this.constructor.providerId || "cli"}[${this.name}]: ${message.slice(0, 200)}`);
    });
    const onExit = info => {
      this.children.delete(process);
      process.warming?.({ event: "exit", info });
      if (this.next === process) this.next = null;
      if (this.proc !== process) return;
      this.proc = null;
      this.listener?.({ event: "exit", info });
    };
    process.on("exit", code => onExit(`código ${code}`));
    process.on("error", error => onExit(error.message));
    return process;
  }

  stop() {
    const current = this.proc;
    this.proc = null;
    current?.kill();
    const replacement = this.next;
    this.next = null;
    replacement?.kill();
    for (const child of this.children) child.kill();
    this.listener?.({ event: "exit", info: "detenido" });
  }

  _cmd() {
    return [this.config.path, [
      ...this.config.argsPrefix,
      "--input-format", "stream-json",
      "--output-format", "stream-json",
      "--model", this.model,
      "--sandbox",
      "--disable-slash-commands",
    ], this.config.env];
  }

  _send(text, process = this.proc) {
    process.stdin.write(JSON.stringify({ event: "user", message: { content: text } }) + "\n");
  }

  _onEvent(event, onDelta, finish) {
    const update = event.step_update;
    if (update?.step_type === "agent_response" && update.text_delta) onDelta(update.text_delta);
    else if (event.event === "result") event.result.status === "SUCCESS"
      ? finish(null, event.result.response || "")
      : finish(new Error(event.result.error || "agy error"));
    else if (event.event === "exit") finish(new Error(`agy terminó (${event.info})`));
  }

  run(text, onDelta = () => {}, timeoutMs = this.timeoutMs ?? this.config.timeoutMs) {
    const job = this.queue.then(() => new Promise((resolve, reject) => {
      if (this.next?.warm && this.next.forModel === this.model) {
        this.proc?.kill();
        this.proc = this.next;
        this.next = null;
        this.turns = 1;
      } else if (!this.proc || this.proc.forModel !== this.model) {
        this.stop();
        this._start();
      }
      if (this.turns === 0) timeoutMs += this._coldMs();
      this.turns++;
      const finish = (error, value) => {
        clearTimeout(timer);
        this.listener = null;
        error ? reject(error) : resolve(value);
      };
      const timer = setTimeout(() => {
        finish(new Error(`${this.constructor.providerId || "cli"} timeout`));
        this.stop();
      }, timeoutMs);
      this.listener = event => this._onEvent(event, onDelta, finish);
      this._send(text);
    }));
    this.queue = job.catch(() => {}).then(() => {
      if (this.proc && this.turns >= (this.maxTurns ?? this.config.turnsPerProcess)) this._refresh();
    });
    return job;
  }

  async *stream(text) {
    const chunks = [];
    let done = false;
    let error = null;
    let wake = null;
    this.run(text, delta => { chunks.push(delta); wake?.(); })
      .then(() => { done = true; wake?.(); }, reason => { error = reason; done = true; wake?.(); });
    while (true) {
      if (chunks.length) { yield chunks.shift(); continue; }
      if (done) break;
      await new Promise(resolve => { wake = resolve; });
      wake = null;
    }
    if (error) throw error;
  }
}
AgyClient.providerId = "agy";

export class ClaudeClient extends AgyClient {
  _cmd() {
    const [, model, effort] = withEffort(this.model).match(/^(.*)-(low|medium|high|xhigh|max)$/);
    return [this.config.path, [
      ...this.config.argsPrefix,
      "-p",
      "--input-format", "stream-json",
      "--output-format", "stream-json",
      "--verbose",
      "--include-partial-messages",
      "--model", model,
      "--effort", effort,
      "--tools", "",
      "--strict-mcp-config",
      "--setting-sources", "",
      "--disable-slash-commands",
      "--no-session-persistence",
      "--system-prompt", this.systemPrompt || "Interpretás a un personaje de stream en vivo según las instrucciones de cada mensaje. Respondés solo con la línea del personaje.",
    ], { ...this.config.env, CLAUDECODE: "" }];
  }

  _send(text, process = this.proc) {
    process.stdin.write(JSON.stringify({ type: "user", message: { role: "user", content: text } }) + "\n");
  }

  _onEvent(event, onDelta, finish) {
    const delta = event.type === "stream_event" && event.event?.delta;
    if (delta?.type === "text_delta" && delta.text) onDelta(delta.text);
    else if (event.type === "result") event.subtype === "success" && !event.is_error
      ? finish(null, event.result || "")
      : finish(new Error(String(event.result || event.subtype || "claude error").slice(0, 200)));
    else if (event.event === "exit") finish(new Error(`claude terminó (${event.info})`));
  }
}
ClaudeClient.providerId = "claude-code";

class OneShotClient {
  constructor(name, model, config = {}) {
    this.name = name;
    this.model = model;
    this.config = {
      path: config.path,
      argsPrefix: config.argsPrefix || [],
      workDir: config.workDir,
      timeoutMs: config.timeoutMs ?? 120000,
      env: config.env || process.env,
    };
    this.queue = Promise.resolve();
    this.children = new Set();
    this.active = new Map();
    this.generation = 0;
  }

  _execute(text, onDelta, generation, timeoutMs) {
    if (generation !== this.generation) return Promise.reject(new Error(`${this.constructor.providerId} detenido`));
    mkdirSync(this.config.workDir, { recursive: true });
    const [command, args, env] = this._cmd();
    return new Promise((resolve, reject) => {
      const child = spawn(command, args, {
        cwd: this.config.workDir,
        windowsHide: true,
        env,
      });
      this.children.add(child);
      let stdout = "";
      let stderr = "";
      let response = "";
      let settled = false;
      const finish = (error, value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        this.active.delete(child);
        error ? reject(error) : resolve(value);
      };
      this.active.set(child, finish);
      const timer = setTimeout(() => {
        finish(new Error(`${this.constructor.providerId} timeout`));
        child.kill();
      }, timeoutMs);

      child.stdout.on("data", data => {
        stdout += data;
        let newline;
        while ((newline = stdout.indexOf("\n")) >= 0) {
          const line = stdout.slice(0, newline);
          stdout = stdout.slice(newline + 1);
          let event;
          try { event = JSON.parse(line); } catch { continue; }
          const delta = this._textFromEvent(event);
          if (!delta) continue;
          response += delta;
          onDelta(delta);
        }
      });
      child.stderr.on("data", data => { stderr = (stderr + data).slice(-500); });
      child.on("error", error => finish(error));
      child.on("exit", code => {
        this.children.delete(child);
        if (settled) return;
        if (code !== 0) return finish(new Error(`${this.constructor.providerId} terminó con código ${code}: ${stderr.trim()}`));
        if (!response) return finish(new Error(`${this.constructor.providerId} terminó sin respuesta válida`));
        finish(null, response);
      });
      child.stdin.on("error", error => finish(error));
      child.stdin.end(String(text));
    });
  }

  run(text, onDelta = () => {}, timeoutMs = this.timeoutMs ?? this.config.timeoutMs) {
    const generation = this.generation;
    const job = this.queue.then(() => this._execute(text, onDelta, generation, timeoutMs));
    this.queue = job.catch(() => {});
    return job;
  }

  async *stream(text) {
    const chunks = [];
    let done = false;
    let error = null;
    let wake = null;
    this.run(text, delta => { chunks.push(delta); wake?.(); })
      .then(() => { done = true; wake?.(); }, reason => { error = reason; done = true; wake?.(); });
    while (true) {
      if (chunks.length) { yield chunks.shift(); continue; }
      if (done) break;
      await new Promise(resolve => { wake = resolve; });
      wake = null;
    }
    if (error) throw error;
  }

  stop() {
    this.generation++;
    for (const [child, finish] of this.active) {
      finish(new Error(`${this.constructor.providerId} detenido`));
      child.kill();
    }
    for (const child of this.children) child.kill();
  }
}

export class CodexClient extends OneShotClient {
  _cmd() {
    return [this.config.path, [
      ...this.config.argsPrefix,
      "exec",
      "--json",
      "--ephemeral",
      "--skip-git-repo-check",
      "-C", this.config.workDir,
      "-s", "read-only",
      ...(this.model === "default" ? [] : ["-m", this.model]),
      "-",
    ], this.config.env];
  }

  _textFromEvent(event) {
    return event.type === "item.completed" && event.item?.type === "agent_message"
      ? event.item.text || ""
      : "";
  }
}
CodexClient.providerId = "codex-cli";

export class OpenCodeClient extends OneShotClient {
  _cmd() {
    return [this.config.path, [
      ...this.config.argsPrefix,
      "run",
      "--format", "json",
      "--standalone",
      ...(this.model === "default" ? [] : ["--model", this.model]),
    ], this.config.env];
  }

  _textFromEvent(event) {
    return event.part?.type === "text" ? event.part.text || "" : "";
  }
}
OpenCodeClient.providerId = "opencode";

export function createClientFactory(config) {
  const classes = {
    agy: AgyClient,
    "claude-code": ClaudeClient,
    "codex-cli": CodexClient,
    opencode: OpenCodeClient,
  };
  return ({ providerId, role, model, options = {} }) => {
    const Client = classes[providerId];
    if (!Client) throw new Error(`Proveedor desconocido: ${providerId}`);
    return new Client(role, model, { ...config[providerId], ...options });
  };
}
