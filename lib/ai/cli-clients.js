import { mkdirSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";

export function terminateChildTree(child) {
  if (!child || child.exitCode !== null) return;
  if (process.platform === "win32" && child.pid) {
    const result = spawnSync("taskkill.exe", ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore",
    });
    if (result.status === 0) return;
  }
  child.kill();
}

export function windowsCommandShim(command, args) {
  if (command.includes('"')) throw new Error("Ruta de CLI inválida");
  const safeArgs = args.map(argument => {
    const value = String(argument);
    if (!/^[\w./:#=-]+$/.test(value)) throw new Error(`Argumento de CLI inválido: ${value}`);
    return value;
  });
  return [process.env.ComSpec || "cmd.exe", ["/d", "/s", "/c", `call "${command}" ${safeArgs.join(" ")}`]];
}

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
      terminateChildTree(process);
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
    const replacement = this.next;
    this.next = null;
    for (const child of new Set([current, replacement, ...this.children])) terminateChildTree(child);
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

  _prepareInput(text) {
    return this.systemPrompt ? `${this.systemPrompt}\n\n${text}` : text;
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
        terminateChildTree(this.proc);
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
      this._send(this._prepareInput(text));
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

  _prepareInput(text) { return text; }

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
      shell: config.shell || false,
      agent: config.agent,
      serverManager: config.serverManager || null,
    };
    this.queue = Promise.resolve();
    this.children = new Set();
    this.active = new Map();
    this.generation = 0;
  }

  async _execute(text, onDelta, generation, timeoutMs, images = []) {
    if (generation !== this.generation) return Promise.reject(new Error(`${this.constructor.providerId} detenido`));
    mkdirSync(this.config.workDir, { recursive: true });
    const deadline = Date.now() + timeoutMs;
    const manager = this.config.serverManager;
    let serverUrl = null;
    if (manager?.enabled) {
      const budget = Math.max(500, Math.min(10000, deadline - Date.now() - 1000));
      serverUrl = await manager.ensureStarted(budget).catch(() => null);
      if (generation !== this.generation) return Promise.reject(new Error(`${this.constructor.providerId} detenido`));
    }
    try {
      return await this._spawnAttempt(text, onDelta, generation, Math.max(50, deadline - Date.now()), serverUrl, images);
    } catch (err) {
      const remaining = deadline - Date.now();
      const canFallback = serverUrl
        && !err.hasPartialResponse
        && generation === this.generation
        && remaining > 1500
        && /ECONNREFUSED|Could not reach server|Unauthorized|fetch failed/i.test(err.message);
      if (!canFallback) throw err;
      manager?.markUnhealthy(serverUrl);
      return this._spawnAttempt(text, onDelta, generation, remaining, null, images);
    }
  }

  _spawnAttempt(text, onDelta, generation, timeoutMs, serverUrl = null, images = []) {
    if (generation !== this.generation) return Promise.reject(new Error(`${this.constructor.providerId} detenido`));
    let [command, args, env] = this._cmd(serverUrl, images);
    if (serverUrl && this.config.serverManager) env = this.config.serverManager.clientEnv(env);
    const windowsVerbatimArguments = this.config.shell && process.platform === "win32";
    if (windowsVerbatimArguments) [command, args] = windowsCommandShim(command, args);
    return new Promise((resolve, reject) => {
      const child = spawn(command, args, {
        cwd: this.config.workDir,
        windowsHide: true,
        env,
        windowsVerbatimArguments,
      });
      this.children.add(child);
      let stdout = "";
      let stderr = "";
      let response = "";
      let failure = ""; // el error que el CLI informa en su JSON (ej: límite de uso); stderr trae ruido de conectores
      let sessionId = null;
      const seenParts = new Map();
      let settled = false;
      const finish = (error, value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        this.active.delete(child);
        if (sessionId && serverUrl) this.config.serverManager?.releaseSession(sessionId);
        if (error && response) error.hasPartialResponse = true;
        error ? reject(error) : resolve(value);
      };
      this.active.set(child, finish);
      const timer = setTimeout(() => {
        finish(new Error(`${this.constructor.providerId} timeout`));
        terminateChildTree(child);
      }, timeoutMs);

      child.stdout.on("data", data => {
        stdout += data;
        let newline;
        while ((newline = stdout.indexOf("\n")) >= 0) {
          const line = stdout.slice(0, newline);
          stdout = stdout.slice(newline + 1);
          let event;
          try { event = JSON.parse(line); } catch { continue; }
          if (event.sessionID && !sessionId && serverUrl) {
            sessionId = event.sessionID;
            this.config.serverManager?.trackSession(sessionId);
          }
          const reported = event.error?.message || (event.type === "error" && event.message);
          if (reported) failure = String(reported).slice(0, 300);
          const rawDelta = this._textFromEvent(event, seenParts);
          if (!rawDelta) continue;
          response += rawDelta;
          onDelta(rawDelta);
        }
      });
      child.stderr.on("data", data => { stderr = (stderr + data).slice(-500); });
      child.on("error", error => finish(error));
      child.on("exit", code => {
        this.children.delete(child);
        if (settled) return;
        if (code !== 0) return finish(new Error(`${this.constructor.providerId} terminó con código ${code}: ${failure || stderr.trim()}`));
        if (!response) return finish(new Error(`${this.constructor.providerId} terminó sin respuesta válida`));
        finish(null, response);
      });
      child.stdin.on("error", error => finish(error));
      child.stdin.end(String(text));
    });
  }

  // images: rutas de archivos para adjuntar (solo los CLIs que lo soportan, como Codex con -i)
  run(text, onDelta = () => {}, timeoutMs = this.timeoutMs ?? this.config.timeoutMs, images = []) {
    const generation = this.generation;
    const input = this.systemPrompt ? `${this.systemPrompt}\n\n${text}` : text;
    const job = this.queue.then(() => this._execute(input, onDelta, generation, timeoutMs, images));
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
    }
    for (const child of this.children) terminateChildTree(child);
  }
}

export class CodexClient extends OneShotClient {
  _cmd(_serverUrl = null, images = []) {
    return [this.config.path, [
      ...this.config.argsPrefix,
      "exec",
      "--json",
      "--ephemeral",
      "--skip-git-repo-check",
      // Vacío para NYRA, como agy: sin config del usuario (MCP, modelo xhigh, notify), reglas, conectores ni plugins
      "--ignore-user-config", "--ignore-rules",
      "--disable", "apps", "--disable", "plugins", "--disable", "multi_agent",
      "-C", this.config.workDir,
      "-s", "read-only",
      ...(this.model === "default" ? [] : ["-m", this.model]),
      ...images.flatMap(file => ["-i", file]), // la captura de pantalla para la visión
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
  _agentArgs() {
    const agent = this.config.agent === undefined ? "nyra" : this.config.agent;
    if (!agent) return [];
    const value = String(agent);
    if (!/^[\w.-]+$/.test(value)) throw new Error(`Agente de OpenCode inválido: ${value}`);
    return ["--agent", value];
  }

  _cmd(serverUrl = null) {
    if (serverUrl && !/^https?:\/\/127\.0\.0\.1:\d+$/.test(String(serverUrl))) {
      throw new Error(`URL de servidor OpenCode inválida: ${serverUrl}`);
    }
    return [this.config.path, [
      ...this.config.argsPrefix,
      "run",
      "--format", "json",
      ...(serverUrl ? ["--server", serverUrl] : ["--standalone"]),
      ...this._agentArgs(),
      ...(this.model === "default"
        ? []
        : ["--model", this.model]),
    ], this.config.env];
  }

  _textFromEvent(event, seenParts = null) {
    if (event.part?.type !== "text") return "";
    const text = event.part.text || "";
    const partId = event.part.id;
    if (!partId || !seenParts) return text;
    const prev = seenParts.get(partId);
    if (prev === undefined) {
      seenParts.set(partId, text);
      return text;
    }
    if (text === prev) return ""; // evento JSON duplicado
    if (text.startsWith(prev)) { // snapshot acumulativo
      seenParts.set(partId, text);
      return text.slice(prev.length);
    }
    seenParts.set(partId, prev + text);
    return text;
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
