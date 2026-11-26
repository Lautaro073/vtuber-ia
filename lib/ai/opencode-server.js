import { randomBytes } from "node:crypto";
import { mkdirSync } from "node:fs";
import { spawn } from "node:child_process";
import { createServer } from "node:net";

import { terminateChildTree, windowsCommandShim } from "./cli-clients.js";

function allocateLoopbackPort() {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close(err => (err ? reject(err) : resolve(port)));
    });
  });
}

/**
 * Administra un proceso único y persistente de `opencode serve` en 127.0.0.1
 * compartido entre todos los roles de NYRA, con autenticación efímera en memoria
 * y limpieza segura de sesiones creadas por NYRA.
 */
export class OpenCodeServerManager {
  constructor(config = {}) {
    this.config = {
      path: config.path || "opencode",
      argsPrefix: config.argsPrefix || [],
      workDir: config.workDir,
      env: config.env || process.env,
      shell: config.shell || false,
      startTimeoutMs: config.startTimeoutMs ?? 10000,
      maxRestarts: config.maxRestarts ?? 3,
      restartWindowMs: config.restartWindowMs ?? 60000,
      healthPath: config.healthPath || "/api/session",
      enabled: config.enabled ?? (process.env.OPENCODE_PERSISTENT !== "0"),
      fetchImpl: config.fetchImpl || fetch,
      portAllocator: config.portAllocator || allocateLoopbackPort,
    };
    this.proc = null;
    this.url = null;
    this._password = null;
    this._starting = null;
    this._stopToken = 0;
    this._restarts = [];
    this._cooldownUntil = 0;
    this._nyraSessions = new Set();
    this._activeSessions = new Set();
  }

  get enabled() {
    return !!this.config.enabled;
  }

  isRunning() {
    return !!(this.url && this.proc && this.proc.exitCode === null);
  }

  _authHeader() {
    return this._password
      ? "Basic " + Buffer.from(`opencode:${this._password}`).toString("base64")
      : null;
  }

  clientEnv(baseEnv = this.config.env) {
    if (!this._password) return baseEnv;
    return { ...baseEnv, OPENCODE_PASSWORD: this._password };
  }

  trackSession(sessionId) {
    if (!sessionId) return;
    this._nyraSessions.add(sessionId);
    this._activeSessions.add(sessionId);
  }

  releaseSession(sessionId) {
    if (!sessionId) return;
    this._activeSessions.delete(sessionId);
    if (!this._nyraSessions.has(sessionId)) return;
    this._deleteSession(sessionId).catch(() => {});
  }

  async _deleteSession(sessionId) {
    if (!this.isRunning() || this._activeSessions.has(sessionId)) return;
    const baseUrl = this.url;
    const auth = this._authHeader();
    if (!baseUrl || !auth) return;
    try {
      const res = await this.config.fetchImpl(`${baseUrl}/api/session/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
        headers: { Authorization: auth },
        signal: AbortSignal.timeout(1500),
      });
      if (res.ok || res.status === 404) this._nyraSessions.delete(sessionId);
    } catch { }
  }

  async ensureStarted(timeoutMs = this.config.startTimeoutMs) {
    if (!this.enabled) return null;
    if (this.isRunning()) return this.url;
    if (Date.now() < this._cooldownUntil) return null;
    if (this._starting) return this._starting;

    const token = this._stopToken;
    this._starting = this._startInternal(timeoutMs, token).finally(() => {
      this._starting = null;
    });
    return this._starting;
  }

  async _startInternal(timeoutMs, token) {
    const now = Date.now();
    this._restarts = this._restarts.filter(ts => now - ts < this.config.restartWindowMs);
    if (this._restarts.length >= this.config.maxRestarts) {
      this._cooldownUntil = now + this.config.restartWindowMs;
      return null;
    }
    if (this._restarts.length > 0) {
      const delay = Math.min(2000, 250 * 2 ** (this._restarts.length - 1));
      await new Promise(r => setTimeout(r, delay));
      if (token !== this._stopToken) return null;
    }
    this._restarts.push(Date.now());

    mkdirSync(this.config.workDir, { recursive: true });
    const port = await this.config.portAllocator();
    if (token !== this._stopToken) return null;

    const password = randomBytes(24).toString("hex");
    const serverEnv = {
      ...this.config.env,
      OPENCODE_SERVER_PASSWORD: password,
      OPENCODE_PASSWORD: password,
    };

    let command = this.config.path;
    let args = [
      ...this.config.argsPrefix,
      "serve",
      "--hostname", "127.0.0.1",
      "--port", String(port),
    ];
    const windowsVerbatimArguments = this.config.shell && process.platform === "win32";
    if (windowsVerbatimArguments) [command, args] = windowsCommandShim(command, args);

    const child = spawn(command, args, {
      cwd: this.config.workDir,
      windowsHide: true,
      env: serverEnv,
      windowsVerbatimArguments,
    });
    this.proc = child;
    this._password = password;

    let exited = false;
    let exitInfo = "";
    child.stderr.on("data", data => {
      const msg = String(data).trim();
      if (msg && !/password/i.test(msg)) exitInfo = msg.slice(-200);
    });
    child.on("exit", code => {
      exited = true;
      if (this.proc === child) {
        this.proc = null;
        this.url = null;
      }
      if (!exitInfo) exitInfo = `código ${code}`;
    });
    child.on("error", err => {
      exited = true;
      exitInfo = err.message;
      if (this.proc === child) {
        this.proc = null;
        this.url = null;
      }
    });

    const url = `http://127.0.0.1:${port}`;
    const auth = "Basic " + Buffer.from(`opencode:${password}`).toString("base64");
    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
      if (token !== this._stopToken || exited) {
        terminateChildTree(child);
        if (this.proc === child) { this.proc = null; this.url = null; this._password = null; }
        return null;
      }
      try {
        const res = await this.config.fetchImpl(`${url}${this.config.healthPath}`, {
          headers: { Authorization: auth },
          signal: AbortSignal.timeout(Math.min(1000, Math.max(200, deadline - Date.now()))),
        });
        if (res.ok) {
          if (token !== this._stopToken) {
            terminateChildTree(child);
            return null;
          }
          this.url = url;
          return url;
        }
      } catch { }
      await new Promise(r => setTimeout(r, 80));
    }

    terminateChildTree(child);
    if (this.proc === child) { this.proc = null; this.url = null; this._password = null; }
    return null;
  }

  markUnhealthy(expectedUrl = null) {
    if (expectedUrl && this.url && this.url !== expectedUrl) return;
    const child = this.proc;
    this.proc = null;
    this.url = null;
    this._password = null;
    if (child) terminateChildTree(child);
  }

  async stop() {
    this._stopToken++;
    const sessions = [...this._nyraSessions].filter(id => !this._activeSessions.has(id));
    if (this.isRunning() && sessions.length > 0) {
      await Promise.race([
        Promise.allSettled(sessions.map(id => this._deleteSession(id))),
        new Promise(r => setTimeout(r, 500)),
      ]);
    }
    const child = this.proc;
    this.proc = null;
    this.url = null;
    this._password = null;
    this._nyraSessions.clear();
    this._activeSessions.clear();
    if (child) terminateChildTree(child);
  }
}
