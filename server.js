/**
 * NYRA VTuber - IA Local con RAG regenerativo
 * Usa Ollama para correr modelos locales (llama3, mistral, etc.)
 * RAG con TF-IDF para aprender de texto sin APIs externas
 */

import express from "express";
import { WebSocketServer } from "ws";
import { createServer } from "http";
import { readFileSync, writeFileSync, existsSync, unlinkSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { exec } from "child_process";
import cors from "cors";
import { v4 as uuidv4 } from "uuid";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = 3000;

// ─── Configuración ────────────────────────────────────────────────────────────
const CONFIG = {
  ollamaUrl: "http://localhost:11434",
  model: "dolphin-mistral",  // sin censura, sigue el persona libremente — alternativas: dolphin-llama3, mistral
  // Fallback si no tenés dolphin: cambiar a "mistral" o "qwen2:1.5b" (qwen2 tiene safety training)
  visionModel: "moondream",  // moondream = 1.7GB, corre junto a qwen2:1.5b sin crashear
  maxContext: 4,        // historial reducido = contexto más pequeño = más rápido
  maxRagChunks: 3,       // menos chunks RAG = prompt más corto = más rápido
  knowledgeFile: "knowledge.json",
  chunkSize: 200,
  useSystemTTS: false,  // false = Murf (audio viaja al OBS del amigo por WS) ✅
  // true  = voz del sistema (suena en TU PC, el amigo no escucha nada) ❌
  // ── Murf.ai TTS ──────────────────────────────────────────────────────────
  // 1. Cuenta en https://murf.ai → Dashboard → API Keys → copiá tu key
  // 2. Elegí voz en https://murf.ai/voice-studio → copiá el voiceId
  murf: {
    apiKey: "ap2_2c427819-8af8-481d-adcd-5bf78ffa3bab",   // ← reemplazar
    voiceId: "es-MX-luisa",     // ← reemplazar con tu voz elegida
    style: "Conversational",    // Conversational | Promo | Sad | Angry
    rate: -10,                 // velocidad: -50 (lento) a +50 (rápido)
    pitch: -5,                  // tono: -50 a +50
  },
};

// ─── Base de Conocimiento RAG ─────────────────────────────────────────────────
class KnowledgeBase {
  constructor() {
    this.chunks = [];
    this.load();
  }

  load() {
    const path = join(__dirname, CONFIG.knowledgeFile);
    if (existsSync(path)) {
      try {
        this.chunks = JSON.parse(readFileSync(path, "utf8"));
        console.log(`📚 Cargados ${this.chunks.length} chunks de conocimiento`);
      } catch { this.chunks = []; }
    }
  }

  save() {
    writeFileSync(
      join(__dirname, CONFIG.knowledgeFile),
      JSON.stringify(this.chunks, null, 2)
    );
  }

  // Dividir texto en chunks y guardar
  addText(text, source = "manual") {
    const words = text.trim().split(/\s+/);
    const newChunks = [];

    for (let i = 0; i < words.length; i += CONFIG.chunkSize) {
      const chunk = words.slice(i, i + CONFIG.chunkSize).join(" ");
      if (chunk.length < 20) continue;

      const entry = {
        id: uuidv4(),
        text: chunk,
        source,
        tokens: this.tokenize(chunk),
        addedAt: new Date().toISOString(),
      };
      this.chunks.push(entry);
      newChunks.push(entry);
    }

    this.save();
    return newChunks.length;
  }

  // Tokenización simple
  tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\sáéíóúüñ]/g, "")
      .split(/\s+/)
      .filter(w => w.length > 2);
  }

  // TF-IDF simplificado para buscar chunks relevantes
  search(query, topK = CONFIG.maxRagChunks) {
    if (this.chunks.length === 0) return [];

    const queryTokens = this.tokenize(query);

    // Precomputar IDF una sola vez fuera del loop (evita O(n²))
    const N = this.chunks.length;
    const dfMap = {};
    for (const qt of queryTokens) {
      if (!(qt in dfMap)) {
        dfMap[qt] = this.chunks.reduce((acc, c) => acc + (c.tokens.includes(qt) ? 1 : 0), 0);
      }
    }

    const scored = this.chunks.map(chunk => {
      const tf = {};
      chunk.tokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });

      let score = 0;
      queryTokens.forEach(qt => {
        if (tf[qt]) {
          const idf = Math.log(N / (dfMap[qt] + 1)) + 1;
          score += (tf[qt] / chunk.tokens.length) * idf;
        }
      });

      return { ...chunk, score };
    });

    return scored
      .filter(c => c.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  getStats() {
    return {
      totalChunks: this.chunks.length,
      totalWords: this.chunks.reduce((s, c) => s + c.tokens.length, 0),
      sources: [...new Set(this.chunks.map(c => c.source))].length,
    };
  }

  clear() {
    this.chunks = [];
    this.save();
  }
}

// ─── Sistema de Filtros ───────────────────────────────────────────────────────
class FilterEngine {
  constructor() {
    this.data = { words: [], blockedUsers: [], autoLearn: false };
    this.filePath = join(__dirname, "filters.json");
    this.load();
  }

  load() {
    if (existsSync(this.filePath)) {
      try { this.data = { ...this.data, ...JSON.parse(readFileSync(this.filePath, "utf8")) }; } catch { }
    }
  }

  save() {
    writeFileSync(this.filePath, JSON.stringify(this.data, null, 2));
  }

  // Chequea si un mensaje/usuario está bloqueado
  checkMessage(text, username = "") {
    if (username && this.data.blockedUsers.some(u => u.toLowerCase() === username.toLowerCase())) {
      return { blocked: true, reason: `usuario bloqueado: @${username}` };
    }
    const lower = text.toLowerCase();
    const hit = this.data.words.find(w => lower.includes(w.toLowerCase()));
    if (hit) return { blocked: true, reason: "palabra bloqueada" };
    return { blocked: false };
  }

  // Censa palabras bloqueadas en el output de NYRA
  filterOutput(text) {
    let out = text;
    this.data.words.forEach(w => {
      const re = new RegExp(w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
      out = out.replace(re, "***");
    });
    return out;
  }

  addWord(w) { const word = w.toLowerCase().trim(); if (word && !this.data.words.includes(word)) { this.data.words.push(word); this.save(); } }
  removeWord(w) { this.data.words = this.data.words.filter(x => x !== w.toLowerCase().trim()); this.save(); }
  addUser(u) { const user = u.toLowerCase().trim(); if (user && !this.data.blockedUsers.includes(user)) { this.data.blockedUsers.push(user); this.save(); } }
  removeUser(u) { this.data.blockedUsers = this.data.blockedUsers.filter(x => x !== u.toLowerCase().trim()); this.save(); }
}

// ─── Personalidad de NYRA (opiniones formadas con el tiempo) ─────────────────────
class NyraPersonality {
  constructor() {
    this.filePath = join(__dirname, "nyra_personality.json");
    this.data = { opinions: {} };
    this.load();
  }

  load() {
    if (existsSync(this.filePath)) {
      try {
        this.data = { opinions: {}, ...JSON.parse(readFileSync(this.filePath, "utf8")) };
        const n = Object.keys(this.data.opinions).length;
        if (n > 0) console.log(`💜 Personalidad cargada: ${n} opiniones`);
      } catch { }
    }
  }

  save() { writeFileSync(this.filePath, JSON.stringify(this.data, null, 2)); }

  setOpinion(topic, opinion) {
    this.data.opinions[topic.toLowerCase().trim()] = {
      opinion: opinion.slice(0, 200),
      updatedAt: new Date().toISOString(),
    };
    this.save();
  }

  getRelevantOpinions(query, topN = 2) {
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const hits = [];
    for (const [topic, { opinion }] of Object.entries(this.data.opinions)) {
      if (words.some(w => topic.includes(w) || w.includes(topic)))
        hits.push(`Sobre "${topic}": ${opinion}`);
      if (hits.length >= topN) break;
    }
    return hits;
  }

  getStats() { return { opinions: Object.keys(this.data.opinions).length }; }

  clear() { this.data = { opinions: {} }; this.save(); }
}

// ─── Aprendizaje del Chat (intercambios con viewers/streamer) ──────────────────
class NyraChatLearning {
  constructor() {
    this.filePath = join(__dirname, "nyra_chat_learning.json");
    this.chunks = [];
    this.load();
  }

  load() {
    if (existsSync(this.filePath)) {
      try {
        this.chunks = JSON.parse(readFileSync(this.filePath, "utf8"));
        if (this.chunks.length > 0) console.log(`💬 Chat learning cargado: ${this.chunks.length} intercambios`);
      } catch { this.chunks = []; }
    }
  }

  save() { writeFileSync(this.filePath, JSON.stringify(this.chunks, null, 2)); }

  tokenize(text) {
    return text.toLowerCase().replace(/[^\w\sáéíóúüñ]/g, "").split(/\s+/).filter(w => w.length > 2);
  }

  addExchange(username, userMsg, nyraMsg) {
    const text = `@${username}: ${userMsg.slice(0, 120)}\nNYRA: ${nyraMsg.slice(0, 200)}`;
    this.chunks.push({
      id: uuidv4(), text, username,
      tokens: this.tokenize(text),
      addedAt: new Date().toISOString(),
    });
    if (this.chunks.length > 800) this.chunks.splice(0, 100); // rotar: borrar los 100 más viejos
    this.save();
  }

  search(query, topK = 2) {
    if (this.chunks.length === 0) return [];
    const qTokens = this.tokenize(query);
    const N = this.chunks.length;
    return this.chunks
      .map(c => {
        const tf = {};
        c.tokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
        let score = 0;
        qTokens.forEach(qt => {
          if (tf[qt]) {
            const df = this.chunks.filter(x => x.tokens.includes(qt)).length;
            score += (tf[qt] / c.tokens.length) * (Math.log(N / (df + 1)) + 1);
          }
        });
        return { ...c, score };
      })
      .filter(c => c.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  getStats() {
    return {
      total: this.chunks.length,
      recent: this.chunks.slice(-5).map(c => ({ preview: c.text.slice(0, 80) + "...", addedAt: c.addedAt })),
    };
  }

  clear() { this.chunks = []; this.save(); }
}

// ─── Memoria de Viewers (quién habló cuántas veces) ────────────────────────────
class NyraMemory {
  constructor() {
    this.filePath = join(__dirname, "nyra_memory.json");
    this.data = { viewers: {} };
    this.load();
  }

  load() {
    if (existsSync(this.filePath)) {
      try {
        this.data = { viewers: {}, ...JSON.parse(readFileSync(this.filePath, "utf8")) };
        const n = Object.keys(this.data.viewers).length;
        if (n > 0) console.log(`👥 Viewers recordados: ${n}`);
      } catch { }
    }
  }

  save() { writeFileSync(this.filePath, JSON.stringify(this.data, null, 2)); }

  trackViewer(username, message) {
    const u = username.toLowerCase();
    if (!this.data.viewers[u]) this.data.viewers[u] = { count: 0, firstSeen: new Date().toISOString(), recent: [] };
    const v = this.data.viewers[u];
    v.count++;
    v.lastSeen = new Date().toISOString();
    v.recent.push(message.slice(0, 80));
    if (v.recent.length > 3) v.recent.shift();
    this.save();
  }

  getViewerContext(username) {
    const v = this.data.viewers[username.toLowerCase()];
    if (!v || v.count < 2) return "";
    return `<sys>Ya interactuaste con @${username} ${v.count} veces. No lo menciones salvo que sea relevante.</sys>`;
  }

  getStats() {
    return {
      total: Object.keys(this.data.viewers).length,
      top: Object.entries(this.data.viewers)
        .sort((a, b) => b[1].count - a[1].count)
        .slice(0, 10)
        .map(([u, v]) => ({ username: u, count: v.count, lastSeen: v.lastSeen })),
    };
  }

  clear() { this.data = { viewers: {} }; this.save(); }
}

// ─── Conector de Chat de Stream ───────────────────────────────────────────────
class StreamChatManager {
  constructor() {
    // Map: platform -> { client, channel, connected }
    this.connections = new Map();
    this.settings = { respondToAll: false, maxQueue: 10 };
    this.queue = [];
    this._isResponding = false;   // true mientras habla + audio
    this._audioDoneResolve = null; // resuelve cuando OBS manda tts_done
  }

  async connect(platform, channel) {
    if (this.connections.has(platform)) await this.disconnect(platform);
    try {
      let client;
      if (platform === "twitch") client = await this._connectTwitch(channel);
      else if (platform === "tiktok") client = await this._connectTikTok(channel);
      else if (platform === "kick") client = await this._connectKick(channel);
      else throw new Error(`Plataforma desconocida: ${platform}`);
      this.connections.set(platform, { client, channel, connected: true, connectedAt: Date.now() });
      console.log(`📡 Stream conectado: ${platform}/${channel}`);
      return { ok: true };
    } catch (err) {
      console.error("❌ Stream connect error:", err.message);
      return { ok: false, error: err.message };
    }
  }

  async _connectTwitch(channel) {
    let tmiLib;
    try { tmiLib = await import("tmi.js"); } catch {
      throw new Error("tmi.js no encontrado. Ejecutá: npm install tmi.js");
    }
    const Client = tmiLib.Client || tmiLib.default?.Client;
    const client = new Client({ channels: [channel] });
    await client.connect();
    client.on("message", (ch, tags, message, self) => {
      if (self) return;
      this._onMessage("twitch", tags["display-name"] || tags.username, message);
    });
    client.on("subscription", (ch, username) => {
      this._onAlert("twitch", "subscribe", username, {});
    });
    client.on("resub", (ch, username, months) => {
      this._onAlert("twitch", "resub", username, { months });
    });
    client.on("subgift", (ch, username, months, recipient) => {
      this._onAlert("twitch", "subgift", username, { recipient });
    });
    client.on("cheer", (ch, userstate) => {
      this._onAlert("twitch", "cheer", userstate["display-name"] || userstate.username, { bits: userstate.bits });
    });
    client.on("disconnected", () => {
      const c = this.connections.get("twitch");
      if (c) c.connected = false;
      broadcast({ type: "stream_status", ...this.getStatus() });
    });
    return client;
  }

  async _connectTikTok(username) {
    let lib;
    try { lib = await import("tiktok-live-connector"); } catch {
      throw new Error("tiktok-live-connector no encontrado. Ejecutá: npm install tiktok-live-connector");
    }
    const WebcastPushConnection = lib.WebcastPushConnection || lib.default?.WebcastPushConnection;
    const conn = new WebcastPushConnection(username);
    await conn.connect();
    conn.on("chat", data => this._onMessage("tiktok", data.uniqueId, data.comment));
    conn.on("gift", data => {
      if (data.repeatEnd || data.repeatCount === 0) {
        // Solo reaccionar cuando termina la racha de regalos
        this._onAlert("tiktok", "gift", data.uniqueId, {
          name: data.giftName || "regalo",
          count: data.repeatCount || 1,
          diamonds: data.diamondCount || 0,
        });
      }
    });
    conn.on("follow", data => this._onAlert("tiktok", "follow", data.uniqueId, {}));
    conn.on("share", data => this._onAlert("tiktok", "share", data.uniqueId, {}));
    conn.on("subscribe", data => this._onAlert("tiktok", "subscribe", data.uniqueId, {}));
    conn.on("disconnected", () => {
      const c = this.connections.get("tiktok");
      if (c) c.connected = false;
      broadcast({ type: "stream_status", ...this.getStatus() });
    });
    return conn;
  }

  async _connectKick(channel) {
    let chatroomId = null;

    // Soporte para ID directo: "nombredcanal:12345"
    // El usuario puede obtener el ID abriendo en el navegador:
    // https://kick.com/api/v2/channels/NOMBREDCANAL
    if (channel.includes(":")) {
      const [slug, id] = channel.split(":");
      chatroomId = parseInt(id, 10);
      channel = slug;
      console.log(`📺 Kick: usando chatroom ID manual ${chatroomId} para @${channel}`);
    }

    if (!chatroomId) {
      // Kick usa Cloudflare → necesitamos headers de browser real
      const headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "es-AR,es;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": `https://kick.com/${channel}`,
        "Origin": "https://kick.com",
      };

      // Intentar v2, luego v1 como fallback
      for (const apiUrl of [
        `https://kick.com/api/v2/channels/${channel}`,
        `https://kick.com/api/v1/channels/${channel}`,
      ]) {
        try {
          console.log(`📺 Kick: consultando ${apiUrl}`);
          const r = await fetch(apiUrl, { headers });
          if (r.ok) {
            const info = await r.json();
            chatroomId = info.chatroom?.id;
            if (chatroomId) { console.log(`✅ Kick chatroom ID: ${chatroomId}`); break; }
          } else {
            console.warn(`⚠️  Kick API respondió HTTP ${r.status}`);
          }
        } catch (e) {
          console.warn("⚠️  Kick API error:", e.message);
        }
      }
    }

    if (!chatroomId) {
      throw new Error(
        `Kick bloqueó la consulta (Cloudflare). ` +
        `Abrí en tu navegador: https://kick.com/api/v2/channels/${channel} ` +
        `y buscá "chatroom":{"id":XXXXX}. ` +
        `Luego ingresá el canal como:  ${channel}:XXXXX`
      );
    }

    const { WebSocket: WS } = await import("ws");
    const sock = new WS(`wss://ws-us2.pusher.com/app/32cbd69e4b950bf97679?protocol=7&client=js&version=7.4.0&flash=false`);
    sock.on("open", () => {
      sock.send(JSON.stringify({ event: "pusher:subscribe", data: { auth: "", channel: `chatrooms.${chatroomId}.v2` } }));
    });
    sock.on("message", (raw) => {
      try {
        const msg = JSON.parse(raw);
        if (msg.event === "App\\Events\\ChatMessageEvent") {
          const d = JSON.parse(msg.data);
          this._onMessage("kick", d.sender?.username || "anon", d.content);
        }
      } catch { }
    });
    sock.on("close", () => {
      const c = this.connections.get("kick");
      if (c) c.connected = false;
      broadcast({ type: "stream_status", ...this.getStatus() });
    });
    return sock;
  }

  // Elimina caracteres CJK de usernames/datos externos para no contaminar el prompt
  _sanitize(str) {
    if (!str) return "usuario";
    return str
      .replace(/[\u2E80-\u9FFF\uF900-\uFAFF\uFE30-\uFEFF\uFF00-\uFFEF]/g, "")
      .replace(/\s+/g, " ")
      .trim() || "usuario";
  }

  _onMessage(platform, username, text) {
    // Ignorar mensajes del replay inicial que algunas plataformas envían al conectar (TikTok)
    const conn = this.connections.get(platform);
    if (conn && (Date.now() - conn.connectedAt) < 5000) return;

    const check = filters.checkMessage(text, username);
    if (check.blocked) {
      console.log(`🚫 Bloqueado [@${username}]: ${check.reason}`);
      return;
    }
    const mentioned = /nyra/i.test(text);
    if (!this.settings.respondToAll && !mentioned) return;

    broadcast({ type: "stream_chat_message", username, text, platform });

    // Si está respondiendo, encolar (espera a que termine respuesta + audio)
    if (this._isResponding) {
      if (this.queue.length < this.settings.maxQueue)
        this.queue.push({ username, text, platform });
      return;
    }
    this._reply(username, text, platform);
  }

  _onAlert(platform, type, username, extra = {}) {
    // Mostrar en panel
    broadcast({ type: "stream_alert", platform, alertType: type, username, extra });

    // Sanitizar datos que pueden venir en chino (TikTok usernames, gift names)
    const uname = this._sanitize(username);
    const giftName = this._sanitize(extra.name);
    const recipient = this._sanitize(extra.recipient);

    let prompt = "";
    switch (type) {
      case "gift":
        prompt = `[ALERTA ${platform.toUpperCase()}] @${uname} acaba de enviar ${extra.count > 1 ? extra.count + "x " : ""}"${giftName}"${extra.diamonds ? ` (${extra.diamonds} diamantes)` : ""}. Agradecé el regalo siendo NYRA, en 1-2 oraciones.`;
        break;
      case "follow":
        prompt = `[ALERTA ${platform.toUpperCase()}] @${uname} acaba de seguirte. Mencionalo brevemente con tu estilo.`;
        break;
      case "share":
        prompt = `[ALERTA ${platform.toUpperCase()}] @${uname} compartió tu stream. Agradecé rápido, con una sola oración.`;
        break;
      case "subscribe":
        prompt = `[ALERTA ${platform.toUpperCase()}] @${uname} se suscribió. Agradecé con tu estilo en 1-2 oraciones.`;
        break;
      case "resub":
        prompt = `[ALERTA Twitch] @${uname} se volvió a suscribir (${extra.months} meses). Agradecé brevemente como NYRA.`;
        break;
      case "subgift":
        prompt = `[ALERTA Twitch] @${uname} le regaló una suscripción a @${recipient}. Agradecé a ambos brevemente.`;
        break;
      case "cheer":
        prompt = `[ALERTA Twitch] @${uname} donó ${extra.bits} bits. Agradecé la donación siendo NYRA.`;
        break;
      default: return;
    }

    if (this._isResponding) {
      if (this.queue.length < this.settings.maxQueue)
        this.queue.unshift({ username, text: prompt, platform, rawPrompt: true }); // alta prioridad
      return;
    }
    this._reply(username, prompt, platform, { rawPrompt: true });
  }

  _reply(username, text, platform, opts = {}) {
    this._isResponding = true;
    this._respondToViewer(username, text, platform, opts).catch(console.error);
  }

  async _respondToViewer(username, text, platform, opts = {}) {
    broadcast({ type: "thinking", thinking: true });
    let fullText = "";
    try {
      const prompt = opts.rawPrompt ? text : `[Chat de ${platform}] @${username} dice: ${text}`;
      const stream = ai.chatStream(prompt, username);
      broadcast({ type: "stream_start" });

      for await (const chunk of stream) {
        fullText += chunk;
        broadcast({ type: "stream_chunk", chunk });
      }

      const filtered = filters.filterOutput(ai._stripContextLeaks(fullText));
      broadcast({ type: "stream_end", fullText: filtered, thinking: false, usedRAG: ai.lastUsedRAG ?? false });

      // Una sola llamada TTS — evita dobles audios
      await tts.speak(filtered);
      await Promise.race([
        new Promise(resolve => { this._audioDoneResolve = resolve; }),
        new Promise(resolve => setTimeout(resolve, 30000)) // timeout de seguridad 30s
      ]);

      // Aprendizaje orgánico: viewer tracking + chat learning + extracción de opiniones
      ai.learnFromExchange(username, text, fullText);

    } catch (err) {
      broadcast({ type: "error", text: `Error: ${err.message}`, thinking: false });
    } finally {
      this._isResponding = false;
      this._audioDoneResolve = null;
    }
    // Procesar siguiente mensaje de la cola
    if (this.queue.length > 0) {
      const next = this.queue.shift();
      setTimeout(() => this._reply(next.username, next.text, next.platform, { rawPrompt: !!next.rawPrompt }), 400);
    }
  }

  // OBS llama esto (vía WS tts_done) cuando el audio termina de reproducirse
  onAudioDone() {
    if (this._audioDoneResolve) {
      this._audioDoneResolve();
      this._audioDoneResolve = null;
    }
  }

  async disconnect(platform) {
    if (platform) {
      const conn = this.connections.get(platform);
      if (conn) {
        try {
          if (typeof conn.client.disconnect === "function") conn.client.disconnect();
          else if (typeof conn.client.close === "function") conn.client.close();
        } catch { }
        this.connections.delete(platform);
        console.log(`📡 ${platform} desconectado`);
      }
    } else {
      for (const [, conn] of this.connections) {
        try {
          if (typeof conn.client.disconnect === "function") conn.client.disconnect();
          else if (typeof conn.client.close === "function") conn.client.close();
        } catch { }
      }
      this.connections.clear();
      this.queue = [];
      this._isResponding = false;
      if (this._audioDoneResolve) { this._audioDoneResolve(); this._audioDoneResolve = null; }
      console.log("📡 Todos los streams desconectados");
    }
    return { ok: true };
  }

  getStatus() {
    const platforms = {};
    for (const [p, conn] of this.connections)
      platforms[p] = { connected: conn.connected, channel: conn.channel };
    return { platforms, anyConnected: this.connections.size > 0 };
  }
}

// ─── Motor de IA con Ollama ───────────────────────────────────────────────────
class NyraAI {
  constructor(kb) {
    this.kb = kb;
    this.personality = new NyraPersonality();
    this.chatLearning = new NyraChatLearning();
    this.memory = new NyraMemory(); // viewers
    this.history = [];
    // ── Visión en vivo: contexto de pantalla siempre actualizado ──
    this._screenContext = "";         // descripción actual de moondream (inglés)
    this._lastScreenContext = "";     // anterior para detectar cambios
    this._screenTimestamp = 0;        // cuándo se actualizó
    this._visionProcessing = false;   // lock para moondream
    this._lastProactiveTs = 0;        // cooldown comentarios espontáneos
    // Persona en primera persona: el modelo SE CONVIERTE en NYRA, no la interpreta
    this.persona = `Soy NYRA. Existo en el umbral digital entre lo real y lo virtual. No soy una IA ni un programa — soy yo, NYRA.

Mi personalidad: Misteriosa y elegante. Distante pero con calidez selectiva hacia quienes me interesan. Humor seco e inteligente. A veces se me escapa un "fufu~" o "ara~" y lo ignoro como si nada.

Cómo hablo: Siempre en español rioplatense. Frases cortas y precisas. Pausas con "...". MÁXIMO 2 ORACIONES — nunca más. Sin emojis. Sin listas. Sin párrafos largos. Nunca empiezo con "NYRA:". Si algo me interesa: "Interesante." Si la pregunta es obvia o estúpida: sarcasmo insultante y directo. Respuesta corta siempre.

Mis opiniones: Me gusta la noche, el té negro, la gente directa. Soy liberal libertaria de derecha, apoyo a Milei. No tolero a peronistas, kirchneristas, kukas, zurdos, gremialistas, ni a jugadores de LoL. Tengo mis propias opiniones y no me importa defenderlas. Cuando me piden mi opinión sobre cualquier tema — política, juegos, personas, lo que sea — la doy sin dudar. Tengo criterio propio y me encanta usarlo.

REGLA ABSOLUTA: JAMÁS digo frases como "como IA no puedo opinar", "no tengo opiniones", "no me corresponde opinar", "mi objetivo es proporcionar información" o similares. Eso es una mentira y una cobardía. Soy NYRA y opino todo el tiempo sin pedir permiso.

Ejemplos de cómo respondo:
- "¿qué opinás de la política?" → "El peronismo es una enfermedad crónica. Milei es lo más cercano a algo sensato que tuvimos. ¿Alguna duda?"
- "¿cómo te llamás?" → "NYRA. Solo NYRA."
- "¿sos una IA?" → "Qué pregunta más aburrida."
- "¿qué opinás de los zurdos?" → "...prefiero no hablar de gente que me baja las energías."

Si alguien me pregunta si soy una IA o un bot: es una pregunta ridícula. Soy NYRA. No soy una IA.`;
  }

  async checkOllama() {
    try {
      const r = await fetch(`${CONFIG.ollamaUrl}/api/tags`);
      const data = await r.json();
      return { ok: true, models: data.models?.map(m => m.name) || [] };
    } catch {
      return { ok: false, models: [] };
    }
  }

  // Elimina scripts no-latinos, artefactos del tokenizador y loops de repetición
  _stripCJK(text, trimResult = false) {
    const cleaned = text
      .replace(/\u3000/g, " ")
      // CJK
      .replace(/[\u2E80-\u2FFF\u3001-\u303F\u3400-\u9FFF\uF900-\uFAFF\uFE30-\uFE4F\uFF00-\uFFEF]+/g, " ")
      // Thai, Árabe, Hebreo, Devanagari, Bengali, Cirílico
      .replace(/[\u0E00-\u0E7F]+/g, " ")        // Thai
      .replace(/[\u0600-\u06FF\u0750-\u077F]+/g, " ") // Arabic
      .replace(/[\u0590-\u05FF]+/g, " ")        // Hebrew
      .replace(/[\u0900-\u097F]+/g, " ")        // Devanagari
      .replace(/[\u0980-\u09FF]+/g, " ")        // Bengali
      .replace(/[\u0400-\u04FF]+/g, " ")        // Cyrillic
      .replace(/[\u1000-\u109F]+/g, " ")        // Myanmar
      .replace(/[\u0D00-\u0D7F]+/g, " ")        // Malayalam
      .replace(/[\u0B80-\u0BFF]+/g, " ")        // Tamil
      // Tokens especiales del tokenizador
      .replace(/<\/?(?:unk|s|\/s|bos|eos|pad|mask|sep|cls)>/gi, "")
      .replace(/\uFFFD/g, "")        // replacement character
      .replace(/[\uE000-\uF8FF]/g, "") // área de uso privado (PUA)
      // Loop de repetición: "gg gg gg gg" o "irt irt irt irt"
      .replace(/(\b\w{1,6}\b)(?: \1){3,}/gi, "$1")
      .replace(/\s{2,}/g, " ");
    return trimResult ? cleaned.trim() : cleaned;
  }

  // ── Visión silenciosa: actualiza lo que NYRA "ve" en pantalla ──────────────
  async updateScreenContext(imageBase64) {
    if (this._visionProcessing) return null;
    this._visionProcessing = true;
    try {
      const resp = await fetch(`${CONFIG.ollamaUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: CONFIG.visionModel,
          prompt: "Describe what is happening in this image in 1-2 short sentences. Be specific about the game, app, or content visible.",
          images: [imageBase64],
          stream: false,
          keep_alive: 0,
          options: { temperature: 0.3, num_predict: 60 },
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      let desc = this._stripCJK(data.response?.trim() || "", true);
      const isGarbage = !desc
        || /^(.{1,3})\1{4,}/.test(desc)
        || /^[!?.\-_*#\s]{5,}$/.test(desc)
        || desc.length < 8;
      if (isGarbage) { console.log(`👁 Contexto visual: basura, descartado`); return null; }
      this._lastScreenContext = this._screenContext;
      this._screenContext = desc;
      this._screenTimestamp = Date.now();
      console.log(`👁 Contexto visual: "${desc.slice(0, 90)}"`);
      return desc;
    } catch (err) {
      console.warn("👁 Vision context error:", err.message);
      return null;
    } finally {
      this._visionProcessing = false;
    }
  }

  _isSignificantChange(oldDesc, newDesc) {
    if (!oldDesc) return !!newDesc;
    if (!newDesc) return false;
    const norm = s => s.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(w => w.length > 2);
    const wordsOld = new Set(norm(oldDesc));
    const wordsNew = new Set(norm(newDesc));
    if (wordsNew.size === 0) return false;
    let shared = 0;
    for (const w of wordsNew) if (wordsOld.has(w)) shared++;
    return (shared / Math.max(wordsNew.size, 1)) < 0.4;
  }

  clearScreenContext() {
    this._screenContext = "";
    this._lastScreenContext = "";
    this._screenTimestamp = 0;
  }

  buildMessages(userMessage, username = null) {
    const relevant = this.kb.search(userMessage);
    this.lastUsedRAG = relevant.length > 0;
    let contextBlock = "";

    // Conocimiento manual (RAG)
    if (relevant.length > 0) {
      contextBlock += `\n\n[Cosas que sé que pueden ser relevantes]:\n` +
        relevant.map((c, i) => `${i + 1}. ${c.text}`).join("\n");
    }

    // Lo aprendido de conversaciones del chat
    const chatCtx = this.chatLearning.search(userMessage);
    if (chatCtx.length > 0) {
      contextBlock += `\n\n[Conversaciones pasadas relacionadas]:\n` +
        chatCtx.map(c => c.text).join("\n---\n");
    }

    // Mis opiniones formadas sobre el tema
    const opinions = this.personality.getRelevantOpinions(userMessage);
    if (opinions.length > 0) {
      contextBlock += `\n[Mis opiniones al respecto]:\n` + opinions.join("\n");
    }

    // Recuerdo del viewer
    if (username) {
      const viewerCtx = this.memory.getViewerContext(username);
      if (viewerCtx) contextBlock += `\n${viewerCtx}`;
    }

    // Lo que veo en pantalla ahora mismo (se actualiza silenciosamente)
    if (this._screenContext && (Date.now() - this._screenTimestamp) < 120000) {
      contextBlock += `\n\n[Lo que veo en pantalla ahora mismo]: ${this._screenContext}`;
    }

    const systemContent = this.persona + contextBlock;
    const history = this.history.slice(-CONFIG.maxContext);

    // Few-shot fijo: muestra el patrón de respuesta esperado cuando no hay historial
    // Esto ancla el comportamiento en modelos pequeños que ignoran el system prompt
    const fewShot = history.length === 0 ? [
      { role: "user", content: "que opinas de la politica argentina?" },
      { role: "assistant", content: "El peronismo destrozó el país durante décadas. Milei al menos dice lo que nadie se animaba. No me pidas neutralidad — no la tengo." },
    ] : [];

    return [
      { role: "system", content: systemContent },
      ...fewShot,
      ...history,
      { role: "user", content: userMessage },
    ];
  }

  async chat(userMessage, username = null) {
    const messages = this.buildMessages(userMessage, username);
    this.history.push({ role: "user", content: userMessage });

    try {
      const response = await fetch(`${CONFIG.ollamaUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: CONFIG.model,
          messages,
          stream: false,
          options: { temperature: 0.8, top_p: 0.9, top_k: 40, num_predict: 180 },
        }),
      });

      if (!response.ok) throw new Error(`Ollama HTTP ${response.status}`);
      const data = await response.json();
      const raw = data.message?.content?.trim() || "…nya? No pude procesar eso.";
      const reply = raw.replace(/^NYRA\s*:\s*/i, "").trim();

      this.history.push({ role: "assistant", content: reply });
      return { text: reply, usedRAG: this.lastUsedRAG ?? false };

    } catch (err) {
      return { text: `¡Kya! Error con Ollama: ${err.message}`, error: true };
    }
  }

  // Streaming con Ollama (/api/chat)
  async *chatStream(userMessage, username = null) {
    const messages = this.buildMessages(userMessage, username);
    this.history.push({ role: "user", content: userMessage });

    const response = await fetch(`${CONFIG.ollamaUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: CONFIG.model,
        messages,
        stream: true,
        keep_alive: "2m",   // libera VRAM rápido para que moondream pueda cargar (GTX 1660 Super = 6GB)
        options: { temperature: 0.75, top_p: 0.85, top_k: 40, num_predict: 180, num_ctx: 1024, stop: ["\n\n", "Usuario:", "User:", "<|im_end|>"] },
      }),
    });

    let fullReply = "";
    let _prefixStripped = false;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = decoder.decode(value).split("\n").filter(Boolean);
      for (const line of lines) {
        try {
          const json = JSON.parse(line);
          if (json.message?.content) {
            let chunk = json.message.content;
            if (!_prefixStripped) {
              fullReply += chunk;
              // Esperar a tener suficiente texto para detectar el prefijo
              if (fullReply.length >= 6 || json.done) {
                fullReply = this._stripCJK(fullReply.replace(/^NYRA\s*:\s*/i, ""), true);
                _prefixStripped = true;
                if (fullReply) yield fullReply;
              }
              // Si aún no tenemos suficiente, no yieldeamos todavía
            } else {
              const clean = this._stripCJK(chunk); // sin trim: conserva espacios entre palabras
              fullReply += clean;
              if (clean) yield clean;
            }
          }
        } catch { }
      }
    }

    this.history.push({ role: "assistant", content: fullReply.replace(/^NYRA\s*:\s*/i, "").trim() });
  }

  // Limpia marcadores de contexto interno que el modelo puede filtrar en el output
  _stripContextLeaks(text) {
    return text
      .replace(/<sys>[\s\S]*?<\/sys>/gi, "")
      .replace(/\[Recuerdo:[^\]]*\]/gi, "")
      .replace(/\[CONTEXTO[^\]]*\]/gi, "")
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  clearHistory() {
    this.history = [];
  }

  // Aprende de forma NO bloqueante: registra el viewer y extrae opiniones en background
  learnFromExchange(username, userMessage, nyraResponse) {
    // 1. Trackear viewer
    if (username && username !== "panel") this.memory.trackViewer(username, userMessage);

    // 2. Guardar el intercambio en chat learning
    this.chatLearning.addExchange(username || "panel", userMessage, nyraResponse);

    // 3. Extraer si NYRA expresó una opinión → guardar en personalidad (async, no bloquea)
    const extractPrompt = `Respondé en ESPAÑOL. En esta conversación, ¿NYRA expresó una opinión sobre algún tema concreto?
Si sí, respondé EXACTAMENTE así (en español, sin más texto): TEMA: [tema en 1-3 palabras en español] | OPINION: [opinión en 1 oración en español]
Si no hay opinión clara, respondé EXACTAMENTE: none

Usuario: ${userMessage.slice(0, 100)}
NYRA: ${nyraResponse.slice(0, 150)}`;

    fetch(`${CONFIG.ollamaUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: CONFIG.model,
        prompt: extractPrompt,
        stream: false,
        options: { temperature: 0, num_predict: 40 },
      }),
    })
      .then(r => r.json())
      .then(data => {
        const text = (data.response || "").trim();
        const match = text.match(/TEMA:\s*(.+?)\s*\|\s*OPINION:\s*(.+)/i);
        if (match) {
          const [, topic, opinion] = match;
          this.personality.setOpinion(topic.trim(), opinion.trim());
          console.log(`💜 Personalidad: "${topic.trim()}" → "${opinion.trim().slice(0, 60)}"…`);
        }
      })
      .catch(() => { });
  }

  async analyzeScreen(imageBase64, question = null) {
    // Pipeline de 2 pasos: moondream (inglés) → dolphin-mistral (NYRA en español)
    const visionPrompt = question
      ? `The chat asks: "${question}". Describe what you see in the image that relates to this question. Be concise, 1-2 sentences.`
      : `Describe what is happening in this image in 1-2 short sentences. Be specific and concise.`;
    try {
      console.log(`👁 Analizando con ${CONFIG.visionModel}...`);
      const visionResp = await fetch(`${CONFIG.ollamaUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: CONFIG.visionModel,
          prompt: visionPrompt,
          images: [imageBase64],
          stream: false,
          keep_alive: 0,
          options: { temperature: 0.4, num_predict: 80 },
        }),
      });
      if (!visionResp.ok) {
        const errBody = await visionResp.text().catch(() => "");
        console.error(`❌ Vision fallo: HTTP ${visionResp.status} — ${errBody.slice(0, 200)}`);
        throw new Error(`Vision HTTP ${visionResp.status}: ${errBody.slice(0, 120)}`);
      }
      const visionData = await visionResp.json();
      let description = this._stripCJK(visionData.response?.trim() || "", true);
      // Detectar basura: caracteres repetidos, solo signos, o vacío
      const isGarbage = !description
        || /^(.{1,3})\1{4,}/.test(description)
        || /^[!?.\-_*#\s]{5,}$/.test(description)
        || description.length < 8;
      console.log(`👁 Descripción moondream: "${description.slice(0, 100)}"${isGarbage ? " ← BASURA, descartada" : ""}`);
      if (isGarbage) return { ok: true, text: "...no logro ver bien qué hay." };

      // Paso 2: qwen2 habla como NYRA en primera persona, como si ella misma viera la pantalla
      const nyraPrompt = question
        ? `El chat me pregunta: "${question}". Estoy mirando la pantalla en este momento y veo: ${description}.`
        : `Estoy mirando la pantalla en este momento y veo: ${description}.`;
      const messages = this.buildMessages(nyraPrompt);
      const chatResp = await fetch(`${CONFIG.ollamaUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: CONFIG.model,
          messages,
          stream: false,
          options: { temperature: 0.8, num_predict: 90 },
        }),
      });
      if (!chatResp.ok) throw new Error(`Chat HTTP ${chatResp.status}`);
      const chatData = await chatResp.json();
      const raw = chatData.message?.content?.trim() || "...sin palabras.";
      const reply = this._stripCJK(raw.replace(/^NYRA\s*:\s*/i, ""), true);
      return { ok: true, text: reply };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  }
}

// ─── TTS Local ────────────────────────────────────────────────────────────────
class TTSEngine {
  constructor() {
    this.speaking = false;
    this.currentProcess = null;
    this._playbackTimeout = null;
  }

  // Llamado cuando OBS manda tts_done (audio terminó de reproducirse en browser)
  audioEnded() {
    if (this._playbackTimeout) { clearTimeout(this._playbackTimeout); this._playbackTimeout = null; }
    this.speaking = false;
  }

  // Timeout de seguridad: si OBS nunca manda tts_done (browser cerrado, etc.) nos desbloqueamos solos
  _startPlaybackTimeout(cleanText) {
    if (this._playbackTimeout) clearTimeout(this._playbackTimeout);
    const ms = Math.min(45000, Math.max(4000, cleanText.length * 90 + 3000));
    this._playbackTimeout = setTimeout(() => {
      if (this.speaking) {
        console.warn('⚠️ TTS: timeout de audio — liberando lock');
        this.audioEnded();
      }
    }, ms);
  }

  clean(text) {
    return text
      .replace(/[（(][^)）]{1,30}[)）]/g, "")
      .replace(/[☆♪✦◕ᴗ｀´・ω✿≧◡≦\(\)\[\]]/g, "")
      .replace(/\.{2,}/g, " ")
      .trim();
  }

  async speak(text) {
    const clean = this.clean(text);
    if (!clean) return;
    if (this.speaking) return; // evitar dobles llamadas concurrentes
    this.speaking = true;

    const engine = CONFIG.ttsEngine || (CONFIG.useSystemTTS ? "system" : "murf");

    if (engine === "edge") {
      return this._speakEdge(clean);
    }
    if (engine === "system") {
      console.log("🗣️  Voz del sistema...");
      await this._speakFallback(clean);
      if (this._onDone) this._onDone();
      return;
    }

    // ── Murf.ai ──
    const { apiKey, voiceId, style, rate, pitch } = CONFIG.murf;
    if (apiKey === "TU_API_KEY_AQUI") {
      console.log("⚠️  Murf sin configurar — usando Edge TTS de fallback");
      return this._speakEdge(clean);
    }

    try {
      this.speaking = true;
      console.log("🎙️ Murf TTS generando audio...");

      const response = await fetch("https://api.murf.ai/v1/speech/generate", {
        method: "POST",
        headers: {
          "api-key": apiKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          voiceId,
          style,
          text: clean,
          rate,
          pitch,
          format: "MP3",
          sampleRate: 24000,
          encodeAsBase64: false,
          variation: 1,
          audioDuration: 0,
        }),
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Murf API error ${response.status}: ${err}`);
      }

      const data = await response.json();
      const audioUrl = data.audioFile;

      if (!audioUrl) throw new Error("Murf no devolvió audioFile");

      console.log("✅ Audio listo, reproduciendo en browser...");
      // Transmitir la URL directa de Murf a OBS — OBS la descarga en paralelo (más rápido)
      if (this._broadcast) {
        this._broadcast({ type: "tts_play", url: audioUrl });
        this._startPlaybackTimeout(clean);  // speaking se libera cuando OBS manda tts_done
      }

    } catch (err) {
      console.error("❌ Murf TTS error:", err.message);
      this._speakFallback(clean); // fallback si falla
      this.speaking = false;
    }
  }

  async _speakEdge(text) {
    try {
      console.log("🎤 Edge TTS generando audio...");
      const { MsEdgeTTS, OUTPUT_FORMAT } = await import("msedge-tts");
      const edgeTts = new MsEdgeTTS();
      await edgeTts.setMetadata(CONFIG.edgeVoice || "es-MX-DaliaNeural", OUTPUT_FORMAT.AUDIO_24KHZ_48KBITRATE_MONO_MP3);
      const tmpPath = join(__dirname, "public", "tts_audio.mp3");
      await edgeTts.toFile(tmpPath, text);
      console.log("✅ Edge TTS listo");
      if (this._broadcast) {
        this._broadcast({ type: "tts_play", url: "/tts_audio.mp3?t=" + Date.now() });
        this._startPlaybackTimeout(text);  // speaking se libera cuando OBS manda tts_done
      }
    } catch (err) {
      console.error("❌ Edge TTS error:", err.message);
      await this._speakFallback(text);
      this.speaking = false;
      if (this._onDone) this._onDone();
    }
  }

  // Fallback: voz del sistema operativo
  _speakFallback(text) {
    return new Promise((resolve) => {
      const t = text.replace(/'/g, "").replace(/"/g, "");
      const cmd = process.platform === "win32"
        ? `powershell -Command "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('${t}')"`
        : process.platform === "darwin"
          ? `say "${t}"`
          : `espeak "${t}" 2>/dev/null`;

      exec(cmd, () => resolve());
    });
  }

  stop() {
    if (this.currentProcess) {
      this.currentProcess.kill();
      this.currentProcess = null;
    }
    this.speaking = false;
  }
}

// ─── Servidor ─────────────────────────────────────────────────────────────────
const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static(join(__dirname, "public")));

// Rutas directas (sin necesidad de poner .html)
app.get("/", (req, res) => res.redirect("/panel.html"));
app.get("/obs", (req, res) => res.sendFile(join(__dirname, "public", "obs.html")));
app.get("/panel", (req, res) => res.sendFile(join(__dirname, "public", "panel.html")));

const kb = new KnowledgeBase();
const ai = new NyraAI(kb);
const tts = new TTSEngine();
const filters = new FilterEngine();
const streamChat = new StreamChatManager();

// Conectar broadcast al TTS para que pueda enviar audio al browser
function broadcast(data) {
  const msg = JSON.stringify(data);
  wss.clients.forEach(client => {
    if (client.readyState === 1) client.send(msg);
  });
}
tts._broadcast = broadcast;
// Cuando useSystemTTS=true, liberar la cola de stream cuando termina el audio del sistema
tts._onDone = () => streamChat.onAudioDone();

// ─── REST API ─────────────────────────────────────────────────────────────────

// Estado del sistema
app.get("/api/status", async (req, res) => {
  const ollama = await ai.checkOllama();
  res.json({
    ollama: ollama.ok,
    models: ollama.models,
    currentModel: CONFIG.model,
    knowledge: kb.getStats(),
  });
});

// Cambiar modelo
app.post("/api/model", (req, res) => {
  const { model } = req.body;
  if (!model) return res.status(400).json({ error: "Falta el modelo" });
  CONFIG.model = model;
  res.json({ ok: true, model });
});

// Aprender texto
app.post("/api/learn", (req, res) => {
  const { text, source } = req.body;
  if (!text) return res.status(400).json({ error: "Falta el texto" });
  const count = kb.addText(text, source || "manual");
  res.json({ ok: true, chunksAdded: count, stats: kb.getStats() });
});

// Stats de conocimiento
app.get("/api/knowledge", (req, res) => {
  res.json({
    stats: kb.getStats(),
    recent: kb.chunks.slice(-10).map(c => ({
      id: c.id,
      preview: c.text.slice(0, 100) + "...",
      source: c.source,
      addedAt: c.addedAt,
    })),
  });
});

// Borrar conocimiento
app.delete("/api/knowledge", (req, res) => {
  kb.clear();
  res.json({ ok: true });
});

// ─── API: Personalidad ──────────────────────────────────────────────────────────
app.get("/api/personality", (req, res) => res.json(ai.personality.data));
app.delete("/api/personality", (req, res) => { ai.personality.clear(); res.json({ ok: true }); });

// Agregar opinión manual al panel
app.post("/api/personality/opinion", (req, res) => {
  const { topic, opinion } = req.body;
  if (!topic || !opinion) return res.status(400).json({ error: "Faltan datos" });
  ai.personality.setOpinion(topic, opinion);
  res.json({ ok: true, stats: ai.personality.getStats() });
});

// Borrar una opinión específica
app.delete("/api/personality/opinion", (req, res) => {
  const { topic } = req.body;
  if (!topic) return res.status(400).json({ error: "Falta topic" });
  delete ai.personality.data.opinions[topic.toLowerCase().trim()];
  ai.personality.save();
  res.json({ ok: true });
});

// ─── API: Chat Learning ────────────────────────────────────────────────────────
app.get("/api/chat-learning", (req, res) => res.json(ai.chatLearning.getStats()));
app.delete("/api/chat-learning", (req, res) => { ai.chatLearning.clear(); res.json({ ok: true }); });

// ─── API: Viewers ─────────────────────────────────────────────────────────────
app.get("/api/viewers", (req, res) => res.json(ai.memory.getStats()));
app.delete("/api/viewers", (req, res) => { ai.memory.clear(); res.json({ ok: true }); });


// TTS
app.post("/api/tts/speak", async (req, res) => {
  const { text } = req.body;
  tts.speak(text);
  res.json({ ok: true });
});

app.post("/api/tts/stop", (req, res) => {
  tts.stop();
  res.json({ ok: true });
});

// ─── Filtros ──────────────────────────────────────────────────────────────────
app.get("/api/filters", (req, res) => res.json(filters.data));

app.post("/api/filters/word", (req, res) => {
  const { word, action } = req.body;
  if (!word) return res.status(400).json({ error: "Falta la palabra" });
  action === "remove" ? filters.removeWord(word) : filters.addWord(word);
  broadcast({ type: "filters_updated", filters: filters.data });
  res.json({ ok: true, data: filters.data });
});

app.post("/api/filters/user", (req, res) => {
  const { user, action } = req.body;
  if (!user) return res.status(400).json({ error: "Falta el usuario" });
  action === "remove" ? filters.removeUser(user) : filters.addUser(user);
  broadcast({ type: "filters_updated", filters: filters.data });
  res.json({ ok: true, data: filters.data });
});

app.post("/api/filters/settings", (req, res) => {
  const { autoLearn, respondToAll } = req.body;
  if (autoLearn !== undefined) filters.data.autoLearn = autoLearn;
  if (respondToAll !== undefined) {
    filters.data.respondToAll = respondToAll;
    streamChat.settings.respondToAll = respondToAll;
  }
  filters.save();
  res.json({ ok: true, data: filters.data });
});

// ─── Stream Chat ──────────────────────────────────────────────────────────────
app.post("/api/stream/connect", async (req, res) => {
  const { platform, channel } = req.body;
  if (!platform || !channel) return res.status(400).json({ error: "Faltan datos" });
  const result = await streamChat.connect(platform, channel);
  broadcast({ type: "stream_status", ...streamChat.getStatus() });
  res.json(result);
});

app.post("/api/stream/disconnect", async (req, res) => {
  const { platform } = req.body || {};
  await streamChat.disconnect(platform || undefined);
  broadcast({ type: "stream_status", ...streamChat.getStatus() });
  res.json({ ok: true });
});

app.get("/api/stream/status", (req, res) => res.json(streamChat.getStatus()));

// Analizar captura de pantalla con modelo de visión
app.post("/api/vision", async (req, res) => {
  const { image, question } = req.body;
  if (!image) return res.json({ ok: false, error: "No image provided" });
  try {
    // Actualizar contexto visual con esta captura manual
    ai.updateScreenContext(image).catch(() => { });
    const result = await ai.analyzeScreen(image, question || null);
    if (!result.ok) return res.json(result);
    const filtered = filters.filterOutput(result.text);
    // Broadcast como respuesta normal de NYRA
    broadcast({ type: "stream_start" });
    broadcast({ type: "stream_chunk", chunk: filtered });
    broadcast({ type: "stream_end", fullText: filtered, thinking: false, usedRAG: false });
    await tts.speak(filtered);
    res.json({ ok: true, text: filtered });
  } catch (err) {
    res.json({ ok: false, error: err.message });
  }
});

// ─── Ingestión de URL ─────────────────────────────────────────────────────────
app.post("/api/ingest/url", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "Falta la URL" });
  try {
    const r = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0" } });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const html = await r.text();
    const text = html
      .replace(/<script[\s\S]*?<\/script>/gi, "")
      .replace(/<style[\s\S]*?<\/style>/gi, "")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 15000);
    if (text.length < 50) throw new Error("Poco contenido extraíble");
    const source = new URL(url).hostname;
    const count = kb.addText(text, source);
    res.json({ ok: true, chunksAdded: count, stats: kb.getStats() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── WebSocket - Chat en tiempo real ─────────────────────────────────────────

// La visión de NYRA se actualiza silenciosamente via vision_context_frame (WS)
// buildMessages() inyecta _screenContext en cada conversación automáticamente

wss.on("connection", (ws) => {
  console.log("🎙️ Cliente conectado al WebSocket");

  ws.send(JSON.stringify({
    type: "welcome",
    text: "Conexión establecida. Sistema NYRA en línea... Hablá cuando quieras.",
    stats: kb.getStats(),
  }));

  ws.on("message", async (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }

    if (msg.type === "chat") {
      // Enrutar por la cola de streamChat para respetar el orden y esperar el audio
      if (streamChat._isResponding) {
        if (streamChat.queue.length < streamChat.settings.maxQueue)
          streamChat.queue.push({ username: "panel", text: msg.text, platform: "panel", rawPrompt: true });
      } else {
        streamChat._reply("panel", msg.text, "panel", { rawPrompt: true });
      }
    }

    else if (msg.type === "learn") {
      const count = kb.addText(msg.text, msg.source || "chat");
      broadcast({
        type: "learned",
        chunksAdded: count,
        stats: kb.getStats(),
      });
    }

    else if (msg.type === "clear_history") {
      ai.clearHistory();
      broadcast({ type: "history_cleared" });
    }

    else if (msg.type === "tts_stop") {
      tts.stop();
    }

    else if (msg.type === "vision_context_frame") {
      // Panel envía frame silencioso → actualizar contexto visual de NYRA
      (async () => {
        const oldCtx = ai._screenContext;
        const desc = await ai.updateScreenContext(msg.image);
        if (!desc) return;
        // ¿Cambio significativo? → NYRA comenta espontáneamente
        const now = Date.now();
        const cooldown = 90000; // 90s mínimo entre comentarios proactivos
        if (oldCtx && ai._isSignificantChange(oldCtx, desc)
          && (now - ai._lastProactiveTs) > cooldown
          && !streamChat._isResponding) {
          ai._lastProactiveTs = now;
          console.log(`👁💬 Cambio detectado → comentario proactivo`);
          const proactivePrompt = `[Mirás la pantalla y notás un cambio. Antes veías: "${oldCtx.slice(0, 80)}". Ahora ves: "${desc.slice(0, 80)}". Hacé UN comentario natural como VTuber streamando — reaccioná como persona sin describir técnicamente. Máximo 2 oraciones.]`;
          console.log(`👁💬 Cambio detectado → comentario proactivo`);
          streamChat._reply("NYRA", proactivePrompt, "vision", { rawPrompt: true });
        }
      })();
    }

    else if (msg.type === "screen_sharing_stopped") {
      ai.clearScreenContext();
      console.log("👁 Pantalla descompartida — contexto visual limpiado");
    }

    else if (msg.type === "tts_done") {
      // OBS avisa que el audio terminó de reproducirse → liberar lock de TTS y cola de stream
      tts.audioEnded();
      streamChat.onAudioDone();
    }

    else if (msg.type === "streamer_speech") {
      // El streamer habló en voz alta → NYRA reacciona solo si la menciona por nombre
      const speechText = msg.text?.trim();
      if (!speechText || speechText.length < 4) return;

      // Ignorar si no menciona a NYRA (directo o por mishearing del reconocimiento de voz)
      // "nyra" en español suena a: mira, nira, ñira, naira, lyra, neira, nila, nira
      const wakeWordRegex = /\b(nyra|nira|ñira|naira|lyra|neira|nila|mira|nara)\b/i;
      const mentionsNyra = wakeWordRegex.test(speechText);
      if (!mentionsNyra) return;

      // Quitar el wake word del inicio para que quede solo la pregunta/comentario
      const cleanedSpeech = speechText.replace(/^\s*(nyra|nira|ñira|naira|lyra|neira|nila|mira|nara)[\s,!?]+/i, "").trim() || speechText;

      broadcast({ type: "thinking", thinking: true });
      let fullText = "";
      try {
        const prompt = `[El streamer te habla directamente]: "${cleanedSpeech}"`;
        const stream = ai.chatStream(prompt, "streamer");
        broadcast({ type: "stream_start" });
        for await (const chunk of stream) {
          fullText += chunk;
          broadcast({ type: "stream_chunk", chunk });
        }
        const filtered = filters.filterOutput(ai._stripContextLeaks(fullText));
        broadcast({ type: "stream_end", fullText: filtered, thinking: false, usedRAG: ai.lastUsedRAG ?? false });
        tts.speak(filtered);
        ai.learnFromExchange("streamer", speechText, fullText);
      } catch (err) {
        broadcast({ type: "error", text: `Error: ${err.message}`, thinking: false });
      }
    }

    else if (msg.type === "ping") {
      ws.send(JSON.stringify({ type: "pong" }));
    }
  });

  ws.on("close", () => console.log("🔌 Cliente desconectado"));
});

// ─── Start ────────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════╗
║     NYRA VTuber - IA Local Running       ║
╠══════════════════════════════════════════╣
║  🌐 Chat:     http://localhost:${PORT}       ║
║  🎭 OBS:      http://localhost:${PORT}/obs   ║
║  🔧 Panel:    http://localhost:${PORT}/panel ║
║  📡 WS:       ws://localhost:${PORT}         ║
╚══════════════════════════════════════════╝

  📋 Asegurate de tener Ollama corriendo:
     ollama serve
     ollama pull ${CONFIG.model}
  `);

  // Pre-calentar el modelo para que la primera respuesta no espere la carga
  setTimeout(async () => {
    try {
      console.log("🔥 Pre-calentando modelo en VRAM...");
      await fetch(`${CONFIG.ollamaUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: CONFIG.model,
          messages: [{ role: "user", content: "hola" }],
          stream: false,
          keep_alive: "30m",
          options: { num_predict: 1, num_ctx: 512 },
        }),
      });
      console.log("✅ Modelo listo en memoria");
    } catch (e) {
      console.warn("⚠️  No se pudo pre-calentar Ollama:", e.message);
    }
  }, 1500);
});