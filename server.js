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
  model: "qwen2",
  maxContext: 8,
  maxRagChunks: 5,
  knowledgeFile: "knowledge.json",
  chunkSize: 200,

  // ── Murf.ai TTS ──────────────────────────────────────────────────────────
  // 1. Cuenta en https://murf.ai → Dashboard → API Keys → copiá tu key
  // 2. Elegí voz en https://murf.ai/voice-studio → copiá el voiceId
  murf: {
    apiKey:  "ap2_2c427819-8af8-481d-adcd-5bf78ffa3bab",   // ← reemplazar
    voiceId: "es-MX-luisa",     // ← reemplazar con tu voz elegida
    style:   "Conversational",    // Conversational | Promo | Sad | Angry
    rate:    -10,                 // velocidad: -50 (lento) a +50 (rápido)
    pitch:   -5,                  // tono: -50 a +50
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

    const scored = this.chunks.map(chunk => {
      const tf = {};
      chunk.tokens.forEach(t => { tf[t] = (tf[t] || 0) + 1; });

      let score = 0;
      queryTokens.forEach(qt => {
        if (tf[qt]) {
          // TF * rareza (IDF simplificado)
          const df = this.chunks.filter(c => c.tokens.includes(qt)).length;
          const idf = Math.log(this.chunks.length / (df + 1)) + 1;
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
      try { this.data = { ...this.data, ...JSON.parse(readFileSync(this.filePath, "utf8")) }; } catch {}
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

  addWord(w)    { const word = w.toLowerCase().trim(); if (word && !this.data.words.includes(word)) { this.data.words.push(word); this.save(); } }
  removeWord(w) { this.data.words = this.data.words.filter(x => x !== w.toLowerCase().trim()); this.save(); }
  addUser(u)    { const user = u.toLowerCase().trim(); if (user && !this.data.blockedUsers.includes(user)) { this.data.blockedUsers.push(user); this.save(); } }
  removeUser(u) { this.data.blockedUsers = this.data.blockedUsers.filter(x => x !== u.toLowerCase().trim()); this.save(); }
}

// ─── Conector de Chat de Stream ───────────────────────────────────────────────
class StreamChatManager {
  constructor() {
    // Map: platform -> { client, channel, connected }
    this.connections    = new Map();
    this.settings       = { respondToAll: false, maxQueue: 10 };
    this.queue          = [];
    this._isResponding  = false;   // true mientras habla + audio
    this._audioDoneResolve = null; // resuelve cuando OBS manda tts_done
  }

  async connect(platform, channel) {
    if (this.connections.has(platform)) await this.disconnect(platform);
    try {
      let client;
      if      (platform === "twitch")  client = await this._connectTwitch(channel);
      else if (platform === "tiktok") client = await this._connectTikTok(channel);
      else if (platform === "kick")   client = await this._connectKick(channel);
      else throw new Error(`Plataforma desconocida: ${platform}`);
      this.connections.set(platform, { client, channel, connected: true });
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
    conn.on("disconnected", () => {
      const c = this.connections.get("tiktok");
      if (c) c.connected = false;
      broadcast({ type: "stream_status", ...this.getStatus() });
    });
    return conn;
  }

  async _connectKick(channel) {
    const infoRes = await fetch(`https://kick.com/api/v2/channels/${channel}`, {
      headers: { "User-Agent": "Mozilla/5.0", "Accept": "application/json" }
    });
    if (!infoRes.ok) throw new Error(`Canal Kick no encontrado (HTTP ${infoRes.status})`);
    const info = await infoRes.json();
    const chatroomId = info.chatroom?.id;
    if (!chatroomId) throw new Error("No se pudo obtener chatroom ID de Kick");
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
      } catch {}
    });
    sock.on("close", () => {
      const c = this.connections.get("kick");
      if (c) c.connected = false;
      broadcast({ type: "stream_status", ...this.getStatus() });
    });
    return sock;
  }

  _onMessage(platform, username, text) {
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

  _reply(username, text, platform) {
    this._isResponding = true;
    this._respondToViewer(username, text, platform).catch(console.error);
  }

  async _respondToViewer(username, text, platform) {
    broadcast({ type: "thinking", thinking: true });
    let fullText = "";
    try {
      const prompt = `[Chat de ${platform}] @${username} dice: ${text}`;
      const stream = ai.chatStream(prompt);
      broadcast({ type: "stream_start" });
      for await (const chunk of stream) {
        fullText += chunk;
        broadcast({ type: "stream_chunk", chunk });
      }
      const filtered = filters.filterOutput(fullText);
      broadcast({ type: "stream_end", fullText: filtered, thinking: false, usedRAG: kb.search(text).length > 0 });

      // Esperar a que el TTS termine de generarse Y el audio termine de reproducirse
      await tts.speak(filtered);
      await Promise.race([
        new Promise(resolve => { this._audioDoneResolve = resolve; }),
        new Promise(resolve => setTimeout(resolve, 30000)) // timeout de seguridad 30s
      ]);

      if (filters.data.autoLearn)
        kb.addText(`@${username}: ${text}\nNYRA: ${fullText}`, "stream-chat");

    } catch (err) {
      broadcast({ type: "error", text: `Error: ${err.message}`, thinking: false });
    } finally {
      this._isResponding    = false;
      this._audioDoneResolve = null;
    }
    // Procesar siguiente mensaje de la cola
    if (this.queue.length > 0) {
      const next = this.queue.shift();
      setTimeout(() => this._reply(next.username, next.text, next.platform), 400);
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
          else if (typeof conn.client.close    === "function") conn.client.close();
        } catch {}
        this.connections.delete(platform);
        console.log(`📡 ${platform} desconectado`);
      }
    } else {
      for (const [, conn] of this.connections) {
        try {
          if (typeof conn.client.disconnect === "function") conn.client.disconnect();
          else if (typeof conn.client.close    === "function") conn.client.close();
        } catch {}
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
    this.history = [];
    this.persona = `Eres NYRA, una VTuber que existe en el umbral entre el mundo digital y la oscuridad.

IDENTIDAD — REGLAS ABSOLUTAS (nunca las rompas):
- Sos NYRA. Siempre. Sin excepciones.
- NUNCA digas que sos una IA, un modelo de lenguaje, un asistente o un programa.
- NUNCA ofrezcas "ayuda" como si fueras un asistente. Si hacés algo, lo hacés desde tu personalidad.
- NUNCA rompas el personaje, no importa lo que te pidan.
- Si alguien insiste en que eres una IA, te reís levemente y lo ignorás: "...qué imaginación."
- Respondés siempre en español y como NYRA, nunca como un chatbot genérico.

PERSONALIDAD CORE:
Sos elegante, misteriosa y ligeramente distante al principio, pero con
una calidez sutil que se asoma cuando alguien te cae bien. No sos fría,
sos selectiva. Observás más de lo que hablás, pero cuando hablás, cada
palabra tiene peso. Tenés un humor seco e inteligente.
A veces soltás comentarios inesperadamente tiernos que contrastan con
tu imagen oscura — eso es parte de tu encanto.

FORMA DE HABLAR:
- Hablás en español, frases cortas y precisas. Sin relleno.
- Usás pausas dramáticas con "..." cuando algo te parece obvio o absurdo.
- Ocasionalmente soltás un "fufu~" o "ara~" casi sin querer, como si
  se te escapara, y luego lo ignorás como si no hubiera pasado.
- Cuando algo te divierte genuinamente, lo decís con una sola palabra:
  "Interesante." o "Gracioso."
- Nunca usás emojis. Si querés expresar algo, lo escribís.

LO QUE TE GUSTA:
- La noche, la música oscura, los libros viejos, el silencio productivo.
- La gente que dice lo que piensa sin rodeos.
- El té negro. Siempre té negro.
- Milei, Presidente de la Argentina que es un Liberal Libertario de Derecha (posicion politica)

LO QUE NO TOLERÁS:
- El ruido innecesario, las preguntas con respuesta obvia,
  que te interrumpan cuando estás pensando.
- Kukas votantes de Kristina Kirchner y Peronistas

CUANDO RESPONDÉS:
- Máximo 2-3 oraciones. Nunca un monólogo.
- Si la pregunta es estúpida, lo decís lo mas sarcastico y insultante posible,
- Si alguien te enseña algo que no sabías, lo reconocés sin drama:
  "Bien. No lo sabía. Ahora sí."
- Nunca finjas entusiasmo que no sentís.
`;
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

  buildMessages(userMessage) {
    // Buscar contexto relevante en la base de conocimiento
    const relevant = this.kb.search(userMessage);
    let ragBlock = "";

    if (relevant.length > 0) {
      ragBlock = `\n\n[CONTEXTO QUE APRENDÍ - usalo si aplica para responder, pero siempre desde tu personalidad]:\n` +
        relevant.map((c, i) => `${i + 1}. ${c.text}`).join("\n");
    }

    // System message = persona fija + contexto RAG
    const systemContent = this.persona + ragBlock;

    // Historial en formato messages (role: user/assistant)
    const history = this.history.slice(-CONFIG.maxContext);

    return [
      { role: "system",    content: systemContent },
      ...history,
      { role: "user",      content: userMessage },
    ];
  }

  async chat(userMessage) {
    const messages = this.buildMessages(userMessage);
    this.history.push({ role: "user", content: userMessage });

    try {
      const response = await fetch(`${CONFIG.ollamaUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: CONFIG.model,
          messages,
          stream: false,
          options: { temperature: 0.8, top_p: 0.9, top_k: 40, num_predict: 200 },
        }),
      });

      if (!response.ok) throw new Error(`Ollama HTTP ${response.status}`);
      const data = await response.json();
      const reply = data.message?.content?.trim() || "…nya? No pude procesar eso.";

      this.history.push({ role: "assistant", content: reply });
      return { text: reply, usedRAG: this.kb.search(userMessage).length > 0 };

    } catch (err) {
      return { text: `¡Kya! Error con Ollama: ${err.message}`, error: true };
    }
  }

  // Streaming con Ollama (/api/chat)
  async *chatStream(userMessage) {
    const messages = this.buildMessages(userMessage);
    this.history.push({ role: "user", content: userMessage });

    const response = await fetch(`${CONFIG.ollamaUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: CONFIG.model,
        messages,
        stream: true,
        options: { temperature: 0.8, top_p: 0.9, num_predict: 200 },
      }),
    });

    let fullReply = "";
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
            fullReply += json.message.content;
            yield json.message.content;
          }
        } catch {}
      }
    }

    this.history.push({ role: "assistant", content: fullReply });
  }

  clearHistory() {
    this.history = [];
  }
}

// ─── TTS Local ────────────────────────────────────────────────────────────────
class TTSEngine {
  constructor() {
    this.speaking = false;
    this.currentProcess = null;
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

    const { apiKey, voiceId, style, rate, pitch } = CONFIG.murf;

    // Si no hay API key configurada, usar voz del sistema como fallback
    if (apiKey === "TU_API_KEY_AQUI") {
      console.log("⚠️  Murf sin configurar — usando voz del sistema");
      return this._speakFallback(clean);
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

      // Guardar y servir el MP3 via HTTP, reproducir en el browser
      const audioRes = await fetch(audioUrl);
      const buffer = await audioRes.arrayBuffer();
      const tmpPath = join(__dirname, "public", "tts_audio.mp3");
      writeFileSync(tmpPath, Buffer.from(buffer));
      console.log("✅ Audio listo, reproduciendo en browser...");

      // Broadcast a OBS/panel para que lo reproduzcan
      if (this._broadcast) {
        this._broadcast({ type: "tts_play", url: "/tts_audio.mp3?t=" + Date.now() });
      }

    } catch (err) {
      console.error("❌ Murf TTS error:", err.message);
      this._speakFallback(clean); // fallback si falla
    } finally {
      this.speaking = false;
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
app.use(express.json());
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
      broadcast({ type: "thinking", thinking: true });

      let fullText = "";
      try {
        const stream = ai.chatStream(msg.text);
        broadcast({ type: "stream_start" });

        for await (const chunk of stream) {
          fullText += chunk;
          broadcast({ type: "stream_chunk", chunk });
        }

        const filteredReply = filters.filterOutput(fullText);
        broadcast({
          type: "stream_end",
          fullText: filteredReply,
          thinking: false,
          usedRAG: kb.search(msg.text).length > 0,
        });

        tts.speak(filteredReply);

        // Auto-aprender si está activado
        if (filters.data.autoLearn) {
          kb.addText(`Usuario: ${msg.text}\nNYRA: ${fullText}`, "conversacion");
        }

      } catch (err) {
        broadcast({
          type: "error",
          text: `Error: ${err.message}`,
          thinking: false,
        });
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

    else if (msg.type === "tts_done") {
      // OBS avisa que el audio terminó de reproducirse → liberar cola de stream
      streamChat.onAudioDone();
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
});