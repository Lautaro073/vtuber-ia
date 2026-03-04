# 🎭 NYRA — VTuber con IA Local Regenerativa

**IA 100% local, sin APIs externas, lista para OBS.**

---

## ⚡ Setup Rápido

### 1. Instalar Ollama (el motor de IA local)

**Windows / Mac:**  
Bajar desde https://ollama.com

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Descargar un modelo de lenguaje

```bash
# Recomendado (buena calidad, ~4.7GB):
ollama pull llama3

# Más liviano (~2.3GB, más rápido):
ollama pull phi3

# En español nativo (~4.1GB):
ollama pull qwen2
```

### 3. Instalar y correr NYRA

```bash
npm install
npm start
```

### 4. Abrir en el navegador

- **Panel de control:** http://localhost:3000
- **Vista para OBS:** http://localhost:3000/obs

---

## 🎬 Configurar OBS

1. En OBS: **Fuentes → + → Navegador**
2. URL: `http://localhost:3000/obs`
3. Ancho: `400` / Alto: `480`
4. **Sin chroma key** — el fondo ya es transparente
5. Activar: **"Actualizar el navegador cuando la escena se active"**

---

## 🧠 Sistema de Aprendizaje (RAG)

NYRA usa **Retrieval-Augmented Generation**:

1. Pegás cualquier texto en "Enseñar"
2. El texto se divide en chunks y se vectoriza con TF-IDF
3. Cuando alguien habla con NYRA, el sistema busca chunks relevantes
4. Esos chunks se inyectan como contexto en el prompt
5. NYRA responde con ese conocimiento de forma natural

El conocimiento se guarda en `knowledge.json` y **persiste entre reinicios**.

---

## 📁 Estructura

```
nyra-vtuber/
├── server.js          # Servidor Node.js + IA + RAG + TTS
├── knowledge.json     # Base de conocimiento (auto-generado)
├── public/
│   ├── obs.html       # ← Esto va en OBS (transparente)
│   ├── panel.html     # Panel de control completo
│   └── index.html     # Redirect al panel
└── package.json
```

---

## 🔧 Personalizar NYRA

En `server.js`, podés cambiar:

```js
// Personalidad
this.persona = `Tu prompt aquí...`;

// Modelo
model: "llama3"  // Cambiar por cualquier modelo de Ollama

// Tamaño de chunks para aprendizaje
chunkSize: 200

// Chunks de contexto por respuesta
maxRagChunks: 5
```

---

## 🗣️ Voz (TTS)

Usa el paquete `say` que utiliza las **voces del sistema operativo**:

- **Windows:** Voces de Microsoft (Panel de Control → Accesibilidad)
- **Mac:** Voces de macOS (`say` nativo)  
- **Linux:** Requiere `espeak`: `sudo apt install espeak`

Para agregar más voces en Windows: Configuración → Hora e idioma → Voz → Agregar voces.

---

## 🚀 Modelos recomendados

| Modelo | Tamaño | Velocidad | Calidad | Español |
|--------|--------|-----------|---------|---------|
| phi3   | 2.3GB  | ⚡⚡⚡    | ★★★     | ★★      |
| llama3 | 4.7GB  | ⚡⚡      | ★★★★    | ★★★     |
| qwen2  | 4.4GB  | ⚡⚡      | ★★★★    | ★★★★★   |
| mistral| 4.1GB  | ⚡⚡      | ★★★★    | ★★★     |
# vtuber-ia
