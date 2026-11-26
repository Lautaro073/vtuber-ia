// Utilidades puras para limpieza de nombres/prefijos, RAG relevante y protección del aprendizaje.

const STOPWORDS = new Set([
  "hola", "holaa", "holaaa", "buenas", "buenos", "dias", "tardes", "noches", "chau", "adios",
  "que", "como", "cuando", "donde", "quien", "cual", "cuales", "para", "por", "con", "sin",
  "una", "uno", "unos", "unas", "los", "las", "del", "eso", "esa", "ese", "esto", "esta", "este",
  "pero", "porque", "aunque", "sino", "entonces", "tambien", "ademas", "igual", "solo", "sola",
  "jaja", "jajaja", "jajajaja", "jeje", "jsjs", "lmao", "omg", "wtf", "bro", "che", "vos", "sos",
  "eres", "soy", "son", "esta", "estas", "estan", "estoy", "estamos", "hay", "ser", "era", "fue", "van", "voy", "ver", "dar", "decir", "dijo", "dice",
  "hace", "hacer", "hizo", "tiene", "tener", "tengo", "puede", "puedo", "puedes", "quiere", "quiero",
  "muy", "mas", "menos", "bien", "mal", "todo", "toda", "todos", "todas", "nada", "algo", "alguien",
  "aqui", "aca", "ahi", "alla", "hoy", "ayer", "ahora", "despues", "antes", "siempre", "nunca",
  "funciona", "sirve", "pasa", "paso", "cosa", "cosas", "tema", "algo", "nyra", "nira", "mira",
]);

export function normalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

export function tokenizeRelevant(text) {
  return normalizeText(text)
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter(w => w.length > 2 && !STOPWORDS.has(w) && !/^\d+$/.test(w));
}

export function dropLeadingName(text, name) {
  let out = String(text || "").trim();
  if (!name) return out;
  const escaped = String(name).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const n = `(?<![\\p{L}@])@?${escaped}(?![\\p{L}\\d])`;
  const leadRe = new RegExp(`^\\s*${n}[\\s,:;!…-]+`, "iu");
  while (leadRe.test(out)) out = out.replace(leadRe, "");
  return out
    .replace(new RegExp(`,\\s*${n}\\s*,`, "giu"), ",")
    .replace(new RegExp(`,\\s*${n}(?![\\p{L}\\d])`, "giu"), "")
    .replace(new RegExp(`${n}\\s*,\\s*`, "giu"), "")
    .replace(/^[\s,;:-]+/, "")
    .replace(/,\s*,+/g, ",")
    .trim()
    .replace(/^\p{Ll}/u, c => c.toUpperCase());
}

export function searchRankedChunks(chunks, query, {
  topK = 3,
  minScore = 0.25,
  currentUsername = null,
  filterFn = null,
} = {}) {
  if (!Array.isArray(chunks) || chunks.length === 0) return [];
  const qTokens = tokenizeRelevant(query);
  if (qTokens.length === 0) return [];

  const N = chunks.length;
  const normalizedChunks = chunks.map(c => ({
    chunk: c,
    tokens: tokenizeRelevant(c.text),
  }));
  const dfs = new Map(qTokens.map(qt => [
    qt,
    normalizedChunks.filter(x => x.tokens.includes(qt)).length,
  ]));

  const currentUserLower = currentUsername ? String(currentUsername).toLowerCase() : null;
  const scored = [];
  for (const { chunk, tokens } of normalizedChunks) {
    if (filterFn && !filterFn(chunk)) continue;
    if (tokens.length === 0) continue;
    const tf = {};
    for (const t of tokens) tf[t] = (tf[t] || 0) + 1;

    let score = 0;
    let matchedTerms = 0;
    for (const qt of qTokens) {
      if (tf[qt]) {
        matchedTerms++;
        const df = dfs.get(qt) || 0;
        score += (tf[qt] / tokens.length) * (Math.log(N / (df + 1)) + 1);
      }
    }
    if (score < minScore || (qTokens.length >= 3 && matchedTerms < 1)) continue;

    // Prioriza recuerdos del interlocutor actual; penaliza otros viewers si no es un match fuerte
    if (currentUserLower && chunk.username) {
      const u = String(chunk.username).toLowerCase();
      if (u === currentUserLower) score *= 1.6;
      else if (u !== "panel" && u !== "streamer") score *= 0.65;
    }
    if (score >= minScore) scored.push({ ...chunk, score });
  }

  scored.sort((a, b) => b.score - a.score);
  const seen = new Set();
  const out = [];
  for (const item of scored) {
    const sig = normalizeText(item.text).slice(0, 60);
    if (seen.has(sig)) continue;
    seen.add(sig);
    out.push(item);
    if (out.length >= topK) break;
  }
  return out;
}

export function formatChatLearningContext(chunks, currentUsername = null) {
  const cur = currentUsername ? String(currentUsername).toLowerCase() : null;
  return chunks.map(c => {
    const [uPart, ...nParts] = String(c.text || "").split("\nNYRA:");
    const asked = uPart.replace(/^@[^:]+:\s*/, "").trim();
    const replied = nParts.join(" ").replace(/^@[\w.-]+,\s*/i, "").trim();
    const isSame = cur && String(c.username || "").toLowerCase() === cur;
    if (!replied) return `- ${uPart.trim()}`;
    return isSame
      ? `- Antes, sobre "${asked}", le respondiste a este mismo viewer: "${replied}"`
      : `- Contexto de charla previa en el stream (con @${c.username} sobre "${asked}"): comentaste "${replied}"`;
  });
}

export function trimAtBoundary(text, maxLen = 220) {
  const s = String(text || "").replace(/\s+/g, " ").trim();
  if (s.length <= maxLen) return s;
  const slice = s.slice(0, maxLen);
  const lastStop = Math.max(slice.lastIndexOf(". "), slice.lastIndexOf("! "), slice.lastIndexOf("? "));
  if (lastStop >= Math.floor(maxLen * 0.55)) return slice.slice(0, lastStop + 1).trim();
  const lastSpace = slice.lastIndexOf(" ");
  return (lastSpace > 40 ? slice.slice(0, lastSpace) : slice).trim();
}

const CONTAMINATED_RE = /<sys>|<\/sys>|\[(?:INVITACI[OÓ]N|META\s+CUMPLIDA|ALERTA|LIVE\s+TRANQUILO|TRIBUNAL|PERSONAJE|CONTEXTO|RESPOND[EÉ]\s+SOLO)|@\w+\s*\(chat\)\s*:|NYRA\s*\((?:al directo|al grupo)\)\s*:|,\s*,|como (?:una )?ia\b|modelo de lenguaje|blocked by gemini/i;

export function isCleanForLearning(userMsg, nyraMsg) {
  const u = String(userMsg || "").trim();
  const n = String(nyraMsg || "").trim();
  if (!u || !n || n === "..." || !/\p{L}/u.test(n)) return false;
  if (CONTAMINATED_RE.test(n)) return false;
  return true;
}
