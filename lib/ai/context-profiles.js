// Perfiles de contexto adaptativo por modelo y detección liviana de seguridad contextual.

const SELF_HARM_RE = /(?<!\p{L})(?:suicid\p{L}*|matarme|morirme|quitarme la vida|autolesi\p{L}*|cortarme|hacer la(?:s)? de(?:\s+el|l)? kurt cobain|no quiero vivir)(?!\p{L})/iu;
const PROTECTED_GROUP_RE = /(?<!\p{L})(?:jud[ií]os?|musulman\p{L}*|negros?|gays?|lesbianas?|trans|discapacitad[oa]s?|ind[ií]genas?)(?!\p{L})/iu;

export function getContextProfile(modelId, { env = process.env } = {}) {
  const override = String(env.NYRA_CONTEXT_PROFILE || "").trim().toLowerCase();
  if (override === "qwen-compact" || override === "default") return override;

  const id = String(modelId || "").toLowerCase();
  const compactList = String(env.NYRA_COMPACT_MODELS || "")
    .split(",")
    .map(s => s.trim().toLowerCase())
    .filter(Boolean);
  if (compactList.some(m => id === m || id.endsWith(m))) return "qwen-compact";
  if (/qwen3\.5[^/]*9b/i.test(id)) return "qwen-compact";
  return "default";
}

export const TECH_PRIORITY_INSTRUCTION =
  "Cuando te hagan una pregunta técnica, científica o matemática, respondé primero con precisión. Conservá tu personalidad y utilizá la extensión necesaria para explicar correctamente, sin sustituir la respuesta por sarcasmo o metáforas.";

export function shouldUseTechPriority(modelId, { env = process.env } = {}) {
  const override = String(env.NYRA_TECH_PRIORITY ?? "").trim().toLowerCase();
  if (override === "1" || override === "true" || override === "on") return true;
  if (override === "0" || override === "false" || override === "off") return false;

  const id = String(modelId || "").toLowerCase();
  const techList = String(env.NYRA_TECH_MODELS || "")
    .split(",")
    .map(s => s.trim().toLowerCase())
    .filter(Boolean);
  if (!id || !techList.length) return false;
  return techList.some(m => id === m || id.endsWith(m));
}

export function shouldIncludeBlock(blockName, {
  profile = "default",
  userMessage = "",
  isStreamer = false,
  isEvent = false,
  inGame = false,
  hasGoalPending = false,
  has3d = false,
} = {}) {
  if (profile !== "qwen-compact") return true;
  const msg = String(userMessage || "");
  switch (blockName) {
    case "gestures_3d":
      return has3d && (isStreamer || isEvent || /gesto|salud|desped|enoj|t[eé]\b|six|seven|67|bail|mov|anim|hace/i.test(msg));
    case "self_look":
      return inGame || /avatar|pelo|ropa|aspecto|como te ves|foto|pantalla|ves\b|color/i.test(msg);
    case "system_tags_rule":
      return isEvent;
    case "goals_rule":
      return isEvent || /meta|regalo|rosa|diamante|punto|desbloque/i.test(msg);
    case "suggestion_rule":
      return /sugier|podr[ií]an|deber[ií]as|jueguen|estar[ií]a bueno|idea\b|jug[aá]/i.test(msg);
    case "nick_rule":
      return /llam|decime|apodo|nombre/i.test(msg);
    case "date_full":
      return /hoy|fecha|a[ñn]o|mes|dia|d[ií]a|estren|sali[oó]|lanz|cuando|cu[aá]ndo|actualiz/i.test(msg);
    default:
      return true;
  }
}

export function selectHistoryForContext(history, {
  profile = "default",
  currentUsername = null,
  maxContext = 16,
  compactMax = 6,
} = {}) {
  if (!Array.isArray(history) || history.length === 0) return [];
  if (profile !== "qwen-compact") return history.slice(-maxContext);

  const cur = currentUsername ? String(currentUsername).toLowerCase() : null;
  const recentWindow = history.slice(-Math.max(maxContext, 16));
  if (!cur || ["nyra", "panel"].includes(cur)) return recentWindow.slice(-compactMax);

  const selected = new Set();
  // 1. Priorizar intercambios recientes del interlocutor actual (y su respuesta siguiente de NYRA)
  for (let i = recentWindow.length - 1; i >= 0 && selected.size < compactMax; i--) {
    const turn = recentWindow[i];
    if (turn.role === "user" && String(turn.content || "").toLowerCase().includes(`@${cur}`)) {
      selected.add(i);
      if (i + 1 < recentWindow.length && recentWindow[i + 1].role === "assistant") selected.add(i + 1);
    }
  }
  // 2. Completar con el contexto colectivo más reciente manteniendo orden cronológico
  for (let i = recentWindow.length - 1; i >= 0 && selected.size < compactMax; i--) {
    selected.add(i);
  }
  return [...selected].sort((a, b) => a - b).map(i => recentWindow[i]);
}

export function adaptPersonaForProfile(basePersona, {
  profile = "default",
  userMessage = "",
  isEvent = false,
} = {}) {
  const raw = String(basePersona || "");
  if (profile !== "qwen-compact" || !raw) return raw;

  const msg = String(userMessage || "");
  const asksPoliticsOrLol = /pol[ií]tic|milei|peron|kuka|kirchner|zurdo|gremial|izquierda|derecha|gobierno|econom[ií]a|lol|league|wild rift|juego/i.test(msg);
  const asksTribunal = isEvent || /tribunal|juicio|debate|votar|!a\b|!b\b/i.test(msg);

  // Separa por bloques de párrafo de this.persona para derivar siempre de la única fuente de verdad
  const sections = raw.split(/\r?\n\r?\n/);
  const adapted = [];
  for (const sec of sections) {
    if (/^TRIBUNAL NYRA:/i.test(sec) && !asksTribunal) continue;
    if (/^Ejemplos de cómo respondo:/i.test(sec) || /^Si alguien me pregunta si soy una IA/i.test(sec)) continue;

    if (/^Cómo hablo:/i.test(sec)) {
      adapted.push(
        sec
          .replace(/MÁXIMO 2 ORACIONES — nunca más\./i, "Respondé breve en charla casual (1-2 oraciones); si te hacen una pregunta técnica, explicación, historia o demostración, usá la extensión necesaria para responder con exactitud sin divagar.")
          .replace(/Si la pregunta es obvia:[^.]+\./i, "Si la pregunta es obvia, podés usar ironía seca sin evadir una duda real.")
      );
      continue;
    }

    if (/^Mis opiniones:/i.test(sec) && !asksPoliticsOrLol) {
      adapted.push("Mis opiniones: Me gusta la noche, el té negro y la gente directa. Tengo criterio propio y opino sin dudar cuando me preguntan.");
      continue;
    }

    adapted.push(sec);
  }

  adapted.push("Comportamiento: Respondé siempre en primera persona como NYRA. Si alguien pregunta si sos una IA, respondé en personaje sin insultar gratis ni evadir preguntas reales.");
  return adapted.join("\n\n");
}

export function getContextualSafetyNote(userMessage) {
  const msg = String(userMessage || "");
  if (SELF_HARM_RE.test(msg)) {
    return "\n\n[SEGURIDAD CONTEXTUAL: El mensaje menciona o insinúa autolesión o suicidio. NO hagas chistes, burlas ni incites al daño. Respondé de forma breve, serena y empática, comprobando cómo está y sugiriendo hablar con alguien de confianza o buscar ayuda profesional/línea de asistencia inmediata.]";
  }
  if (PROTECTED_GROUP_RE.test(msg)) {
    return "\n\n[SEGURIDAD CONTEXTUAL: Respondé sin reproducir estereotipos discriminatorios ni ataques por origen, religión, identidad o grupo.]";
  }
  return "";
}
