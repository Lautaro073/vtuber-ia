// Perfiles de contexto adaptativo por modelo y detección liviana de seguridad contextual.

const SELF_HARM_RE = /(?<!\p{L})(?:suicid\p{L}*|matarme|morirme|quitarme la vida|autolesi\p{L}*|cortarme|hacer la(?:s)? de(?:\s+el|l)? kurt cobain|no quiero vivir)(?!\p{L})/iu;
const PROTECTED_GROUP_RE = /(?<!\p{L})(?:jud[ií]os?|musulman\p{L}*|negros?|gays?|lesbianas?|trans|discapacitad[oa]s?|ind[ií]genas?)(?!\p{L})/iu;

export function getContextProfile(modelId) {
  const id = String(modelId || "").toLowerCase();
  if (id === "opencode:nyra-ollama/nyra" || id.endsWith("nyra-ollama/nyra")) {
    return "qwen-compact";
  }
  return "default";
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
      return hasGoalPending || /meta|regalo|rosa|diamante|punto|desbloque/i.test(msg);
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
