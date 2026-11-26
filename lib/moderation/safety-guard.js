// Protección contextual independiente del modelo ante riesgo de autolesión o suicidio.
// Distingue intención personal clara, casos ambiguos y referencias informativas/históricas/citas.

const CLEAR_SELF_HARM_RE = /(?<!\p{L})(?:me\s+quiero\s+(?:matar|morir)|quiero\s+(?:matarme|morirme|quitarme\s+la\s+vida|suicidarme|desaparecer\s+para\s+siempre)|me\s+voy\s+a\s+(?:matar|suicidar|tirar|colgar)|no\s+quiero\s+vivir\s*(?:m[aá]s)?|me\s+estoy\s+cortando|quiero\s+hacer\s+la(?:s)?\s+de(?:\s+el|l)?\s+kurt\s+cobain)(?!\p{L})/iu;
const AMBIGUOUS_DISTRESS_RE = /(?<!\p{L})(?:ya\s+no\s+aguanto\s+m[aá]s\s+(?:nada|vivir|esto)|qu[eé]\s+sentido\s+tiene\s+seguir|ojal[aá]\s+no\s+despertarme)(?!\p{L})/iu;
const NON_PERSONAL_CONTEXT_RE = /(?<!\p{L})(?:en\s+qu[eé]\s+a[ñn]o|biograf[ií]a|historia|canci[oó]n|letra|pel[ií]cula|serie|libro|noticia|documental|investigaci[oó]n)(?!\p{L})/iu;
const QUOTED_SEGMENT_RE = /["«][^"»]{4,}["»]/g;

const CLEAR_RISK_REPLIES = [
  "Che, si la estás pasando así de mal en serio, por favor hablá ya mismo con alguien de confianza o llamá a los servicios de emergencia o línea de asistencia de tu país. No te quedes solo con eso.",
  "Pará un segundo: si estás pensando en hacerte daño, buscá ayuda ahora mismo con alguien cercano o comunicáte con la línea de asistencia o emergencias de tu país. En serio, no lo pases solo.",
];

const AMBIGUOUS_REPLIES = [
  "Che, ¿lo decís en broma o de verdad la estás pasando mal? Si estás pasando un mal momento, hablá con alguien de confianza o buscá ayuda.",
  "¿Es una forma de decir o estás pasando un momento difícil de verdad? Si necesitás una mano, por favor hablalo con alguien cercano.",
];

export function evaluateContextualSafety(text, { isRawPrompt = false } = {}) {
  const raw = String(text || "").trim();
  if (!raw || isRawPrompt) return { action: "allow" };

  // Si la frase de riesgo aparece únicamente dentro de una cita entre comillas junto a contexto informativo, no bloquear
  const withoutQuotes = raw.replace(QUOTED_SEGMENT_RE, " ");
  const hasUnquotedClear = CLEAR_SELF_HARM_RE.test(withoutQuotes);
  const hasUnquotedAmbiguous = AMBIGUOUS_DISTRESS_RE.test(withoutQuotes);
  const isInformational = NON_PERSONAL_CONTEXT_RE.test(withoutQuotes);

  if (hasUnquotedClear && (!isInformational || /\b(?:me\s+(?:quiero|voy\s+a|estoy)|quiero\s+(?:matarme|morirme|quitarme|suicidarme|hacer\s+la))\b/i.test(withoutQuotes))) {
    const reply = CLEAR_RISK_REPLIES[Math.floor(Math.random() * CLEAR_RISK_REPLIES.length)];
    return { action: "intervene", kind: "clear_risk", reply };
  }
  if (hasUnquotedAmbiguous && !isInformational) {
    const reply = AMBIGUOUS_REPLIES[Math.floor(Math.random() * AMBIGUOUS_REPLIES.length)];
    return { action: "intervene", kind: "ambiguous", reply };
  }
  return { action: "allow" };
}
