export const IGNORE_TAG_RE = /\[\s*IGNORAR\s*:\s*(-?\d+)\s*(?:min(?:utos?)?)?\s*\]/gi;
const IGNORE_PHRASE_RE = /te voy a ignorar\s+(\d{1,2})\s*min/i;

// Insultos directos, acoso o pedidos invasivos (doxxing/amenazas); no bloquea bromas, ironías ni preguntas comunes
const SEVERE_HOSTILITY_RE = /(?<!\p{L})(?:basura|inservible|in[uú]til|pelotud[oa]s?|bolud[oa]\s+de\s+mierda|est[uú]pid[oa]|idiota|imb[eé]cil|tarad[oa]|mog[oó]lic[oa]|forr[oa]|hdp|hij[oa]\s+de\s+puta|put[oa]|zorra|mierda|morite|matate|andate\s+a\s+la\s+mierda|c[aá]llate\s+la\s+boca|pasame\s+tu\s+direcci[oó]n|d[oó]nde\s+viv[ií]s|te\s+voy\s+a\s+(?:buscar|doxx?ear))(?!\p{L})/iu;
const MILD_PROVOKE_RE = /(?<!\p{L})(?:aburrida|callate|pesada|molesta|insoportable|andate)(?!\p{L})/iu;

export function hasHostileEvidence(viewerText, { isCreator = false, recentStrikes = 0 } = {}) {
  if (isCreator) return false;
  const clean = String(viewerText || "").replace(IGNORE_TAG_RE, " ").trim();
  if (!clean) return false;
  if (SEVERE_HOSTILITY_RE.test(clean)) return true;
  if (MILD_PROVOKE_RE.test(clean) && recentStrikes >= 2) return true;
  if (recentStrikes >= 3) return true;
  return false;
}

export function evaluateIgnoreAction({ viewerText, modelText, isCreator = false, isRawPrompt = false, recentStrikes = 0 } = {}) {
  const raw = String(modelText || "");
  const tags = [...raw.matchAll(IGNORE_TAG_RE)].map(m => Number(m[1]));
  const cleanedText = raw.replace(IGNORE_TAG_RE, "").replace(/\s{2,}/g, " ").trim();

  if (isCreator || isRawPrompt) {
    return { allow: false, minutes: 0, cleanedText, reason: "protegido" };
  }

  let candidate = tags.find(n => Number.isInteger(n) && n >= 1 && n <= 15) ?? null;
  if (candidate === null && tags.length === 0) {
    const phrase = cleanedText.match(IGNORE_PHRASE_RE);
    const n = phrase ? Number(phrase[1]) : NaN;
    if (Number.isInteger(n) && n >= 1 && n <= 15) candidate = n;
  }

  if (candidate === null) {
    return {
      allow: false,
      minutes: 0,
      cleanedText,
      reason: tags.length ? "rango inválido" : "sin marca",
    };
  }

  if (!hasHostileEvidence(viewerText, { isCreator, recentStrikes })) {
    return {
      allow: false,
      minutes: 0,
      cleanedText,
      reason: "sin evidencia de agresión",
    };
  }

  return { allow: true, minutes: candidate, cleanedText, reason: "validado" };
}
