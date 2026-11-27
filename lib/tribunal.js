// Tribunal NYRA: entender votos y pedidos aunque no escriban el comando exacto

const norm = s => String(s || "").toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^\p{L}\p{N} ]/gu, " ").replace(/\s+/g, " ").trim();

// "!a", "a!", "A", "voto a", "voto por la b", "lado a", "team b porque..." → { side, arg }
const LETTER_RE = /^(?:!+\s*|(?:mi\s+)?(?:voto|vot[oó]|lado|team|equipo|opcion|opción)\s+(?:(?:por|al?|el|la)\s+)*(?:lado\s+|opci[oó]n\s+)?)?([ab])(?![\p{L}\p{N}])[!.]*\s*(.*)$/iu;
// "voto rojo", "team pizza con ananá" → compara con el texto de cada lado
const NAME_RE = /^!?\s*(?:mi\s+)?(?:voto|vot[oó]|team|equipo)\s+(?:(?:por|al?|el|la|los|las)\s+)*(.+)$/iu;

export function parseVote(text, tribunal) {
  if (!tribunal) return null;
  const raw = String(text || "").trim();
  const letter = raw.match(LETTER_RE);
  // una letra sola cuenta solo si no hay más texto que parezca charla ("a ver", "a mí me...")
  if (letter && (/^[!]|^(?:mi\s+)?(?:voto|vot[oó]|lado|team|equipo|opci[oó]n)/iu.test(raw) || /^[ab][!.]*$/i.test(raw) || /^[ab]!/i.test(raw)))
    return { side: letter[1].toLowerCase(), arg: letter[2].trim() };
  const named = raw.match(NAME_RE);
  if (!named) return null;
  const said = norm(named[1]);
  const hits = ["a", "b"].filter(side => {
    const name = norm(tribunal[side]);
    return name && (said === name || said.startsWith(name + " ") || (name.length >= 4 && said.includes(name)));
  });
  if (hits.length !== 1) return null; // ambiguo o ninguno
  return { side: hits[0], arg: said.slice(norm(tribunal[hits[0]]).length).trim() };
}

// "!tribunal ..." sin las barras, "hagamos un tribunal", "armá una votación sobre..."
export const TRIBUNAL_ASK_RE = /^!\s*(?:tribunal|juicio|votacion|votación)\b|(?<!\p{L})(?:arm[aá]|arm[eé]n|hac[eé]|hagan|hagamos|abr[ií]|abran|cre[aá]|pon[eé]|quiero|queremos|podemos hacer|y si hacemos)\s+(?:un|una|otro|otra)\s+(?:tribunal|juicio|votaci[oó]n|encuesta)(?!\p{L})/iu;
