// Limpia turnos inventados, etiquetas internas del sistema/contexto y repeticiones literales/casi idénticas
// sin recortar respuestas largas legítimas, menciones a usuarios, texto natural entre corchetes
// ni repeticiones humorísticas cortas. Conserva [GESTO:], [ANIMACIÓN:], [SUGERENCIA:] e [IGNORAR:].

const INTERNAL_SECTION_RE = /(?:^|\r?\n)\s*\[(?:PERSONAJE|LO [UÚ]LTIMO DEL STREAM[^\]]*|RESPOND[EÉ] SOLO A ESTO|Cosas que s[eé][^\]]*|Conversaciones pasadas[^\]]*|Mis opiniones[^\]]*|Lo que veo en pantalla[^\]]*|Lo que viste antes[^\]]*|CONTEXTO[^\]]*|PERSONAS|ÓRDENES ANOTADAS|ORDENES ANOTADAS|LO QUE DIJO QUIEN LA CRE[OÓ]|CUADERNO|CHARLA NUEVA)\]/i;
const INTERNAL_TURN_RE = /(?:^|\r?\n)\s*(?:@[\w.-]+\s*\(chat\)\s*:|NYRA\s*\((?:al directo|al grupo)\)\s*:)/i;
const NEWLINE_SPEAKER_RE = /\r?\n+\s*(?:NYRA|@[\w.-]+)\s*:\s*(?=«|["“¿¡\p{L}])/u;

// Etiquetas internas conocidas del sistema/contexto (toleran mayúsculas, tildes y espacios extra)
const KNOWN_INTERNAL_TAG_RE = /\[\s*(?:SISTEMA|INVITACI[OÓ]N|META\s+CUMPLIDA|ALERTA(?:\s+[A-ZÁÉÍÓÚÑ\w]+)?|LIVE\s+TRANQUILO|TRIBUNAL\s+NYRA(?:\s*-\s*SENTENCIA)?|DATO\s+VERIFICADO|B[UÚ]SQUEDA|DATOS\s+DE\s+METAS(?:\s*\([^)]*\))?|GESTO\s+EN\s+PANTALLA|MENSAJE\s+DEL\s+STREAMER(?:\s+EN\s+EL\s+CHAT)?|PERSONAJE|RESPOND[EÉ]\s+SOLO\s+A\s+ESTO|LO\s+[UÚ]LTIMO\s+DEL\s+STREAM[^\]\n]*|CONTEXTO(?:\s*:[^\]\n]*)?|RECUERDO(?:\s*:[^\]\n]*)?|PERSONAS|[OÓ]RDENES\s+ANOTADAS|LO\s+QUE\s+DIJO\s+QUIEN\s+LA\s+CRE[OÓ]|CLASE|CLASES|GRUPO|CUADERNO|CHARLA\s+NUEVA|LO\s+[UÚ]LTIMO\s+QUE\s+DIJO\s+NYRA|LO\s+QUE\s+SE\s+HABL[OÓ]|FRASE\s+NUEVA|Cosas\s+que\s+s[eé][^\]\n]*|Conversaciones\s+pasadas[^\]\n]*|Mis\s+opiniones\s+al\s+respecto|Lo\s+que\s+veo\s+en\s+pantalla[^\]\n]*|Lo\s+que\s+viste\s+antes[^\]\n]*|Hora\s+real\s+del\s+stream[^\]\n]*|Fecha\s+de\s+hoy[^\]\n]*|Tema\s+del\s+directo[^\]\n]*)\s*\]/giu;

// Regla general: cualquier etiqueta en MAYÚSCULAS tipo sistema que no sea una de las 4 acciones válidas de NYRA
const VALID_ACTION_PREFIX = "(?:GESTO|ANIMACI[OÓ]N|ANIMACION|SUGERENCIA|IGNORAR)\\b";
const GENERIC_SYSTEM_TAG_RE = new RegExp(`\\[\\s*(?!${VALID_ACTION_PREFIX})[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9_\\s-]{3,45}(?::[^\\]\\n]*)?\\]`, "gu");

function normalizeForCompare(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/\[(?:GESTO|ANIMACI[OÓ]N|SUGERENCIA|IGNORAR)\s*:[^\]]*\]/gi, "")
    .replace(/[^\p{L}\p{N}\s]/gu, "")
    .replace(/\s+/g, " ")
    .trim();
}

function isNearDuplicate(a, b) {
  const na = normalizeForCompare(a);
  const nb = normalizeForCompare(b);
  if (na.length < 25 || nb.length < 25) return false;
  if (na === nb) return true;
  const shorter = na.length <= nb.length ? na : nb;
  const longer = na.length > nb.length ? na : nb;
  if (longer.includes(shorter) && shorter.length / longer.length >= 0.9) return true;
  return false;
}

function stripLeadingPrefix(text) {
  return String(text || "").replace(/^\s*NYRA\s*(?:\([^)]*\))?\s*:\s*/i, "");
}

function stripInternalTags(text) {
  return String(text || "")
    .replace(KNOWN_INTERNAL_TAG_RE, "")
    .replace(GENERIC_SYSTEM_TAG_RE, "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/^[ \t]+|[ \t]+$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function cutInventedTurns(text) {
  let out = stripLeadingPrefix(text);
  for (const re of [INTERNAL_SECTION_RE, INTERNAL_TURN_RE, NEWLINE_SPEAKER_RE]) {
    const m = out.match(re);
    if (m && m.index !== undefined) {
      if (m.index > 0) {
        out = out.slice(0, m.index);
      } else if (re === INTERNAL_TURN_RE) {
        const nyraReply = out.match(/NYRA\s*(?:\([^)]*\))?\s*:\s*([\s\S]+)/i);
        if (nyraReply) out = nyraReply[1];
      }
    }
  }
  return out.trim();
}

function deduplicateInlineRepeat(block) {
  const trimmed = block.trim();
  const norm = normalizeForCompare(trimmed);
  if (norm.length < 50) return trimmed;
  const sentences = trimmed.match(/[^.!?…]+[.!?…]*/g);
  if (!sentences || sentences.length < 2) return trimmed;
  for (let split = 1; split <= Math.floor(sentences.length / 2); split++) {
    const first = sentences.slice(0, split).join("").trim();
    const rest = sentences.slice(split, split * 2).join("").trim();
    if (isNearDuplicate(first, rest)) return first;
  }
  return trimmed;
}

export function sanitizeModelResponse(text) {
  const cut = stripInternalTags(cutInventedTurns(text));
  if (!cut) return "";

  const blocks = cut.split(/\r?\n{2,}/).map(b => b.trim()).filter(Boolean);
  const kept = [];
  for (const block of blocks) {
    if (kept.some(prev => isNearDuplicate(prev, block))) continue;
    kept.push(deduplicateInlineRepeat(block));
  }
  return kept.join("\n\n").trim();
}
