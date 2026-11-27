/**
 * Pronunciación: palabras en inglés escritas como tienen que sonar en español. Se aplica a todas las voces.
 * Estas son las de fábrica; las que se editan desde el panel quedan en nyra_pronunciation.json (lista completa).
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";

export const DEFAULT_PRONUNCIATIONS = [
  // Frases compuestas (siempre antes que palabras sueltas)
  ["league of legends", "Lig of Leyends", "wildrift"],
  ["wild rift", "Guail Rift", "wildrift"],
  ["six seven", "siks séven", "stream"],
  ["twisted fate", "Tuisted Feit", "wildrift"],
  ["miss fortune", "Mis Fórchun", "wildrift"],
  ["master yi", "Máster Yi", "wildrift"],
  ["aurelion sol", "Aurélion Sol", "wildrift"],
  ["lee sin", "Li Sin", "wildrift"],
  ["xin zhao", "Shin Zhao", "wildrift"],
  ["tahm kench", "Tam Kench", "wildrift"],
  ["jarvan iv", "Yárvan cuatro", "wildrift"],

  // Streaming, chat y memes
  ["streamear", "estrimear", "stream"],
  ["streamer", "estrímer", "stream"],
  ["stream", "estrím", "stream"],
  ["vtuber", "Vitúber", "stream"],
  ["discord", "Díscord", "stream"],
  ["viewers", "viúers", "stream"],
  ["viewer", "viúer", "stream"],
  ["cringe", "crinch", "stream"],
  ["hype", "jaip", "stream"],
  ["chill", "chil", "stream"],
  ["random", "rándom", "stream"],
  ["okay", "okéi", "stream"],
  ["ok", "okéi", "stream"],
  ["wtf", "uatafak", "stream"],
  ["afk", "a efe ka", "stream"],
  ["fps", "efe pe ese", "stream"],
  ["obs", "O Be Ese", "stream"],
  ["gg", "yi yi", "stream"],
  ["ia", "i a", "stream"],

  // Gaming / Wild Rift
  ["rankeds", "ránkeds", "wildrift"],
  ["ranked", "ránked", "wildrift"],
  ["carrear", "karrear", "wildrift"],
  ["carry", "kerri", "wildrift"],
  ["support", "sopórt", "wildrift"],
  ["feedear", "fidear", "wildrift"],
  ["feed", "fid", "wildrift"],
  ["skins", "eskins", "wildrift"],
  ["skin", "eskin", "wildrift"],
  ["build", "bild", "wildrift"],
  ["buff", "baf", "wildrift"],
  ["blue", "blu", "wildrift"],
  ["kda", "ka de a", "wildrift"],
  ["adc", "a de ce", "wildrift"],

  // Campeones con pronunciación difícil en español
  ["kha'zix", "Ka Ziks", "wildrift"],
  ["vel'koz", "Vel Koz", "wildrift"],
  ["kai'sa", "Kaisa", "wildrift"],
  ["k'sante", "Kasante", "wildrift"],
  ["warwick", "Guórwik", "wildrift"],
  ["thresh", "Zresh", "wildrift"],
  ["twitch", "Tuich", "wildrift"],
  ["ezreal", "Ezrial", "wildrift"],
  ["jinx", "Yinks", "wildrift"],
  ["pyke", "Paik", "wildrift"],
  ["lux", "Luks", "wildrift"],
  ["jax", "Yaks", "wildrift"],
  ["jhin", "Yin", "wildrift"],
  ["kayn", "Kein", "wildrift"],
  ["vayne", "Vein", "wildrift"],
  ["jayce", "Yeis", "wildrift"],
];

export const PRONUNCIATION_GROUPS = ["wildrift", "stream", "general"];
const FILE = new URL("./nyra_pronunciation.json", import.meta.url);
const MAX_WORDS = 500;

let words = DEFAULT_PRONUNCIATIONS.map(([word, say, group]) => ({ word, say, group }));
try { if (existsSync(FILE)) words = JSON.parse(readFileSync(FILE, "utf8")).words; }
catch (err) { console.warn("🗣️ No pude leer nyra_pronunciation.json, uso las de fábrica:", err.message); }

const escapeRegex = str => str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
let rules = [];
function compile() {
  // Las más largas primero: "streamer" antes que "stream", "lee sin" antes que "lee"
  rules = [...words].sort((a, b) => b.word.length - a.word.length)
    .map(({ word, say }) => ({ regex: new RegExp(`(?<!\\p{L})${escapeRegex(word)}(?!\\p{L})`, "giu"), say }));
}
compile();

export const listPronunciations = () => words;

export function applyPhonetics(text) {
  let out = text;
  for (const { regex, say } of rules) out = out.replace(regex, say);
  return out;
}

const norm = (v, max) => String(v ?? "").replace(/\s+/g, " ").trim().slice(0, max);
function save() {
  compile();
  writeFileSync(FILE, JSON.stringify({ words }, null, 2));
}

// Agrega o reemplaza (sin importar mayúsculas). Devuelve la lista o tira error con un mensaje para el panel.
export function setPronunciation({ word, say, group }) {
  word = norm(word, 60).toLowerCase(); say = norm(say, 80);
  if (!word || !say) throw new Error("Falta la palabra o cómo se dice");
  if (!PRONUNCIATION_GROUPS.includes(group)) group = "general";
  const i = words.findIndex(w => w.word === word);
  if (i < 0 && words.length >= MAX_WORDS) throw new Error(`Máximo ${MAX_WORDS} palabras`);
  if (i < 0) words.push({ word, say, group }); else words[i] = { word, say, group };
  save();
  return words;
}

export function deletePronunciation(word) {
  word = norm(word, 60).toLowerCase();
  words = words.filter(w => w.word !== word);
  save();
  return words;
}
