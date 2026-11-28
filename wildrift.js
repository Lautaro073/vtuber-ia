// Datos de Wild Rift para que NYRA lea bien la pantalla y relate como alguien que juega.
// shortcut: la lista es el roster completo de LoL + Norra (Riot no publica el de Wild Rift); sobran algunos que WR no
// tiene, pero solo se usa para corregir nombres. Sumar los campeones nuevos cuando salgan.
export const CHAMPIONS = `Aatrox, Ahri, Akali, Akshan, Alistar, Ambessa, Amumu, Anivia, Annie, Aphelios, Ashe, Aurelion Sol, Aurora, Azir,
Bard, Bel'Veth, Blitzcrank, Brand, Braum, Briar, Caitlyn, Camille, Cassiopeia, Cho'Gath, Corki, Darius, Diana, Dr. Mundo, Draven,
Ekko, Elise, Evelynn, Ezreal, Fiddlesticks, Fiora, Fizz, Galio, Gangplank, Garen, Gnar, Gragas, Graves, Gwen, Hecarim,
Heimerdinger, Hwei, Illaoi, Irelia, Ivern, Janna, Jarvan IV, Jax, Jayce, Jhin, Jinx, K'Sante, Kai'Sa, Kalista, Karma, Karthus,
Kassadin, Katarina, Kayle, Kayn, Kennen, Kha'Zix, Kindred, Kled, Kog'Maw, LeBlanc, Lee Sin, Leona, Lillia, Lissandra, Lucian,
Lulu, Lux, Malphite, Malzahar, Maokai, Master Yi, Mel, Milio, Miss Fortune, Mordekaiser, Morgana, Naafiri, Nami, Nasus,
Nautilus, Neeko, Nidalee, Nilah, Nocturne, Norra, Nunu y Willump, Olaf, Orianna, Ornn, Pantheon, Poppy, Pyke, Qiyana, Quinn,
Rakan, Rammus, Rek'Sai, Rell, Renata Glasc, Renekton, Rengar, Riven, Rumble, Ryze, Samira, Sejuani, Senna, Seraphine, Sett,
Shaco, Shen, Shyvana, Singed, Sion, Sivir, Skarner, Smolder, Sona, Soraka, Swain, Sylas, Syndra, Tahm Kench, Taliyah, Talon,
Taric, Teemo, Thresh, Tristana, Trundle, Tryndamere, Twisted Fate, Twitch, Udyr, Urgot, Varus, Vayne, Veigar, Vel'Koz, Vex,
Vi, Viego, Viktor, Vladimir, Volibear, Warwick, Wukong, Xayah, Xerath, Xin Zhao, Yasuo, Yone, Yorick, Yuumi, Zac, Zed, Zeri,
Ziggs, Zilean, Zoe, Zyra`.replace(/\s+/g, " ");

// Se nota que es una partida de Wild Rift (o LoL) en lo que describió la visión
export const IN_GAME_RE = /wild ?rift|league of legends|\blol\b|campe[oó]n|minimapa|jungla|torre|drag[oó]n|bar[oó]n|heraldo|nexo|kda/i;

// Lo que tiene que sacar la visión de la captura cuando es una partida
export const HUD_GUIDE = `Si es una partida de Wild Rift (o LoL), describí en 2-3 oraciones cortas: qué campeón se controla (el que sigue la cámara, con su retrato y habilidades abajo a la derecha), el marcador de muertes de cada equipo, el KDA y el tiempo de partida (arriba) si se leen, y qué está pasando (pelea, muerte, objetivo, farmeo, base) y en qué parte del mapa (línea del Barón, del medio o del Dragón, jungla propia o enemiga, Blue, Red, río, Dragón, Barón). Nombres de campeones solo de esta lista y solo si estás segura: ${CHAMPIONS}. No inventes números ni nombres que no se lean. Hablá del jugador en tercera persona ("el jugador juega Kai'Sa"), nunca "controlás" ni "tu campeón".`;

// Lo básico del juego, para que NYRA comente como alguien que lo conoce
export const GAME_PRIMER = `[Se está jugando Wild Rift, el LoL de celular (5 contra 5, unos 15-20 min por partida). Líneas: Barón (arriba, tanques y luchadores), medio (magos y asesinos), Dragón (abajo, tirador + soporte) y jungla. Objetivos: Dragones elementales, Heraldo, Barón Nashor, Dragón Anciano, torres, inhibidores y el nexo. Blue y Red son los buffs de la jungla. Los kits de los campeones son casi los mismos que en LoL: usá lo que sabés de ellos, pero los números cambian por parche, así que no afirmes daños exactos. Los nombres de campeones que te llegan por voz pueden venir mal transcritos (ej: "Horn" = Ornn, "Scarner" = Skarner): interpretalos por parecido con los campeones reales.]`;

// El campeón del jugador según lo que describió la visión ("el jugador juega Kai'Sa", "Itami, con Kayn...")
const CHAMP_ALT = CHAMPIONS.split(", ").sort((a, b) => b.length - a.length).map(c => c.replace(/\./g, "\\.").replace(/ /g, "\\s+")).join("|");
const PLAYS_RE = new RegExp(String.raw`(?:juega|jugando|maneja|controla|con)\s+(?:a\s+)?(${CHAMP_ALT})(?!\p{L})`, "iu");
export function matchChamp(desc) {
  const m = String(desc || "").match(PLAYS_RE);
  return m ? CHAMPIONS.split(", ").find(c => c.toLowerCase() === m[1].toLowerCase().replace(/\s+/g, " ")) : null;
}
// Terminó la partida (o todavía no empezó): se suelta el campeón fijado
export const MATCH_OVER_RE = /termin[oó]\s+(?:en|con)\s+(?:derrota|victoria)|men[uú] principal|\blobby\b|sala clasificatoria|selecci[oó]n de campe/i;
