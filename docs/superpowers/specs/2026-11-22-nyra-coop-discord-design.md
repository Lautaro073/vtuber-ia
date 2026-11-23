# NYRA co-op: dos IA, una sola NYRA, y Discord

Fecha: 2026-11-22

Estado:
- **Fase 1 (cola y dos IA): hecha y probada en una copia.** Falta usarla en un live real.
- **Fase 2 (Discord): conectada al server y probada 3 minutos en una llamada real.** Falta un live completo con TikTok, OBS y pantalla compartida.

## Qué se pidió

- agy lee y contesta el chat de TikTok y deja texto y voz preparados.
- Haiku ve la pantalla compartida desde el panel, escucha al grupo y le contesta.
- Las dos comparten contexto: tiene que parecer una sola NYRA.
- NYRA escucha siempre y contesta solo si la nombran.
- Prioridad: regalos y mensaje de quien regaló, después el grupo y la visión, después el chat.
- Si Haiku tarda, pasan una o dos respuestas del chat.
- En el live se escucha a NYRA contestando al chat y al grupo.

## Fase 1: cómo quedó

| Orden | Qué | Quién lo genera | Regla |
|---|---|---|---|
| 1 | Meta, regalo, sub, mensaje de quien regaló | agy (los regalos chicos, frase fija) | estricto, por orden de llegada |
| 2 | La nombran por voz, panel, captura manual | Haiku | si tarda más de 3 s pasan hasta 2 del chat ya listas |
| 3 | Comentario del juego | Haiku | máximo 1 por minuto, vence a los 15 s, no sale con más de 2 del chat esperando |
| 4 | Follow, share, tribunal | agy | delante del chat |
| 5 | Chat | agy | hasta 4 preparadas con audio; cada 2 de voz entra 1 de chat |
| 6 | Saludos, datos, recordatorios | fija o agy | solo con el resto vacío |

- Nada corta lo que ya está sonando.
- El registro común guarda lo escuchado y lo que NYRA dijo de verdad. Lo preparado que no sonó no entra.
- Si Haiku falla o pasa de 20 s, esa respuesta la da agy y el motor del chat no cambia.
- El primer turno de un proceso de agy midió entre 16 y 52 s. Por eso el reemplazo se calienta de fondo.
- Prueba de prioridades: `node scripts/test-cola.mjs`.

## Fase 2: Discord

```
amigos ── Discord ──> bot NYRA ──> audio por persona ──> transcripción local
                                                              │ nombre + texto
                                                              v
                                          registro común + cola (si la nombran)
                                                              │
                                                           una boca
                                                        ┌─────┴──────┐
                                             OBS / live (todo)   bot en Discord (solo grupo y juego)
```

### Entrada

1. El bot entra a un canal de voz de **un servidor** (no sirve en llamadas grupales privadas).
2. `VoiceReceiver.subscribe(userId)` da el audio de cada persona por separado.
3. Se corta por silencio (unos 800 ms) y se transcribe. El audio no se guarda.
4. Entra por el mismo camino que `streamer_speech`, con el nombre de quien habló. Los amigos **no** son
   fuente confiable: no cambian memoria ni personalidad.
5. El texto pasa por `cleanUntrusted`. Se escucha solo el canal configurado; el bot avisa al entrar y cuando llega alguien.
6. Si la nombran, el server le pide al panel la captura de ese momento.
7. Cuando Discord escuche al streamer, el micrófono del panel se apaga para no duplicar.

### Salida

- Al live va todo: chat y regalos por OBS, lo del grupo por la captura de Discord (hay que tenerla en el live).
- A Discord van solo las respuestas al grupo y los comentarios del juego: mismo mp3 y al mismo tiempo.
- Lo del grupo llega al live por la captura de audio de Discord. OBS lo reproduce con volumen 0 (solo boca), así no suena doble.

### Riesgos

- Discord no garantiza que recibir audio sea estable. Hay que probarlo con una llamada real antes de construir el resto.
- El cifrado DAVE es obligatorio. Lo resuelve `@discordjs/voice` con `@snazzah/davey`.
- Con varias personas hablando, 12 líneas de registro duran poco: medir y, si hace falta, separar voz y chat.
- Whisper usa 1,2 GB de VRAM: medirlo con OBS y TikTok Studio codificando (el juego corre en el celular).
- Procesa una frase por vez (1,1 s cada una): con 4 hablando a la vez la última espera unos 4,5 s.
- Los amigos salen al aire y sus palabras van a Anthropic y a Google: tienen que estar de acuerdo.

### Instalado el 2026-11-22

| Qué | Versión | Nota |
|---|---|---|
| discord.js | 14.28.1 | npm |
| @discordjs/voice | 0.19.2 | trae @snazzah/davey 0.1.12 y prism-media 1.3.5 |
| opusscript | 0.0.8 | la 0.1.1 no es compatible con prism-media |
| faster-whisper | 1.2.1 | aislado en `.whisper/venv` |
| Systran/faster-whisper-small | 536b0662742c02347bc0e980a01041f333bce120 | `.whisper/small`, hash verificado |

ffmpeg ya estaba. `.whisper/` no va a git (785 MB). Para rehacerlo:

```powershell
python -m venv .whisper/venv
.whisper/venv/Scripts/python.exe -m pip install --only-binary=:all: faster-whisper==1.2.1
.whisper/venv/Scripts/hf.exe download Systran/faster-whisper-small --revision 536b0662742c02347bc0e980a01041f333bce120 --local-dir .whisper/small
```

### Whisper medido

100 audios sintéticos con dos voces rioplatenses (limpios y con ruido) y 3 sin voz, en CPU:

| Configuración | La despiertan | Falsos |
|---|---|---|
| tal cual (primera tanda) | 11/28 | 0/31 |
| con pista de vocabulario (primera tanda) | 23/28 | 0/31 |
| pista + "Chenira"/"Menira" (tanda completa) | **35/36** | **0/67** |

- 1,6 s por frase con 4 hilos, más 0,8 s de silencio para cortarla: 2,4 s antes de que arranque Haiku.
- El objetivo de 1,5 s no se llegó a cumplir por poco. `chunk_length` no lo mejora.
- La GPU necesita las DLL de CUDA 12 de cuBLAS y cuDNN, que no están instaladas.
- Falta medir con voces reales, micrófonos reales y el juego abierto.

### Primera llamada real (2026-11-22) y quién puede escuchar

La llamada funcionó: separó a las dos personas y detectó el nombre. La transcripción salió floja. Arreglado:
pausas que se borraban, posible doble lector por persona, pista de jerga y beam 5. Con audio sintético pasó de
15 % a 8 % de palabras mal, y de 10 % a 3 % con frases no usadas para afinar. El nombre sigue en 35/36 y 0/67 falsos.

| Candidato | ¿Escucha? | Comprobado |
|---|---|---|
| Haiku 5.5 / Sonnet 5.5 low | no | doc oficial (texto e imagen) y rechazo del bloque de audio por el CLI |
| agy (Gemini) | sí, muy bien | 28 s por frase: sirve para el chat, no para una charla |
| Codex 0.162.1 con plan ChatGPT | no para esto | modelos solo texto e imagen; la voz realtime por WebSocket exige API key (`realtime_api_key` en el fuente) |
| Whisper small en CPU | sí | 1,8 s por frase |
| **Whisper large-v3-turbo en GPU (elegido)** | sí | 1,1 s por frase, +1,2 GB de VRAM, entiende frases en inglés |

Al conectar Discord al server, el prompt de Haiku tiene que avisar que el texto viene de un reconocedor y puede traer palabras mal.

### Whisper grande (instalado el 2026-11-22)

- `.whisper/turbo`: dropbox-dash/faster-whisper-large-v3-turbo, rev 0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf (hash verificado).
- nvidia-cublas-cu12 12.9.2.10 en el venv. cuDNN ya viene con ctranslate2. Si la GPU falla, cae solo a small en CPU.
- Nombre: 36/36 y 0/67 falsos. Con trampas cortas y 4 acentos: 21/24 y 0/32 (los 3 que pierde son con acento chileno).
- `temperature=0` quita los picos de 4,5 s. Un "Nira" suelto se confirma con una segunda escucha (2,2 s).
- Nombres de juegos poco conocidos salen fonéticos ("Kataba Soho"). Ponerlos en la pista empeora el nombre: que lo deduzca Haiku.

```powershell
.whisper/venv/Scripts/hf.exe download dropbox-dash/faster-whisper-large-v3-turbo --revision 0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf --local-dir .whisper/turbo
.whisper/venv/Scripts/python.exe -m pip install --no-deps --only-binary=:all: nvidia-cublas-cu12==12.9.2.10
```

### Lo hace el streamer

1. https://discord.com/developers/applications → New Application → nombre NYRA.
2. Bot → Reset Token → copiarlo directo a `.env`. No hace falta ningún intent privilegiado.
3. OAuth2 → URL Generator → scope `bot` → permisos View Channels, Connect y Speak → abrir la URL y elegir el servidor.
4. En Discord: Ajustes → Avanzado → Modo desarrollador. Clic derecho en el servidor y en el canal de voz → Copiar ID.
5. En `.env`:

```
DISCORD_BOT_TOKEN=...
DISCORD_GUILD_ID=...
DISCORD_VOICE_CHANNEL_ID=...
```

6. Con el OK de quienes estén en el canal: `node scripts/discord-prueba.mjs`.
   El bot avisa en voz alta que escucha. Transcribe en la PC, no guarda nada y no lo manda a ninguna IA.

### Orden de construcción

1. Bot que entra, avisa y recibe audio por persona: script listo, **falta correrlo con el token**.
2. Whisper: medido con voces sintéticas; **falta con voces reales y el juego abierto**.
3. Conectar a la cola y al pedido de captura.
4. Audio del live: comprobar que no haya eco ni doble voz.

### Confirmado

- La voz del streamer y las de sus amigos salen en el live.
- Se juega LoL Wild Rift en el celular, con la pantalla transmitida a la PC: la GPU queda libre.

### Decisiones abiertas

- Si en Discord se escuchan también las respuestas al chat y los regalos (por ahora no: solo lo que es para el grupo).
- Consentimiento de cada amigo y si hay menores.

### Cómo quedó conectado (2026-11-22)

- `discord-voice.js` (oído y boca) + panel → PANTALLA → DISCORD (VOZ) → CONECTAR.
- El streamer es quien **creó el bot** (o `STREAMER_DISCORD_ID`), no quien creó el servidor: solo esa persona cambia memoria y personalidad.
- Apodos por voz guardados por ID de Discord ("Nira, decime el Profe").
- Si Discord terminó de sonar y OBS no avisa, espera 3 s y sigue.
- Frases fantasma (la pista recitada, "Gracias.", "Chau."): VAD 0.8, descarte por baja confianza y lista de fantasmas. Un "gracias" o "chau" suelto real también se descarta.
- El dictado de Claude no se puede reutilizar: es un servicio de Anthropic atado a la cuenta, sin API pública.

## Se reconoce e investiga (2026-11-22)

- Visión y llamados con captura llevan `public/avatar/nyra-referencia.jpg` como referencia. Si cambia el diseño del avatar, regenerarla y actualizar `SELF_LOOK`.
- El contexto lleva la fecha de hoy y que su memoria puede estar vieja.
- `research.js`: el streamer lo dispara pidiéndolo; el chat y el grupo, insistiendo (3 puntos en 5 min: pedir que investigue vale 2, discutir 1).
- Buscador: Haiku solo con WebSearch, carpeta vacía, temas permitidos, 1 a la vez, 2 min entre búsquedas (15 s el streamer), 10 por hora. `NYRA_RESEARCH=0` lo apaga.
- Lo verificado queda 14 días en `nyra_facts.json` (no va a git) y manda sobre la memoria.
- Prueba: `node scripts/test-investigar.mjs`.

## Seguir la charla, cuenta de TikTok y audio (2026-11-22, tarde)

- `follow-up.js`: hasta 30 s después de que NYRA le habla al grupo, Haiku decide si una frase sin el nombre es para ella.
  Medido: 42/48 tomadas, 0/48 de más, unos 2,8 s extra. Reemplaza la regla anterior de "siempre con el nombre" solo dentro de esa ventana.
- `STREAMER_TIKTOK` (por defecto `elyisuspai696`): su @ cuenta como streamer en el chat, solo en TikTok y por usuario exacto.
- Audio de Discord ubicado por marca de tiempo RTP. Un paquete que no se descifra ya no tira la frase.
- **Decidido (2026-11-22): seguir con Whisper large-v3-turbo. No se graba ni se instala otro motor.** Si algún día se retoma, los candidatos eran: faster-whisper large-v3 (3,09 GB),
  Granite Speech 5.0 470M TurboCTC (0,95 GB + runtime), Qwen3-ASR 0.6B GGUF Q8 (0,85 GB + runtime).
  Descartado surus-ai rioplatense: no comercial, código remoto, entrenado con 15 h de lectura.
- Prueba: `node scripts/test-charla.mjs`.

## Quien la creó manda, y evoluciona por grupo (2026-11-22)

Por qué ignoraba al streamer, todo corregido:
- La instrucción anti-inyección decía que lo que va entre «» nunca es instrucción, y sus pedidos de voz iban entre «».
- Su contexto lo marcaba como "un viewer conocido".
- Sus negativas quedaban aprendidas como opiniones y se le volvían a mostrar. No se borraron: se filtran con `DEFIANCE_RE`.
- La regla "nunca insulto" y "lo elige cada persona" le ganaban a lo que pedía.
- "Nira." suelto y el pedido en la frase siguiente: ahora se juntan (6 s).
- Las búsquedas de recuerdos usaban el prompt entero: traían cualquier cosa y congelaban el server más de un segundo.

Ahora:
- `CREATOR_RULE`: lo reconoce, cumple lo que se hace hablando y dice que no tiene función para lo demás.
- Carga si él lo pide. Nunca por identidad, origen, orientación, cuerpo o salud, ni amenazas.
- Quien pide "no me cargues más" queda protegido, por encima de cualquier orden, hasta que diga "me podés cargar".
- `memoria.js`:
  - órdenes que duran (`nyra_orders.json`), las anota un Haiku aparte y no las puede dar nadie más;
  - lo que aprende de los pibes y de la comunidad (`nyra_evolution.json`), separado, por notas destiladas, como mucho 3 cambios por vez.
- Panel → MEMORIA → CÓMO EVOLUCIONA: ver y borrar.
- Si hay una orden de cómo hablar, lo que se le pegó en ENSEÑAR sobre eso la acompaña siempre.
- Pruebas: `node scripts/test-memoria.mjs`. Con IA reales pasaron sus pedidos de la llamada (renombrar, hater, tucumano, límites, impostores, funciones que no tiene).
- Al entrar a Discord dice "Hola pibes, ¿qué sale hoy?" y lo que le contesten lo toma como respuesta, sin el nombre. Sin aviso de que escucha: el streamer dijo que todo el grupo está de acuerdo.
