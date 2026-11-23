---
name: animar-nyra
description: Cómo animar a NYRA (avatar 3D VRoid del visor en public/avatar3d) en Blender y llevarlo al visor como .vrma, y cómo revisarlo hasta que se vea natural. Usar siempre que el usuario pida una animación, gesto o pose nueva para NYRA 3D, quiera corregir una existente (saludo, té, festejo, caminar...), pase un GIF o video de referencia, o diga que algo "se ve raro / tieso / no natural". Incluye el circuito Blender → .vrma → visor, las medidas del modelo, los errores que ya cometimos y el método de revisión en grilla.
---

# Animar a NYRA

Skill viva: **al terminar cada animación, agregá lo que aprendiste en "Lecciones"** (y corregí lo que resultó estar mal).
El usuario no sabe Blender: todo se hace por script, en segundo plano, y se le muestra el resultado en el visor.
Sirve para cualquier agente (Claude Code, Antigravity/agy, Codex…): lo que depende de la herramienta está marcado.

## Antes de empezar: comprobá que tenés lo necesario

Esta skill necesita **tres capacidades**. Comprobalas al principio, no a mitad de la animación:

| capacidad | para qué | cómo comprobarla |
|---|---|---|
| Correr comandos en la PC | Blender en segundo plano, Python, el server de prueba | correr `"D:\blender-5.3.0-windows-x64\blender.exe" --version` y el chequeo del complemento VRM (abajo) |
| Ver imágenes | revisar los renders de Blender, la grilla y las capturas del visor | abrir un PNG cualquiera y describir lo que muestra |
| Navegador controlable | abrir el visor de prueba, ejecutar JavaScript en la página, sacar capturas | abrir `http://localhost:3010/obs?3d` y leer un valor con JavaScript |

**Si te falta alguna, avisale al usuario antes de empezar**, diciendo cuál falta y qué parte no vas a poder hacer:
- sin comandos: no podés animar (todo pasa por Blender en segundo plano). Decilo y no sigas;
- sin ver imágenes: solo podés revisar con números, y eso no alcanza para decir que se ve natural. Avisá que el usuario
  va a tener que mirar cada versión y describirte lo que ve;
- sin navegador: podés armar el `.vrma` y revisarlo en los renders de Blender, pero no comprobarlo en el visor.
  Avisá que el usuario tiene que abrir `/obs?3d` y probarlo.
No digas "listo" sobre algo que no pudiste ver ni medir: decí qué quedó sin comprobar.

**Blender y el complemento VRM.** Hay un solo Blender: `D:\blender-5.3.0-windows-x64\blender.exe` (5.3), en **modo
portable**. Sus ajustes y el complemento VRM están en `D:\blender-5.3.0-windows-x64\portable\` (no en `%APPDATA%`): así
todos los agentes ven lo mismo, aunque corran aislados y vean otra carpeta de usuario (pasó con agy: veía vacío el
perfil de `%APPDATA%`). El 5.2 de `C:\Program Files` se desinstaló (nov 2026). Comprobalo con este comando; tiene que
mostrar `VRM: ['bl_ext.blender_org.vrm'] op: True` y la carpeta `portable`:
```
"D:\blender-5.3.0-windows-x64\blender.exe" --background --python-expr "import addon_utils,bpy
mods=[m.__name__ for m in addon_utils.modules() if 'vrm' in m.__name__.lower()]
ok=True
try: bpy.ops.import_scene.vrm.get_rna_type(); bpy.ops.export_scene.vrma.get_rna_type()
except Exception as e: ok=repr(e)[:80]
print('VRM:', mods, 'op:', ok); print('config:', bpy.utils.resource_path('USER'))"
```
Ojo: `hasattr(bpy.ops.import_scene, 'vrm')` siempre da `True` (es un proxy dinámico); la prueba real es `get_rna_type()`.
Si falla, no instales otro Blender ni cambies de versión ni de carpeta: avisale al usuario.

**Shell en Windows.** Los scripts `.sh` se corren con Git Bash, llamándolo explícito:
`"C:\Program Files\Git\bin\bash.exe" .claude/skills/animar-nyra/herramientas/copia_prueba.sh …`.
Desde PowerShell, `bash` a secas abre el WSL de Windows (que acá no tiene distros) y falla.

## Herramientas según el agente

- **Claude Code**: `preview_start "nyra-copia"` levanta la copia y la muestra en el navegador del panel;
  `preview_stop` la apaga. Las herramientas `mcp__Claude_Browser__*` ejecutan JavaScript y sacan capturas.
- **Antigravity (agy) u otros con navegador propio**: levantar la copia con el script y abrirla en su navegador:
  ```
  bash .claude/skills/animar-nyra/herramientas/copia_prueba.sh crear     <carpeta_temporal>
  bash .claude/skills/animar-nyra/herramientas/copia_prueba.sh restaurar-launch <carpeta_temporal>
  bash .claude/skills/animar-nyra/herramientas/copia_prueba.sh arrancar  <carpeta_temporal>   # http://localhost:3010/obs?3d
  bash .claude/skills/animar-nyra/herramientas/copia_prueba.sh parar     <carpeta_temporal>
  bash .claude/skills/animar-nyra/herramientas/copia_prueba.sh limpiar   <carpeta_temporal>
  ```
  En Windows, `bash` = `"C:\Program Files\Git\bin\bash.exe"` (ver "Shell en Windows"). `parar` apaga solo el `node` que
  escucha en el 3010, después de verificarlo.
  Si tu herramienta cierra los procesos cuando termina el comando, el server de prueba muere con la terminal: levantalo
  como proceso independiente (o en una terminal que quede abierta) y apagalo igual con `parar`.
- **Para todos**: Blender, `grilla.py` y los scripts de Blender son comandos comunes y andan igual en cualquier agente.
- **Nunca apagues ni reinicies el server del puerto 3000**: es el del directo. Antes de cerrar cualquier proceso,
  verificá cuál es (puerto, nombre, quién lo arrancó). Si el 3000 no responde, avisale al usuario; no lo levantes vos.

## El circuito

1. **Referencia.** Si el usuario pasa un GIF/video, sacá sus cuadros y estudialos antes de animar (ver `herramientas/gif_cuadros.js`).
   Anotá: altura de mano/codo respecto a la cabeza, de dónde sale el giro (codo, muñeca), cuánto va a cada lado,
   ritmo (cuadros por vaivén), qué hacen los dedos, la cabeza y el torso.
2. **Script de Blender** en `scripts/blender/<gesto>.py`, copiando la estructura de `scripts/blender/saludo.py`
   (importa `public/avatar3d/NYRA.vrm`, pone cuadros clave, exporta `public/avatar3d/<gesto>.vrma`, renderiza controles).
   Blender: `D:\blender-5.3.0-windows-x64\blender.exe` con el complemento VRM (extensión oficial `vrm`).
   ```
   & "D:\blender-5.3.0-windows-x64\blender.exe" --background --python scripts/blender/<gesto>.py -- public/avatar3d/NYRA.vrm public/avatar3d/<gesto>.vrma <carpeta_renders>
   ```
3. **Revisar los renders en grilla** (`herramientas/grilla.py`) desde frente, perfil y arriba. Corregir y repetir.
4. **Visor**: agregar el gesto a `CLIPS` en `public/avatar3d/vrm.js` con la lista de huesos que toma del archivo
   (solo los que anima; el resto sigue en código). Si el `.vrma` no carga, queda el gesto en código de `GESTURES`.
   El nombre tiene que existir también en `GESTURES` de `memoria.js` (los que la IA puede pedir con `[GESTO: …]`).
5. **Revisar en el visor** con una copia aislada (`herramientas/copia_prueba.sh`, ver "Herramientas según el agente"):
   números + capturas desde 3 vistas.
6. **Mostrárselo al usuario** en bucle en el navegador, y recién con su OK, commit.

## Medidas del modelo (coordenadas del visor: y arriba, el modelo mira a +Z, su mano derecha está en x < 0)

En Blender: `three(x, y, z) → Vector((x, -z, y))` (z arriba, el modelo mira a -Y). Cabeza en Blender ≈ (0, 0.025, 1.386).

| punto | posición |
|---|---|
| hueso cabeza (base) | (0, 1.386, -0.025) — los ojos ≈ 1.45 |
| cuello | (0, 1.313, -0.034) |
| hombro derecho (UpperArm) | (-0.106, 1.275, -0.03) |
| brazo colgando: codo / nudillo medio | (-0.18, 1.07, -0.03) / (-0.24, 0.79, -0.01) |
| antebrazo codo→nudillos | 0.28 |
| asa de la taza (`taza.glb`, sin escalar) | y 0.030–0.067, z -0.056 a -0.080 |

Huesos (Blender, VRoid): `J_Bip_C_{Hips,Spine,Chest,UpperChest,Neck,Head}`, `J_Bip_{L,R}_{Shoulder,UpperArm,LowerArm,Hand}`,
dedos `J_Bip_R_{Thumb,Index,Middle,Ring,Little}{1,2,3}`. En el visor: nombres humanoides de VRM
(`rightUpperArm`, `rightIndexProximal`, `rightThumbMetacarpal`…); los nodos normalizados se llaman `Normalized_J_Bip_…`.

## Cómo posar sin adivinar (lo que funcionó)

- **Apuntar huesos a posiciones objetivo** (`aim`), no girar ángulos a ojo: codo hacia un punto, antebrazo hacia los nudillos.
  Las posiciones se piensan en el espacio (¿dónde está la mano respecto a la cara?), que es lo que el usuario ve.
- **Giros con signo comprobado** (`rotate_toward`): girar, medir hacia dónde se movió la punta y, si fue al revés, invertir.
  Nunca asumir que "x positivo es adelante": en este esqueleto depende de la pose (ver Lecciones).
- **Buscar la orientación por medición**: p. ej. la palma a cámara se encontró probando giros de muñeca y midiendo la
  normal de la palma (`palm_forward`). Con números, no con capturas.
- **Mover los objetivos con el torso**: si el torso se inclina, transformar los objetivos de la mano con el movimiento del
  pecho para que la mano acompañe al cuerpo; el brazo libre apunta a posiciones fijas del mundo (cuelga por gravedad).
- **Trayectorias**: entre dos poses lejanas poné una intermedia (la mano pasa frente al pecho con el codo doblado).
  Interpolar directo abre el brazo hacia el costado o lo pasa por detrás del cuerpo.

## Principios que hacen que se vea natural (aprendidos con el saludo)

- **Todo el cuerpo acompaña**: el torso compensa el peso del brazo (se inclina al lado contrario), el pecho gira apenas,
  el cuello endereza la mitad, la cabeza se inclina hacia la mano, el otro brazo no queda tieso. Sin esto "se ve tieso".
- **Superposición**: hombro y codo arrancan primero, después antebrazo, muñeca y dedos; el pecho llega 1 cuadro tarde y
  la cabeza 2 (`key(frame, finger_lag, body_lag)`).
- **Amago y rebote**: cargarse apenas al lado contrario antes del gesto y pasarse un poco del centro al terminar.
- **El giro sale de la articulación correcta**: un saludo es un limpiaparabrisas desde el codo con la muñeca exagerando,
  no la mano desplazándose de lado.
- **Asimetría de la referencia**: el saludo va más hacia la cabeza que hacia afuera. Copiar la referencia, no simetrizar.
- **Dedos vivos**: cascada (índice más recto, meñique más curvo), se juntan o abren en abanico según la fase, las puntas
  llegan un cuadro tarde; el pulgar nunca en pose T (acercarlo al índice y curvarlo).
- **Intensidad**: la primera versión suele quedar tímida (2 cm no se notan). Medir y subir hasta que se vea en el encuadre.
- **Ritmo**: 30 cuadros/s; un vaivén de saludo ≈ 0,4 s (12 cuadros ida y vuelta).

## El ojo: disciplina de revisión (lo que más mejora el resultado)

La calidad no sale de la primera versión: sale de **renderizar, mirar, medir y corregir varias veces**. El saludo
necesitó tres versiones y varias rondas cada una. No es opcional ni depende de "ser meticuloso": seguí estos pasos.

1. **Mínimo dos rondas** de render → revisión → corrección antes de mostrarle algo al usuario. Si en la segunda ronda
   encontrás otro error, hacé una tercera.
2. **Describí por escrito cada imagen de la grilla** antes de decidir: dónde está la mano respecto a la cara (arriba,
   abajo, adelante, atrás), qué hacen el codo, los dedos, la cabeza, el torso y el otro brazo. Escribirlo obliga a
   mirar de verdad; "se ve bien" sin describir no cuenta.
3. **Repasá la lista de errores conocidos** (sección Revisión) imagen por imagen.
4. **Compará contra la referencia** lado a lado si el usuario pasó una: altura, de dónde sale el giro, asimetrías, ritmo.
5. **Medí con números** los puntos clave (mano, codo, cabeza) y compará con los objetivos. Si la imagen y los números no
   coinciden, creele a los números y averiguá por qué la imagen engaña.
6. **Después del último cambio, otra grilla nueva**: nunca digas que está bien con capturas de una versión anterior.
7. **Reportá lo que viste, incluidos los errores** que encontraste y corregiste, y lo que quedó sin comprobar
   (p. ej. "no lo vi de corrido, solo cuadros sueltos").

## Revisión (obligatoria antes de decir "está bien")

- **Grilla**: 5–8 cuadros × frente / perfil / arriba. Mirar cada imagen buscando lo obvio: mano detrás del cuerpo,
  brazo de zombi, ropa que se rompe (axilas del buzo), palma de canto, dedos tiesos, objetos flotando o mal agarrados,
  parpadeo sobre ojos cerrados, cuerpo quieto mientras el brazo se mueve.
- **La vista de arriba decide adelante/atrás**; la de perfil engaña (una mano estirada al costado queda cerca de la
  cámara y parece estar adelante de la cara). Arriba: los pies quedan abajo = adelante.
- **Números > capturas**: medir posiciones de mano, codo, cabeza (`getWorldPosition`) en cada cuadro clave y compararlas
  con los objetivos. Las capturas del navegador de prueba llegan atrasadas: la pestaña queda "oculta" y corre a ~1,5 fps
  (usa la GPU, no es problema de OBS). Esperar ≥1,5 s tras cambiar la pose antes de capturar.
- **Congelar un cuadro** en el visor: `__G.t1 = { ...G, dur: 999, clip: () => G.clip(t, 1) }; nyra3d.gesture('t1')`
  (para gestos en código: `pose: (p, rest) => G.pose(t, rest)`, y `walkP: () => t` si camina).
- Primeros planos para manos y dedos; plano entero para el cuerpo; una línea vertical de referencia muestra la inclinación.

## Lecciones (agregar al final de cada animación)

- **Saludo (nov 2026)**: tres versiones. 1) Mano en la mejilla y desplazamiento lateral: "se ve raro". 2) Siguiendo un GIF:
  mano a la altura de la frente, limpiaparabrisas desde el codo, dedos en abanico: aprobado. 3) Con el cuerpo acompañando:
  "esta es la buena". El usuario nota enseguida lo tieso: siempre animar el cuerpo, no solo el miembro.
- En el visor, con ángulos Euler en huesos normalizados, `x` del brazo **no** es "adelante" fijo: con el brazo colgando,
  x negativo lleva el codo adelante, pero doblar el antebrazo en `z` lo sube **hacia atrás**. Se resolvió midiendo.
- Mano de taza: el asa se engancha en la 2.ª falange del índice, pulgar acostado encima, resto doblado debajo; la taza
  se orienta sola cada cuadro (derecha al sostener, hacia los labios al tomar) en vez de depender del giro de la mano.
- Festejo: brazos arriba rompen las axilas del buzo; puños frente al pecho con codos al costado funciona.
- Caminar: sin piernas alternadas se desliza; brazos opuestos a las piernas.
- Herramientas: `sed` de reemplazo puede pisar también la definición de la función (pasó con `walkOff`/`walkPose`):
  usar reemplazos exactos con Python y verificar con `node --check`.
- Los cambios de otra IA (GPT) en este visor se dejaron en la rama `gpt/animaciones-vrma`: no mezclarlos con main sin
  que el usuario lo pida.
- Cada agente puede ver una carpeta de usuario distinta (agy corre aislado y veía vacío `%APPDATA%`). Lo que tienen
  que compartir todos va a una ruta fija: por eso Blender quedó en modo portable en `D:\`. Y un chequeo tiene que probar
  lo real (`get_rna_type()`), no algo que siempre da `True`.
- Procesos en Windows: el PID que da Git Bash (`$!`) no es el del `node` real, así que `kill` puede apuntar a otro
  proceso. Por eso `parar` busca quién escucha en el 3010 y verifica antes de apagar. Pasó que el server del directo
  (3000) apareció apagado justo después de un `kill` sin verificar: nunca matar procesos a ciegas.
