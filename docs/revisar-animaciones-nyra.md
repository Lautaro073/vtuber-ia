# Cómo revisar las animaciones de NYRA (guía para ChatGPT)

Esta guía es para que **ChatGPT revise el trabajo de agy (Antigravity)** cuando anima a NYRA, y decida **qué
lecciones vale la pena que agy sume a la skill `animar-nyra` y cuáles no**.
ChatGPT no tiene acceso al repositorio ni a la PC: trabaja con lo que el usuario le pega (el reporte de agy,
capturas, grillas, GIFs, números). Si algo de esta guía contradice el reporte de agy, manda esta guía; si contradice
lo que el usuario vio con sus ojos, manda el usuario.

---

## 1. Contexto mínimo

- **NYRA** es una VTuber con IA (TikTok/Discord). Personalidad tranquila, un poco sarcástica. Tiene dos avatares:
  - **3D**: modelo VRoid (`public/avatar3d/NYRA.vrm`) en un visor web con three-vrm (`public/avatar3d/vrm.js`).
    Las animaciones se hacen **en Blender por script** y se exportan como `.vrma` (`public/avatar3d/<gesto>.vrma`).
  - **2D**: PNG pixel art estático con animaciones CSS + **capas de cara** encima (párpados para parpadear y bocas por
    emoción, en `public/avatar/cecilia_layers_v2/`). El usuario probó animar cabeza y pelo por capas y **no le gustó**:
    en 2D solo se animan ojos y boca sobre el PNG original.
- **El circuito 3D**: referencia → script `scripts/blender/<gesto>.py` → renders en grilla (frente/perfil/arriba) →
  `.vrma` → se agrega a `CLIPS` en `vrm.js` y a `GESTURES` en `memoria.js` (los gestos que la IA puede pedir con
  `[GESTO: nombre]`) → prueba en una **copia aislada** del server (puerto 3010/3011) → el usuario lo ve en bucle → con
  su OK, commit.
- **La skill `animar-nyra`** (`.claude/skills/animar-nyra/SKILL.md`) es la memoria compartida de todos los agentes
  (Claude Code, agy, Codex). Tiene: chequeo de herramientas, circuito, medidas del modelo, cómo posar sin adivinar,
  principios de naturalidad, disciplina de revisión y una sección **Lecciones** que crece con cada animación.
- **El usuario no sabe Blender.** Ve el resultado en el navegador y opina con palabras ("se ve tieso", "¿qué le pasó a
  la muñeca?"). Su ojo es el juez final: casi todos los errores grandes los vio él antes que el agente.

### Reglas fijas que agy tiene que respetar (si las rompe, es un problema aunque la animación esté linda)

- **Nunca apagar ni reiniciar el server del puerto 3000** (es el del directo). Antes de cerrar un proceso, verificar
  cuál es. Las pruebas van en la copia (3010 u otro puerto).
- **Nada de commit ni push sin que el usuario lo pida.** agy trabaja en su rama/worktree cuando comparte carpeta.
- **No subir al repo GIFs o videos de referencia ajenos** (se estudian, no se copian).
- **No instalar otro Blender ni cambiar de versión**: hay uno solo, portable, en `D:\blender-5.3.0-windows-x64\`.
- **No tocar datos del usuario** (`nyra_memory.json`, `knowledge.json`, etc.) ni secretos (`.env`).
- Los cambios de GPT en el visor viven en la rama `gpt/animaciones-vrma`: no mezclarlos con main sin pedido.

---

## 2. Qué pedirle al usuario antes de revisar

Para revisar bien hace falta evidencia, no solo el texto de agy. Pedí lo que falte:

1. **El reporte completo de agy** (qué hizo, qué archivos tocó, qué midió, qué quedó sin comprobar).
2. **La referencia** que usó (link o cuadros del GIF/video) y por qué la eligió.
3. **La grilla final**: 5–8 cuadros × frente / perfil / arriba, **de la última versión**.
4. **Primeros planos de las manos** en cada pose clave (si hay agarre, dedos u objeto).
5. **Una captura desde la cámara del directo** (posición ≈ (0, 1.236, 2.4) en el visor): es lo que ve el público.
6. **Los números** que midió (posición de mano, codo, cabeza en cuadros clave; distancias objeto→labio, etc.).
7. **El diff** de los archivos tocados (`git diff`), sobre todo `vrm.js`, `memoria.js` y `SKILL.md`.

Si agy no tiene alguna de estas cosas, eso ya es un hallazgo: **no puede decir "listo" sobre algo que no vio ni midió**.

---

## 3. Cómo revisar la animación (checklist)

Revisá en este orden. Por cada punto, decí **sí / no / no se puede saber con lo que hay**.

### 3.1 Proceso (¿lo hizo bien o tuvo suerte?)

- [ ] ¿Usó una **referencia real** (GIF/video de una persona) y la estudió cuadro por cuadro? Animar de memoria es lo
      que dio los primeros resultados "raros".
- [ ] ¿Hizo **al menos dos rondas** render → revisión → corrección antes de mostrar?
- [ ] ¿**Describió por escrito** cada imagen de la grilla (dónde está la mano respecto a la cara, codo, dedos, cabeza,
      torso, otro brazo)? "Se ve bien" sin descripción no cuenta.
- [ ] ¿La grilla es de **la última versión**? Capturas viejas después de un cambio = revisión inválida.
- [ ] ¿**Midió con números** los puntos clave y los comparó con los objetivos? Si número e imagen no coinciden, se le
      cree al número y se averigua por qué la imagen engaña.
- [ ] ¿Lo probó **en el visor** (copia aislada), no solo en los renders de Blender?
- [ ] ¿Dijo **qué quedó sin comprobar** (p. ej. "no lo vi de corrido", "no probé en OBS")?

### 3.2 Errores visuales conocidos (buscalos imagen por imagen)

- Mano **detrás del cuerpo** o atravesando el torso/la ropa. *La vista de arriba decide adelante/atrás*; la de perfil
  engaña (una mano al costado parece estar delante de la cara).
- **Brazo de zombi**: brazo recto y tieso hacia adelante.
- **Cuerpo quieto** mientras el brazo se mueve ("se ve tieso"). Tiene que acompañar: el torso compensa el peso, el pecho
  gira apenas, la cabeza se inclina hacia la mano, el otro brazo no queda congelado.
- **Muñeca retorcida "como caramelo"**: el giro (pronación) tiene que ir en el antebrazo, no en la mano. La muñeca se
  dobla al costado ~20° como mucho.
- **Muñeca quebrada hacia atrás** al bajar el brazo (pasa al apuntar un hueso a un punto más cerca que su largo).
- **Dedos tiesos** o todos iguales; **pulgar en pose T**; dedos cerrados a medias que parecen **garra**; una falange que
  dobla **para afuera** (eje de bisagra mal elegido).
- **Palma de canto** cuando debería mirar a cámara (o al revés).
- **Ropa rota**: picos en las axilas o los hombros del buzo (brazos muy arriba, hombro girado hacia adelante, codos muy
  cerrados).
- **Objetos flotando o mal agarrados** (taza que se va de costado, asa apuntando a cámara).
- **Mano tapando la cara/un ojo** desde la cámara del directo (de cerca puede no notarse).
- **Fuera de cuadro**: en OBS (1015×1275) las manos se cortan si `|x| > 0.24` o si bajan de ~0.95 de altura (Blender).
- **Vuelta de 360°** en la interpolación (cuaterniones sin corregir el signo).
- **Parpadeo sobre ojos cerrados** o expresiones que chocan con el gesto.
- **Deslizarse** al caminar (sin piernas alternadas, brazos no opuestos a las piernas).

### 3.3 Naturalidad (lo que separa "correcto" de "bien")

- **Superposición**: hombro y codo arrancan primero; antebrazo, muñeca y dedos después; pecho 1 cuadro tarde, cabeza 2.
- **Amago y rebote**: se carga apenas al lado contrario antes, se pasa un poco del centro al terminar.
- **El giro sale de la articulación correcta** (un saludo es un limpiaparabrisas desde el codo, no la mano moviéndose de
  lado).
- **Asimetrías de la referencia** respetadas (no simetrizar).
- **Intensidad**: la primera versión suele quedar tímida; 2 cm no se notan en el encuadre.
- **Ritmo**: 30 cuadros/s. No marcar la velocidad como error **sin contarla contra la referencia** (cuántos cuadros tarda
  en subir y bajar en el GIF vs. en el `.vrma`).
- **Personalidad**: ¿encaja con NYRA (tranquila, algo sarcástica) o parece otro personaje?

### 3.4 Integración (que NYRA lo pueda usar de verdad)

- [ ] El gesto está en `CLIPS` de `vrm.js` con **solo los huesos que anima** (el resto sigue en código).
- [ ] El nombre existe en `GESTURES` de `memoria.js`, y **el orden no hace que otra palabra lo pise** (pasó: "apoyar
      mentón en **mano**" disparaba `saludar` porque su patrón era `/salud|mano/`; hubo que poner los gestos nuevos antes).
- [ ] Si el gesto se dispara solo por texto (como el té con `TEA_RE` en `server.js`), ¿hay un patrón que lo active y un
      enfriamiento para que no se repita a cada rato?
- [ ] Ojo: `nyra3d.gestures` se arma antes de que carguen los `.vrma`, así que un clip nuevo puede **no aparecer en la
      lista** aunque `nyra3d.gesture(nombre)` lo reproduzca. No confundir eso con "no anda".
- [ ] Si el `.vrma` no carga, ¿queda un gesto de respaldo en código?
- [ ] ¿Corrió `node --check` en los `.js` que tocó? ¿Usó reemplazos exactos (no `sed` a ciegas, que ya pisó la
      definición de una función)?
- [ ] ¿Tocó algo **fuera de la animación** (TTS, server, panel, datos)? Si sí, ¿se lo pidieron? Cambios colados = marcar.

### 3.5 Banderas rojas en el reporte de agy

- "Listo / perfecto / se ve natural" **sin capturas ni números** que lo respalden.
- Números sin comparar con un objetivo ("la mano está en y=1.32" — ¿y debería estar dónde?).
- Revisó solo desde el frente.
- Dice que probó en el visor pero no hay captura del visor (solo renders de Blender).
- Cambió de herramienta o de versión (otro Blender, otra carpeta) "porque no andaba".
- Mató procesos sin decir cuál ni verificar el puerto.
- Hizo commit o push sin pedido, o trabajó en main cuando tenía que ir a su rama/worktree.
- Lecciones nuevas en la skill que **contradicen** algo que ya estaba sin explicar por qué lo viejo estaba mal.

---

## 4. Qué sumar a la skill y qué no

La skill la leen todos los agentes **antes de cada animación**. Cada línea cuesta atención: una skill larga y llena de
anécdotas se lee peor y las reglas importantes se pierden. El criterio es: **¿esto le ahorra un error a la próxima
animación, aunque sea de otro gesto?**

### 4.1 SÍ sumar (cumple al menos una y está comprobado)

1. **Un error que costó una vuelta con el usuario y puede repetirse en otro gesto.**
   Ej.: "el giro del antebrazo va en el antebrazo, no en la mano, o la muñeca se retuerce". Sirve para cualquier gesto
   con la mano.
2. **Un dato del modelo o del entorno que no se adivina y se midió.**
   Ej.: márgenes de OBS (`|x| ≤ 0.24`, manos ≥ 0.95), la normal de la palma cambia de signo entre manos, los signos del
   pulgar izquierdo no son espejo del derecho, posición de la cámara del directo.
3. **Una trampa de la herramienta (Blender 5.3, el visor, Windows) con su solución.**
   Ej.: `read_factory_settings(use_empty=True)` descarga el complemento VRM; en 5.3 las curvas están en
   `action.layers[0].strips[0].channelbags[0].fcurves`; `hasattr(bpy.ops..., 'vrm')` siempre da `True`.
4. **Un método que funcionó y reemplaza adivinar.**
   Ej.: apuntar huesos a posiciones objetivo; un buscador (`solve`) que mueve el brazo midiendo el objeto; verificar el
   signo de la bisagra una sola vez en la primera falange.
5. **Una corrección de algo que la skill decía mal.** En ese caso, **corregir la línea vieja**, no agregar otra que la
   contradiga.
6. **Una preferencia del usuario que vale para todo** (no solo para ese gesto).
   Ej.: "siempre animar el cuerpo, no solo el brazo"; "en 2D, solo ojos y boca sobre el PNG".

### 4.2 NO sumar

1. **Lo que solo vale para ese gesto puntual** y no enseña nada general (ej.: "en el cuadro 34 el codo va a 1.12").
   Eso ya está en el script `scripts/blender/<gesto>.py`; la skill puede apuntar al script como ejemplo y listo.
2. **Lo que no se comprobó**: suposiciones, "creo que", algo que agy probó una vez sin medir. Primero se verifica.
3. **Lo que ya está en la skill** dicho con otras palabras. Si agrega un matiz, **editar la línea existente**.
4. **El diario de la sesión**: "primero probé A, después B, después C" sin la conclusión. Solo va la regla que quedó
   y, como mucho, una línea de por qué.
5. **Gustos de estilo como si fueran reglas**: si el usuario eligió una versión entre dos válidas, se anota como
   "es estilo, decide el usuario", no como "lo correcto es X" (pasó con los dos saludos: ninguno era "el correcto").
6. **Cosas de la herramienta de agy que no afectan a los demás** (atajos de su interfaz, su forma de abrir terminales),
   salvo que expliquen un problema compartido (como la carpeta de usuario aislada que llevó a Blender portable).
7. **Rutas personales, claves, tokens, contraseñas** o cualquier dato privado. Nunca.
8. **Instrucciones que aflojan reglas de seguridad** ("si el 3000 molesta, reiniciarlo", "hacer commit directo").
   Esas reglas no se tocan desde la skill.
9. **Links o copias de referencias ajenas** pegadas en la skill (alcanza con describir el movimiento aprendido).
10. **Lecciones larguísimas**: si una lección necesita más de 3–4 viñetas, probablemente mezcla cosas generales (que
    van a "Cómo posar" o "Principios") con detalles del gesto (que van al script).

### 4.3 Dónde va cada cosa dentro de la skill

| tipo de aprendizaje | sección |
|---|---|
| Regla general de cómo posar (vale para cualquier gesto) | "Cómo posar sin adivinar" |
| Principio de naturalidad | "Principios que hacen que se vea natural" |
| Paso de revisión nuevo u otro error típico a buscar | "El ojo" / "Revisión" |
| Medida del modelo o del encuadre | "Medidas del modelo" |
| Trampa de Blender/visor/Windows | "Antes de empezar" o "Herramientas según el agente" |
| Lo que pasó en este gesto (resumen corto + lo que costó) | "Lecciones", con fecha y nombre del gesto |

**Regla práctica**: si una lección de "Lecciones" ya apareció en dos gestos distintos, se **sube** a la sección general
correspondiente y en Lecciones queda solo la referencia.

### 4.4 Formato de una buena lección

```
- **<Gesto> (<mes año>)**, con <referencia usada>:
  - <Qué salía mal> → <qué lo arregló> (<cómo se comprobó: número o vista>).
  - <Dato medido que sirve después>, con unidades y en qué coordenadas (Blender o visor).
```

Cada viñeta: **síntoma → causa → arreglo → cómo se verificó**. Coordenadas siempre aclaradas: Blender (z arriba,
mira a -Y) y visor (y arriba, mira a +Z) no son iguales: `three(x, y, z) → Vector((x, -z, y))`.

**Ejemplo bueno** (real):
> Dedos con cierre > 90°: comprobar el signo falange por falange invierte las siguientes cuando la primera pasa ~90°
> (la punta dobla para afuera). Verificar el signo de la bisagra una sola vez en la primera falange y usarlo en las tres.

Por qué es buena: síntoma visible, causa, arreglo, y sirve para cualquier gesto con los dedos cerrados.

**Ejemplo malo** (inventado):
> Hoy estuve como 2 horas con los dedos, al final lo dejé en 87 grados y quedó bien, el usuario dijo que le gustó.

Por qué es mala: es diario, el número no sirve sin contexto, no dice qué estaba mal ni cómo se reconoce.

---

## 5. Cómo decido yo (Claude) — el criterio de fondo

Esto es lo que tengo en cuenta cuando reviso una animación o una propuesta de lecciones. Usalo para pensar igual.

1. **El usuario ve antes que el agente.** En el té hicieron falta ~8 vueltas y cada error lo notó el usuario primero.
   Entonces lo que más vale sumar a la skill no es "qué hice", sino **qué tendría que haber mirado para verlo antes que
   él** (ej.: "mirar de cerca la mano desde frente, 3/4 y arriba en cada pose clave, y compararla con su foto").
2. **Números antes que capturas, pero las capturas deciden lo que ve el público.** Un número correcto con una imagen
   rara significa que se midió lo equivocado. Siempre las dos cosas.
3. **"Se ve tieso" casi siempre es el cuerpo, no el brazo.** Antes de retocar el miembro, revisar que torso, cabeza y
   el otro brazo acompañen.
4. **Una lección vale si cambia lo que haría el próximo agente.** Si leerla no cambia ninguna decisión, no va.
5. **Corregir es mejor que agregar.** Una skill que dice una cosa y tres párrafos después la contraria es peor que una
   skill corta. Si algo resultó estar mal, se edita donde estaba.
6. **Lo que depende del gusto se marca como gusto.** Si el usuario eligió, se anota que eligió y qué, no se convierte en
   ley para todos los gestos.
7. **Desconfío de lo que "no se pudo comprobar" escondido en un "listo".** Si agy no tenía navegador o no pudo ver
   imágenes, la revisión está incompleta y tiene que decirlo.
8. **Seguridad y orden no se negocian por una animación**: puerto 3000, commits, ramas, datos y referencias ajenas.
   Si agy rompió una de esas reglas, se reporta aunque el resultado sea lindo.
9. **Lo chico y completo gana.** Un gesto bien integrado (CLIPS + GESTURES + disparo + respaldo) vale más que tres gestos
   lindos que NYRA nunca dispara. (Pasó: el té existía en el visor pero NYRA nunca lo activaba en vivo hasta que se
   agregó el disparo por texto.)

---

## 6. Plantilla de respuesta para la revisión

Pedile a ChatGPT que responda con este formato:

```
## Veredicto
Aprobar / Corregir antes de mostrar / Rehacer — en una línea, por qué.

## Lo que está bien
- …

## Problemas (ordenados por gravedad)
1. <qué> — <dónde se ve: cuadro y vista> — <cómo arreglarlo>
2. …

## Lo que no se puede saber con lo que hay
- … (y qué captura o número haría falta)

## Reglas del proyecto
- Rompió alguna: sí/no — cuál.

## Para la skill
### Sumar (texto listo para pegar, con la sección destino)
- [Sección] …
### Corregir (línea vieja → línea nueva)
- …
### No sumar (y por qué)
- …
```

---

## 7. Prompt para pasarle a agy con lo que hay que sumar

Una vez decidido qué entra, este es el mensaje para agy (completar lo que está entre `< >`):

```
Revisé tu animación de <gesto>. Antes de seguir:

Corregí en la animación:
1. <problema> — <cómo>.
2. …
Después hacé una grilla nueva (frente, perfil, arriba + cámara del directo) de la última versión y describí cada imagen
por escrito. No digas que está bien con capturas viejas.

En .claude/skills/animar-nyra/SKILL.md:
- Sumá en <sección>: "<texto>".
- Corregí en <sección> la línea "<vieja>" por "<nueva>" (no agregues otra que la contradiga).
- En "Lecciones", una entrada corta para <gesto> (<mes año>) con solo: <puntos>. Nada de diario de sesión.
- No sumes: <lo descartado> (<motivo>).

Reglas: trabajá en tu rama/worktree, sin commit ni push, sin tocar el puerto 3000 ni datos del usuario. Al terminar,
decí qué cambiaste, qué mediste y qué quedó sin comprobar.
```

---

## 8. Límites de esta revisión

- ChatGPT revisa **con lo que le pegan**: no puede correr Blender, el visor ni medir. Si falta evidencia, lo correcto es
  decir "no se puede saber" y pedirla, no aprobar por las dudas.
- Las imágenes estáticas no muestran el **ritmo ni la suavidad**: para eso hace falta ver el GIF o el bucle en el visor
  (o que el usuario lo describa).
- La aprobación final la da **el usuario** mirando el bucle en el navegador. La revisión de ChatGPT sirve para que llegue
  a esa instancia con menos vueltas, no para reemplazarla.
