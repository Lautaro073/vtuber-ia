# Correcciones localizadas de sprites de NYRA

## Intención

Corregir el parpadeo, tres visemas y tres emociones de Cecilia/NYRA sin introducir saltos al alternar sprites. El cuerpo, el pelo, la ropa, el contorno, el lienzo y todos los píxeles exteriores a ojos o boca deben conservarse exactamente.

## Alcance

- Crear ocho rasgos maestros localizados: ojos `half`, ojos `angry`, ojos `happy`, boca `o`, boca `u`, boca `fv`, boca sorprendida cerrada y boca sorprendida abierta.
- Aplicar los ojos `half` idénticos a los ocho archivos `cecilia_neutral_eye-half_mouth-*.png`.
- Aplicar cada visema `o`, `u` y `fv` idéntico a los estados `eye-open`, `eye-half` y `eye-closed`.
- Aplicar los mismos ojos corregidos a las dos variantes de boca de `angry` y `happy`.
- Cambiar solo la boca en las dos variantes `surprised`.
- Se modificarán 20 PNG únicos.

## Fuera de alcance

- No modificar `sad`, `thinking` ni las bocas `a`, `ei`, `l`, `mbp` o `closed`.
- No cambiar tamaño, posición, silueta, alfa, cuerpo, ropa, pelo, auriculares ni fondo.
- No alterar la lógica o el código de la aplicación VTuber.

## Enfoque elegido

Se usará una edición híbrida localizada. La referencia `revision-sprites/correcciones.png` y los sprites actuales guiarán la creación de los ocho rasgos maestros. Luego se aplicará cada maestro con máscaras estrictas sobre copias de los PNG originales. Todo píxel exterior a la máscara se conservará desde el archivo original, lo que evita deriva generativa y garantiza continuidad.

Las alternativas descartadas son editar cada sprite completo por separado, porque produciría deriva entre variantes, y redibujar automáticamente toda la cara, porque excedería las zonas autorizadas.

## Máscaras y propagación

- Ojos: dos máscaras separadas ajustadas a cada globo ocular y párpados, sin invadir pelo, cejas, nariz ni mejillas. Los maestros se copian como un único conjunto para que ambos ojos mantengan registro y para que las variantes de boca sean idénticas.
- Boca: máscara localizada alrededor de `(480, 247)`, protegiendo piel, nariz, mentón y lunar. Los visemas toman la paleta rosada y el tramado de `mouth-a` como referencia.
- Las máscaras finales se derivarán de los píxeles realmente modificados y quedarán contenidas dentro de la zona facial auditada `(390,145)-(595,290)`.

## Respaldo

Antes de reemplazar archivos, se copiarán los 20 originales con el mismo nombre a `public/avatar/_backup/2026-11-20_before_nyra_face_corrections/`. Los hashes del respaldo deberán coincidir con los hashes previos a la edición.

## Verificación

- Los 33 PNG siguen siendo 1024x1536 y conservan alfa binario 0/255.
- En cada archivo editado, todos los píxeles exteriores a su máscara son idénticos al respaldo.
- Los ocho `eye-half` comparten exactamente los mismos ojos.
- Cada boca `o`, `u` y `fv` es idéntica en `eye-open`, `eye-half` y `eye-closed`.
- `angry` y `happy` comparten ojos exactos entre sus variantes de boca.
- Los ojos de `surprised` permanecen idénticos a sus respaldos.
- Los 13 archivos excluidos conservan el mismo SHA-256.
- Se realizará una inspección visual ampliada de los rasgos y se entregará la lista exacta de archivos reemplazados.
