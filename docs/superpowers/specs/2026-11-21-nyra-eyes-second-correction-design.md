# Segunda corrección localizada de ojos de NYRA

## Intención

Corregir los ojos enojados y eliminar el destello claro del parpadeo cerrado sin alterar la continuidad del PNGTuber. La boca, el pelo, el cuerpo, el contorno, el alfa, el lienzo y todo píxel exterior a los ojos deben quedar idénticos al estado actual.

## Alcance

- Rehacer un único maestro de ojos enojados partiendo de `cecilia_neutral_eye-open_mouth-closed.png` y aplicarlo idéntico a las dos variantes `angry`.
- Mantener iris redondeados y visibles, una sola línea fina de párpado por ojo y una inclinación descendente hacia la nariz.
- Crear un único maestro de sombreado de ojos cerrados partiendo de `cecilia_neutral_eye-closed_mouth-closed.png`, usando el neutral abierto como referencia de color y tramado.
- Conservar intacta la línea y forma de pestañas cerradas; cambiar únicamente el sombreado de la piel alrededor.
- Aplicar ese maestro idéntico a los ocho archivos `cecilia_neutral_eye-closed_mouth-*.png`.
- Respaldar los diez originales en `public/avatar/_backup2/` antes de reemplazarlos.

## Restricciones

- Los únicos píxeles editables están dentro de dos máscaras oculares contenidas en `x=400..585, y=160..235`.
- Se conserva el alfa original de cada píxel y el lienzo `1024x1536`.
- Todo píxel fuera de las máscaras debe coincidir exactamente con `_backup2`.
- No se modifica ningún otro PNG.

## Enfoque

Se usarán dos ediciones visuales localizadas con imagegen: un maestro de enojo y un maestro de parpadeo sombreado. Después, un script determinista copiará únicamente el RGB dentro de las dos elipses oculares y conservará el alfa y todos los demás píxeles del archivo destino. Este enfoque evita deriva generativa en pelo, boca o cuerpo y garantiza que las variantes compartan exactamente los mismos ojos.

## Verificación

- Backup de exactamente diez PNG cuyos hashes coincidan con el estado previo.
- Diez objetivos modificados y veintitrés PNG no objetivo con sus hashes previos.
- Dimensión `1024x1536` y alfa binario `0/255` en los 33 sprites.
- Cero diferencias fuera de las máscaras oculares.
- Región ocular idéntica en los dos `angry` y en los ocho `eye-closed`.
- Revisión visual ampliada antes/después.
