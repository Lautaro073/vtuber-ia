# Junta los renders de un script de Blender (NNN_frente.png, NNN_perfil.png, NNN_arriba.png) en una sola imagen.
# Uso: python -X utf8 grilla.py <carpeta_renders> [salida.png] [lado_px]
# Una fila por cuadro, una columna por vista, con el número de cuadro y una vertical de referencia (sirve para ver
# si el torso se inclina).
import os
import sys

from PIL import Image, ImageDraw

folder = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(folder, "grilla.png")
side = int(sys.argv[3]) if len(sys.argv) > 3 else 230
views = [v for v in ("frente", "perfil", "arriba") if any(f.endswith(f"_{v}.png") for f in os.listdir(folder))]
frames = sorted({f.split("_")[0] for f in os.listdir(folder) if f.endswith(".png") and "_" in f and f.split("_")[0].isdigit()})
assert frames and views, f"no hay renders NNN_<vista>.png en {folder}"

sheet = Image.new("RGB", (side * len(views), side * len(frames)), "black")
draw = ImageDraw.Draw(sheet)
for r, frame in enumerate(frames):
    for c, view in enumerate(views):
        path = os.path.join(folder, f"{frame}_{view}.png")
        if not os.path.exists(path):
            continue
        sheet.paste(Image.open(path).convert("RGB").resize((side, side)), (c * side, r * side))
        x = c * side + side // 2
        draw.line([(x, r * side), (x, r * side + side)], fill=(0, 200, 200), width=1)
        draw.text((c * side + 6, r * side + 6), f"{frame} {view}", fill="yellow")
sheet.save(out)
print(out, sheet.size, "cuadros:", ", ".join(frames))
