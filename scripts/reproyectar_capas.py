"""Copia los píxeles del sprite original sobre las capas de Cecilia en todo lo que se VE con la pose neutral,
así el armado neutral queda idéntico al original. Lo tapado (lo que GPT completó) no se toca.
Los huecos (silueta del original que ninguna capa cubre) van a la capa más cercana.
Uso: python reproyectar_capas.py <carpeta_capas> <carpeta_salida>"""
import shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

src, out = Path(sys.argv[1]), Path(sys.argv[2])
orig = np.array(Image.open(Path(__file__).parent.parent / 'public' / 'avatar' / 'cecilia_neutral_eye-open_mouth-closed.png').convert('RGBA'))
# Orden de atrás hacia adelante del armado neutral (las variantes de ojos, bocas y expresiones se copian tal cual)
STACK = ['01_capucha_atras', '02_pelo_atras', '03_cuerpo', '04_brazo_izq', '05_brazo_der', '06_cuello', '07_cara_base',
         '08_rubor', '09_ojo_izq_blanco', '10_ojo_der_blanco', '11_iris_izq', '12_iris_der', '17_cejas_neutral',
         '23_boca_closed', '31_mechones_laterales_izq', '32_mechones_laterales_der', '33_flequillo',
         '34_cordones_capucha', '35_auriculares']
# Piezas chicas que ya son recortes exactos: no reciben huecos de alrededor
NO_FILL = {'08_rubor', '09_ojo_izq_blanco', '10_ojo_der_blanco', '11_iris_izq', '12_iris_der', '17_cejas_neutral', '23_boca_closed'}

out.mkdir(parents=True, exist_ok=True)
for f in src.glob('*.png'):
    if f.stem not in STACK: shutil.copy2(f, out / f.name)

layers = [np.array(Image.open(src / f'{n}.png').convert('RGBA')) for n in STACK]
top = np.full(orig.shape[:2], -1)
for i, a in enumerate(layers): top[a[..., 3] > 128] = i
sil = orig[..., 3] > 128

# Huecos: a la capa más cercana que pueda recibirlos
can = np.isin(top, [i for i, n in enumerate(STACK) if n not in NO_FILL])
_, (iy, ix) = ndimage.distance_transform_edt(~can, return_indices=True)
holes = sil & (top < 0)
top[holes] = top[iy[holes], ix[holes]]

for i, (n, a) in enumerate(zip(STACK, layers)):
    vis = sil & (top == i)
    a[vis, :3] = orig[vis, :3]
    a[vis, 3] = 255
    Image.fromarray(a).save(out / f'{n}.png')

# Chequeo: el armado neutral tiene que dar el original
c = Image.new('RGBA', orig.shape[1::-1], (0, 0, 0, 0))
for n in STACK: c.alpha_composite(Image.open(out / f'{n}.png'))
c = np.array(c).astype(int)
same = (np.abs(c[..., :3] - orig[..., :3].astype(int)).max(-1)[sil] < 12).mean()
print(f'huecos rellenados: {holes.sum()} px · armado neutral igual al original: {same * 100:.2f}%')
assert same > 0.99
