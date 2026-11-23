# Arma el saludo de NYRA con cuadros clave y lo exporta como .vrma (VRM Animation).
# Uso (Blender 5.3 con el complemento VRM instalado):
#   blender --background --python scripts/blender/saludo.py -- public/avatar3d/NYRA.vrm public/avatar3d/saludo.vrma [carpeta_renders]
# Las posiciones vienen del visor (three.js: y arriba, el modelo mira a +Z) y se pasan a Blender (z arriba, mira a -Y).
# Cada hueso se apunta a su objetivo en vez de posarlo a ojo: así el codo queda abajo y adelante y la mano al lado de la cara.
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

args = sys.argv[sys.argv.index("--") + 1:]
VRM, OUT = Path(args[0]).resolve(), Path(args[1]).resolve()
RENDERS = Path(args[2]).resolve() if len(args) > 2 else None
FPS = 30


def three(x, y, z):
    """Punto del visor (three.js) en coordenadas de Blender."""
    return Vector((x, -z, y))


# Objetivos en coordenadas del visor: (codo, nudillo del medio). Ver public/avatar3d/vrm.js
# Referencia: GIF de alguien saludando a cámara. Codo casi a la altura del hombro, mano arriba a la altura de la frente
# y un poco separada de la cabeza, y el antebrazo que se balancea desde el codo como un limpiaparabrisas
REST = (three(-0.18, 1.07, -0.03), three(-0.24, 0.79, -0.01))   # brazo colgando
VIA = (three(-0.25, 1.10, 0.10), three(-0.27, 1.30, 0.22))      # codo doblado, mano frente al pecho: sube y baja por adelante
ELBOW = three(-0.30, 1.24, 0.03)                                 # codo durante el saludo
FOREARM = 0.28                                                   # codo → nudillos (medido)


def wiper(angle):
    """Nudillos girados desde el codo en el plano de frente: angle > 0 hacia la cabeza."""
    return ELBOW + Vector((FOREARM * math.sin(angle), -0.02, FOREARM * math.cos(angle)))


for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.import_scene.vrm(filepath=str(VRM))
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
P = arm.pose.bones
inv = arm.matrix_world.inverted()
up, low, hand = P["J_Bip_R_UpperArm"], P["J_Bip_R_LowerArm"], P["J_Bip_R_Hand"]
shoulder, head = P["J_Bip_R_Shoulder"], P["J_Bip_C_Head"]
WAVE_BONES = [shoulder, up, low, hand, head]
# Dedos: cada uno con sus 3 falanges; el índice más recto y el meñique más curvo, como una mano relajada
FINGERS = {f: [P[f"J_Bip_R_{f}{j}"] for j in (1, 2, 3)] for f in ("Index", "Middle", "Ring", "Little")}
CASCADE = {"Index": 0.8, "Middle": 1.0, "Ring": 1.15, "Little": 1.3}
THUMB = [P[f"J_Bip_R_Thumb{j}"] for j in (1, 2, 3)]
FINGER_BONES = [pb for bones in FINGERS.values() for pb in bones] + THUMB


def refresh():
    bpy.context.view_layer.update()


def world(pb, tail=False):
    return arm.matrix_world @ (pb.tail if tail else pb.head)


def rotate_about(pb, q, pivot):
    """Gira el hueso (en el espacio del esqueleto) alrededor de pivot."""
    pb.matrix = Matrix.Translation(pivot) @ q.to_matrix().to_4x4() @ Matrix.Translation(-pivot) @ pb.matrix
    refresh()


def aim(pb, target_world):
    a, b = pb.head.copy(), pb.tail.copy()
    rotate_about(pb, (b - a).normalized().rotation_difference((inv @ target_world - a).normalized()), a)


def palm_normal():
    h = world(hand)
    return (world(P["J_Bip_R_Index1"]) - h).cross(world(P["J_Bip_R_Little1"]) - h).normalized()


def palm_forward():
    """Gira la mano sobre su eje hasta que la palma mire a cámara (-Y en Blender)."""
    axis, start, pivot = (hand.tail - hand.head).normalized(), hand.matrix.copy(), hand.head.copy()

    def twist(deg):
        hand.matrix = Matrix.Translation(pivot) @ Quaternion(axis, math.radians(deg)).to_matrix().to_4x4() @ Matrix.Translation(-pivot) @ start
        refresh()
        return palm_normal().y  # más negativo = palma más hacia cámara

    twist(min(range(0, 360, 3), key=twist))


def rotate_toward(pb, axis, angle, probe, want):
    """Gira pb sobre axis y comprueba que la punta de probe se movió hacia want; si no, gira al revés."""
    before = world(probe, tail=True)
    rotate_about(pb, Quaternion(axis, angle), pb.head)
    if (world(probe, tail=True) - before).dot(want) < 0:
        rotate_about(pb, Quaternion(axis, -2 * angle), pb.head)


def fingers(curl, tips, spread):
    """curl: cierra todas las falanges; tips: solo las dos de la punta; spread: abre los dedos en abanico."""
    n = inv.to_3x3() @ palm_normal()  # hacia donde mira la palma
    mid = FINGERS["Middle"][0]
    for finger, bones in FINGERS.items():
        if spread and finger != "Middle":  # se separa del dedo del medio
            rotate_toward(bones[0], n, spread * (1.6 if finger == "Little" else 1.0), bones[0], bones[0].head - mid.head)
        for j, pb in enumerate(bones):
            d = (pb.tail - pb.head).normalized()
            angle = curl * CASCADE[finger] + (tips * (0.8 if j == 1 else 1.2) if j else 0)
            rotate_about(pb, Quaternion(d.cross(n).normalized(), angle), pb.head)  # gira la punta hacia la palma
    # Pulgar: en pose T sale en ángulo recto; se acerca al índice y se curva apenas, como en una mano suelta
    toward = (FINGERS["Index"][0].tail - THUMB[0].head).normalized()
    d = (THUMB[0].tail - THUMB[0].head).normalized()
    rotate_about(THUMB[0], d.rotation_difference(d.lerp(toward, 0.45 - spread).normalized()), THUMB[0].head)
    for pb in THUMB[1:]:
        d = (pb.tail - pb.head).normalized()
        rotate_about(pb, Quaternion(d.cross(n).normalized(), 0.2 + curl), pb.head)


TO_HEAD = Vector((1, 0, 0))  # la mano derecha está en x < 0: hacia la cabeza es +x


def pose_arm(elbow, knuckle, palm=False, wrist=0.0, tilt=0.0, lift=0.0, curl=0.3, tips=0.0, spread=0.0):
    """wrist > 0 inclina la mano hacia la cabeza; tilt > 0 inclina la cabeza hacia la mano; lift sube el hombro."""
    for pb in WAVE_BONES + FINGER_BONES:
        pb.matrix_basis = Matrix.Identity(4)
    refresh()
    if lift:
        rotate_toward(shoulder, Vector((0, 1, 0)), lift, shoulder, Vector((0, 0, 1)))
    aim(up, elbow)
    aim(low, knuckle)
    aim(hand, knuckle + (knuckle - world(low)).normalized() * 0.05)
    if palm:
        palm_forward()
    if wrist:
        rotate_toward(hand, inv.to_3x3() @ palm_normal(), abs(wrist), hand, TO_HEAD * math.copysign(1, wrist))
    if tilt:
        rotate_toward(head, Vector((0, 1, 0)), tilt, head, -TO_HEAD)
    fingers(curl, tips, spread)


def key(frame, finger_lag=0):
    """finger_lag: los dedos llegan unos cuadros después que la mano (las puntas siguen el movimiento)."""
    for pb in WAVE_BONES + FINGER_BONES:
        pb.rotation_mode = "QUATERNION"
        pb.keyframe_insert("rotation_quaternion", frame=frame + (finger_lag if pb in FINGER_BONES else 0))


scene = bpy.context.scene
scene.render.fps = FPS
print("cabeza en Blender:", tuple(round(v, 3) for v in world(head)), "(debería ser ~ (0, 0.03, 1.39))")

# Tiempo (30 cuadros/s): sube por adelante (0,5 s), saluda 3 veces ida y vuelta con un vaivén de 0,4 s, baja (0,6 s)
UPPER = dict(palm=True, tilt=0.09, lift=0.14)
pose_arm(*REST); key(1)
pose_arm(*VIA, lift=0.05, curl=0.2); key(8)
pose_arm(ELBOW, wiper(0), curl=0.06, **UPPER); key(15)
f = 15
for i in range(6):
    f += 6
    if i % 2 == 0:  # hacia la cabeza: la muñeca exagera el giro y los dedos van juntos, apenas curvados
        pose_arm(ELBOW, wiper(0.30), wrist=0.35, curl=0.12, tips=0.12, **UPPER)
    else:           # hacia afuera: la mano se endereza y los dedos se abren en abanico
        pose_arm(ELBOW, wiper(-0.16), wrist=-0.05, curl=0.03, tips=-0.03, spread=0.12, **UPPER)  # en el GIF va menos para afuera que para adentro
    key(f, finger_lag=1)
pose_arm(ELBOW, wiper(0), curl=0.06, **UPPER); key(f + 5)
pose_arm(*VIA, lift=0.05, curl=0.2); key(f + 14)
pose_arm(*REST); key(f + 22)
scene.frame_start, scene.frame_end = 1, f + 22

OUT.parent.mkdir(parents=True, exist_ok=True)
print("export:", bpy.ops.export_scene.vrma(filepath=str(OUT), armature_object_name=arm.name))

if RENDERS:  # cuadros de control desde tres puntos de vista
    RENDERS.mkdir(parents=True, exist_ok=True)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.lens = 50
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.rotation_euler = (0.8, 0.2, 0.3)
    scene.collection.objects.link(sun)
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = 360, 360
    scene.render.film_transparent = False
    target = Vector((-0.15, -0.02, 1.35))  # brazo y cabeza
    views = {"frente": Vector((-0.15, -1.3, 1.38)), "perfil": Vector((-1.4, -0.02, 1.38)), "arriba": Vector((-0.15, -0.3, 2.6))}
    for frame in (8, 15, 21, 27, 33, scene.frame_end - 9):
        scene.frame_set(frame)
        for name, pos in views.items():
            cam.location = pos
            cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = str(RENDERS / f"{frame:03d}_{name}.png")
            bpy.ops.render.render(write_still=True)
    print("renders en", RENDERS)
