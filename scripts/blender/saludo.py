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
# Cuerpo: el torso compensa el peso del brazo levantado y el otro brazo cuelga por gravedad
spine, chest, upper_chest, neck = (P[f"J_Bip_C_{n}"] for n in ("Spine", "Chest", "UpperChest", "Neck"))
l_shoulder, l_up, l_low = (P[f"J_Bip_L_{n}"] for n in ("Shoulder", "UpperArm", "LowerArm"))
BODY_BONES = [spine, chest, upper_chest, neck, l_shoulder, l_up, l_low]
L_REST = (three(0.18, 1.07, -0.03), three(0.24, 0.79, -0.01))   # brazo izquierdo colgando (espejo del derecho)
UPPER_CHEST_REST = upper_chest.matrix.copy()                     # para mover los objetivos del brazo con el torso


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
AWAY = TO_HEAD               # inclinarse "lejos de la mano" lleva el torso hacia su izquierda (+x)


def pose_body(lean, twist, sway):
    """lean > 0: el torso se inclina hacia el lado contrario al brazo que saluda (compensa el peso), repartido
    entre las tres vértebras; sway: balanceo extra del pecho en cada vaivén; twist > 0: adelanta el hombro derecho.
    Devuelve cómo se movió el pecho, para que la mano siga al cuerpo."""
    for pb, share in ((spine, 0.35), (chest, 0.35), (upper_chest, 0.3)):
        amount = lean * share + (sway if pb is upper_chest else 0)
        if amount:
            rotate_toward(pb, Vector((0, 1, 0)), abs(amount), neck, AWAY * math.copysign(1, amount))
    if twist:
        rotate_toward(chest, Vector((0, 0, 1)), abs(twist), shoulder, Vector((0, -1, 0)) * math.copysign(1, twist))
    if lean:  # el cuello endereza la mitad para que la cabeza no quede caída
        rotate_toward(neck, Vector((0, 1, 0)), abs(lean) * 0.5, head, -AWAY * math.copysign(1, lean))
    return arm.matrix_world @ upper_chest.matrix @ UPPER_CHEST_REST.inverted() @ inv


def pose_arm(elbow, knuckle, palm=False, wrist=0.0, tilt=0.0, lift=0.0, curl=0.3, tips=0.0, spread=0.0,
             lean=0.0, twist=0.0, sway=0.0, l_sway=0.0):
    """wrist > 0 inclina la mano hacia la cabeza; tilt > 0 inclina la cabeza hacia la mano; lift sube el hombro.
    lean/twist/sway mueven el torso (ver pose_body); l_sway separa un poco el brazo izquierdo del cuerpo."""
    for pb in WAVE_BONES + FINGER_BONES + BODY_BONES:
        pb.matrix_basis = Matrix.Identity(4)
    refresh()
    moved = pose_body(lean, twist, sway)
    elbow, knuckle = moved @ elbow, moved @ knuckle  # la mano acompaña al torso
    aim(l_up, L_REST[0] + Vector((l_sway * 0.5, 0, 0)))  # el brazo izquierdo cuelga vertical aunque el torso se incline
    aim(l_low, L_REST[1] + Vector((l_sway, 0, 0)))
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


def key(frame, finger_lag=0, body_lag=0):
    """finger_lag / body_lag: los dedos y el cuerpo llegan unos cuadros después que la mano (siguen el movimiento).
    La cabeza llega un cuadro más tarde que el torso."""
    for pb in WAVE_BONES + FINGER_BONES + BODY_BONES:
        if pb in FINGER_BONES:
            lag = finger_lag
        elif pb is head:
            lag = body_lag + 1 if body_lag else 0
        elif pb in BODY_BONES:
            lag = body_lag
        else:
            lag = 0
        pb.rotation_mode = "QUATERNION"
        pb.keyframe_insert("rotation_quaternion", frame=frame + lag)


scene = bpy.context.scene
scene.render.fps = FPS
print("cabeza en Blender:", tuple(round(v, 3) for v in world(head)), "(debería ser ~ (0, 0.03, 1.39))")

# Tiempo (30 cuadros/s): amago (0,1 s), sube por adelante (0,4 s), saluda 3 veces ida y vuelta con un vaivén de 0,4 s,
# baja (0,5 s) y el cuerpo se pasa un poco del centro antes de acomodarse
UPPER = dict(palm=True, lift=0.14, lean=0.13, twist=0.08)
pose_arm(*REST); key(1)
pose_arm(*REST, lean=-0.035, l_sway=-0.008); key(4)                    # amago: se carga apenas hacia la mano antes de subirla
pose_arm(*VIA, lift=0.05, curl=0.2, lean=0.06, twist=0.04, l_sway=0.012); key(9)
pose_arm(ELBOW, wiper(0), curl=0.06, tilt=0.09, l_sway=0.025, **UPPER); key(16)
f = 16
for i in range(6):
    f += 6
    if i % 2 == 0:  # hacia la cabeza: la muñeca exagera el giro, los dedos van juntos y el pecho y la cabeza acompañan
        pose_arm(ELBOW, wiper(0.30), wrist=0.35, curl=0.12, tips=0.12, tilt=0.12, sway=0.028, l_sway=0.034, **UPPER)
    else:           # hacia afuera: la mano se endereza, los dedos se abren en abanico y el cuerpo vuelve
        pose_arm(ELBOW, wiper(-0.16), wrist=-0.05, curl=0.03, tips=-0.03, spread=0.12, tilt=0.065, sway=-0.016, l_sway=0.018,
                 **UPPER)  # en el GIF va menos para afuera que para adentro
    key(f, finger_lag=1, body_lag=1)
pose_arm(ELBOW, wiper(0), curl=0.06, tilt=0.09, l_sway=0.025, **UPPER); key(f + 5, body_lag=1)
pose_arm(*VIA, lift=0.05, curl=0.2, lean=0.06, twist=0.04, l_sway=0.012); key(f + 14, body_lag=1)
pose_arm(*REST, lean=-0.03, twist=-0.015, l_sway=-0.008); key(f + 21)   # se pasa un poco del centro...
pose_arm(*REST); key(f + 28)                                           # ...y se acomoda
scene.frame_start, scene.frame_end = 1, f + 28  # último cuadro clave

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
    target = Vector((0, -0.02, 1.15))  # todo el torso
    views = {"frente": Vector((0, -2.6, 1.2)), "perfil": Vector((-2.6, -0.02, 1.2)), "arriba": Vector((0, -0.4, 3.6))}
    for frame in (1, 4, 9, 16, 22, 28, scene.frame_end - 7, scene.frame_end):
        scene.frame_set(frame)
        for name, pos in views.items():
            cam.location = pos
            cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = str(RENDERS / f"{frame:03d}_{name}.png")
            bpy.ops.render.render(write_still=True)
    print("renders en", RENDERS)
