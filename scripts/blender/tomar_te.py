# Arma "tomar el té" de NYRA con cuadros clave y lo exporta como .vrma (VRM Animation).
# Uso (Blender 5.3 con el complemento VRM):
#   blender --background --python scripts/blender/tomar_te.py -- public/avatar3d/NYRA.vrm public/avatar3d/tomar_te.vrma [carpeta_renders]
# La taza no está acá: el visor la pone entre el índice y el pulgar y la inclina sola (tilt de tomar_te en vrm.js,
# máxima entre 2,15 y 2,75 s). Por eso el tiempo de este script tiene que coincidir con ese tilt.
# Referencia: April (Parks and Rec) tomando té de frente: codo abajo y cerca del cuerpo, taza frente a la boca, la cabeza
# baja apenas a buscarla y los ojos miran a cámara por encima; al bajar la taza la cabeza vuelve y se acomoda.
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
    """Punto o dirección del visor (three.js: y arriba, mira a +Z) en coordenadas de Blender."""
    return Vector((x, -z, y))


# Objetivos (codo, muñeca, nudillo del medio), medidos en el visor con el té en código que ya agarraba bien la taza
REST = (three(-0.18, 1.07, -0.03), three(-0.216, 0.855, -0.025), three(-0.24, 0.79, -0.01))
# Agarre de las fotos de referencia: el asa siempre al costado derecho y la palma mirando a la taza (al centro), también
# al tomar: la taza no se cruza frente a la boca, va derecha y solo se inclina el borde hacia los labios. Esa
# inclinación la da el antebrazo subiendo hacia adelante desde el codo, que queda al costado del cuerpo.
VIA = (three(-0.18, 1.01, 0.0), three(-0.15, 1.02, 0.21), None)    # taza baja, frente a la panza
HOLD = (three(-0.17, 1.09, 0.0), three(-0.12, 1.12, 0.205), None)  # sostenida frente al pecho (antebrazo horizontal)
NEAR = (three(-0.24, 1.15, 0.06), three(-0.13, 1.26, 0.19), None)  # llegando a la boca, la mano en diagonal
SIP = (three(-0.25, 1.17, 0.03), three(-0.13, 1.30, 0.14), None)   # en los labios: mano al costado y adelante de la boca
SIP_DEEP = (three(-0.25, 1.14, 0.03), three(-0.12, 1.31, 0.13), None)  # sorbo largo
PALM_HOLD = three(1, 0, 0)  # palma a la taza: el asa queda al costado
# Al tomar, la mano va en diagonal (al costado y adelante de la boca): el asa queda al costado y un poco adelante y la
# taza derecha frente a la boca. Con el asa justo al costado el codo tendría que ir detrás del cuerpo; con la palma a la
# cara, el asa apuntaba a cámara y la taza se veía cruzada.
PALM_SIP = three(0.7, 0, -0.7).normalized()
# La taza va fija a la mano (vrm.js hace la misma cuenta), como en las fotos de referencia del agarre: su eje sigue la
# fila de nudillos (del meñique al índice) y el asa queda del lado de la palma. O sea: la taza está derecha solo si la
# mano está horizontal con los nudillos en fila vertical, y para inclinarla hay que girar la mano.
# Medidas de taza.glb ya escalada (x0,8): asa, alto del borde y radio del borde.
HANDLE_UP, HANDLE_OUT, RIM_H, RIM_R = 0.0384, 0.0528, 0.0448, 0.0344
LIP = three(0, 1.372, 0.07)  # labio inferior con la cabeza en reposo (medido con rayos sobre la cara)

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
ARM_BONES = [shoulder, up, low, hand, head]
FINGERS = {f: [P[f"J_Bip_R_{f}{j}"] for j in (1, 2, 3)] for f in ("Index", "Middle", "Ring", "Little")}
THUMB = [P[f"J_Bip_R_Thumb{j}"] for j in (1, 2, 3)]
FINGER_BONES = [pb for bones in FINGERS.values() for pb in bones] + THUMB
spine, chest, upper_chest, neck = (P[f"J_Bip_C_{n}"] for n in ("Spine", "Chest", "UpperChest", "Neck"))
l_shoulder, l_up, l_low = (P[f"J_Bip_L_{n}"] for n in ("Shoulder", "UpperArm", "LowerArm"))
BODY_BONES = [spine, chest, upper_chest, neck, l_shoulder, l_up, l_low]
L_REST = (three(0.18, 1.07, -0.03), three(0.24, 0.79, -0.01))
UPPER_CHEST_REST = upper_chest.matrix.copy()
FWD, BACK, TO_CENTER = Vector((0, -1, 0)), Vector((0, 1, 0)), Vector((1, 0, 0))  # Blender: el modelo mira a -Y


def refresh():
    bpy.context.view_layer.update()


def world(pb, tail=False):
    return arm.matrix_world @ (pb.tail if tail else pb.head)


def rotate_about(pb, q, pivot):
    pb.matrix = Matrix.Translation(pivot) @ q.to_matrix().to_4x4() @ Matrix.Translation(-pivot) @ pb.matrix
    refresh()


def aim(pb, target_world):
    a, b = pb.head.copy(), pb.tail.copy()
    rotate_about(pb, (b - a).normalized().rotation_difference((inv @ target_world - a).normalized()), a)


def bend(pb, angle, want, probe=None):
    """Gira pb para que la punta de probe (o la suya) se mueva hacia want (dirección en el mundo). Signo comprobado."""
    probe = probe or pb
    d = (probe.tail - pb.head).normalized()
    axis = d.cross(inv.to_3x3() @ want)
    if axis.length < 1e-6 or not angle:
        return
    before = world(probe, tail=True)
    rotate_about(pb, Quaternion(axis.normalized(), angle), pb.head)
    if (world(probe, tail=True) - before).dot(want) < 0:
        rotate_about(pb, Quaternion(axis.normalized(), -2 * angle), pb.head)


def palm_normal():
    h = world(hand)
    return (world(P["J_Bip_R_Index1"]) - h).cross(world(P["J_Bip_R_Little1"]) - h).normalized()


def palm_toward(n):
    """Gira el antebrazo entero sobre su eje (pronación) hasta que la palma mire hacia n. Se gira el antebrazo y no la
    mano: el modelo no tiene huesos de giro y torcer solo la mano retuerce la muñeca como un caramelo (se ve quebrada);
    el antebrazo queda dentro de la manga del buzo."""
    # cuenta directa: el ángulo entre la palma y n, proyectadas en el plano perpendicular al antebrazo (un solo giro;
    # probar 180 giros por pose hacía que Blender tardara más de 15 minutos)
    axis = (arm.matrix_world.to_3x3() @ (low.tail - low.head)).normalized()
    cur = palm_normal()
    a = (cur - axis * cur.dot(axis)).normalized()
    b = (n - axis * n.dot(axis)).normalized()
    angle = math.atan2(a.cross(b).dot(axis), a.dot(b))
    rotate_about(low, Quaternion((low.tail - low.head).normalized(), angle), low.head)
    if palm_normal().dot(n) < cur.dot(n) - 1e-6:  # el signo del giro en el espacio del esqueleto quedó al revés
        rotate_about(low, Quaternion((low.tail - low.head).normalized(), -2 * angle), low.head)


def grip(amount, pinky=0.0):
    """amount 0..1: de mano suelta a agarrar el asa como en las fotos de referencia: el índice pasa enganchado por el
    asa, el pulgar aprieta encima, el mayor la sostiene desde abajo y anular y meñique se doblan hacia la palma.
    pinky abre un poco el meñique (más vivo, no un puño)."""
    n = inv.to_3x3() @ palm_normal()
    # cerrados del todo, como en la foto: las puntas quedan escondidas contra la palma (antes, a medias, parecían una garra)
    full = {"Index": (1.0, 1.5, 0.9), "Middle": (1.4, 1.6, 0.9), "Ring": (1.45, 1.6, 0.9), "Little": (1.55 - pinky, 1.6, 0.95)}  # meñique cerrado como los demás (suelto se veía estirado)
    loose = (0.25, 0.25, 0.2)
    for finger, bones in FINGERS.items():
        # una sola bisagra por dedo, sacada del primer tramo: si cada falange usara su propia dirección, al cerrarse
        # ~90° el eje se daba vuelta y la falange siguiente doblaba para afuera (le pasaba al meñique)
        hinge = (bones[0].tail - bones[0].head).normalized().cross(n).normalized()
        for j, pb in enumerate(bones):
            rotate_about(pb, Quaternion(hinge, loose[j] + (full[finger][j] - loose[j]) * amount), pb.head)
    toward = (FINGERS["Index"][1].head - THUMB[0].head).normalized()  # el pulgar va hacia la 2.ª falange del índice
    d = (THUMB[0].tail - THUMB[0].head).normalized()
    rotate_about(THUMB[0], d.rotation_difference(d.lerp(toward, 0.35 + 0.4 * amount).normalized()), THUMB[0].head)
    for pb in THUMB[1:]:
        d = (pb.tail - pb.head).normalized()
        rotate_about(pb, Quaternion(d.cross(n).normalized(), 0.15 + 0.35 * amount), pb.head)


def signed(pb, value, direction):
    """bend con signo: value > 0 mueve la punta hacia direction, value < 0 hacia el lado contrario."""
    bend(pb, abs(value), direction if value > 0 else -direction)


def pose_body(fwd=0.0, twist=0.0, lean=0.0):
    """fwd > 0: el torso se inclina apenas hacia adelante (va a buscar la taza); twist > 0 adelanta el hombro derecho;
    lean > 0 compensa el peso hacia la izquierda. Devuelve cómo se movió el pecho para que la mano lo siga."""
    for pb, share in ((spine, 0.4), (chest, 0.35), (upper_chest, 0.25)):
        if fwd:
            bend(pb, abs(fwd) * share, FWD * math.copysign(1, fwd), probe=neck)
        if lean:
            bend(pb, abs(lean) * share, TO_CENTER * math.copysign(1, lean), probe=neck)
    if twist:
        before = world(shoulder)
        rotate_about(chest, Quaternion((0, 0, 1), twist), chest.head)
        if (world(shoulder) - before).dot(FWD) < 0:
            rotate_about(chest, Quaternion((0, 0, 1), -2 * twist), chest.head)
    if fwd:  # el cuello compensa la mitad: la cabeza no se va adelante con el torso
        bend(neck, abs(fwd) * 0.5, BACK * math.copysign(1, fwd), probe=head)
    return arm.matrix_world @ upper_chest.matrix @ UPPER_CHEST_REST.inverted() @ inv


def pose(targets, shift=None, roll=0.0, palm=None, grip_k=0.0, wrist=0.0, nod=0.0, tilt=0.0, lift=0.0, fwd=0.0, twist=0.0, lean=0.0,
         l_fwd=0.0, pinky=0.0):
    """targets: (codo, muñeca, nudillo). wrist > 0 dobla la mano hacia la cara (la taza se inclina hacia la boca);
    nod > 0 baja la cabeza (negativo: la levanta); tilt > 0 la inclina hacia la mano; l_fwd adelanta el brazo izquierdo."""
    for pb in ARM_BONES + FINGER_BONES + BODY_BONES:
        pb.matrix_basis = Matrix.Identity(4)
    refresh()
    moved = pose_body(fwd, twist, lean)
    elbow, wrist_t = moved @ targets[0], moved @ targets[1]
    if shift is not None:  # corrimiento que busca solve() para que el borde llegue a la boca
        elbow, wrist_t = elbow + shift * 0.5, wrist_t + shift
    aim(l_up, L_REST[0] + FWD * (l_fwd * 0.3))
    aim(l_low, L_REST[1] + FWD * l_fwd + Vector((0, 0, l_fwd * 0.4)))  # se dobla apenas, no queda tieso
    if lift:
        signed(shoulder, lift, Vector((0, 0, 1)))
    aim(up, elbow)
    aim(low, wrist_t)
    aim(hand, world(hand) + (world(hand) - world(low)))  # la mano sigue al antebrazo; la muñeca solo se dobla un poco (wrist)
    if palm is not None:
        palm_toward(palm)
    if roll:  # gira el antebrazo (pronación): así se inclina la taza hacia la boca, con la muñeca derecha
        rotate_about(low, Quaternion((low.tail - low.head).normalized(), roll), low.head)
    if wrist:  # gira la mano sobre el eje de la palma: inclina la taza hacia la cara o hacia afuera, sin volcarla de costado
        rotate_about(hand, Quaternion((inv.to_3x3() @ palm_normal()).normalized(), wrist), hand.head)
    if nod:
        bend(head, abs(nod), (FWD if nod > 0 else BACK) + Vector((0, 0, -0.3 if nod > 0 else 0.3)))
    if tilt:
        signed(head, tilt, -TO_CENTER)
    grip(grip_k, pinky)


def key(frame, finger_lag=0, body_lag=0):
    """Los dedos y el cuerpo llegan unos cuadros después que el brazo; la cabeza uno más que el torso."""
    for pb in ARM_BONES + FINGER_BONES + BODY_BONES:
        lag = finger_lag if pb in FINGER_BONES else (body_lag + 1 if body_lag else 0) if pb is head else body_lag if pb in BODY_BONES else 0
        pb.rotation_mode = "QUATERNION"
        q = pb.rotation_quaternion
        if pb.name in LAST and q.dot(LAST[pb.name]) < 0:  # mismo giro con el signo opuesto: si no, interpola dando la vuelta
            q.negate()
        LAST[pb.name] = q.copy()
        pb.keyframe_insert("rotation_quaternion", frame=frame + lag)


LAST = {}


def cup():
    """Dónde queda la taza con la pose actual (misma cuenta que vrm.js): eje, centro del borde y punto del borde que
    queda más cerca de la cara."""
    n = palm_normal()
    up = world(P["J_Bip_R_Index1"]) - world(P["J_Bip_R_Little1"])
    up = (up - n * up.dot(n)).normalized()
    hd = -n  # el asa, del lado de la palma
    pinch = (world(FINGERS["Index"][1]) + world(FINGERS["Index"][2])) / 2
    rim = pinch - up * HANDLE_UP - hd * HANDLE_OUT + up * RIM_H
    x = up.cross(-hd)
    ring = [rim + (x * math.cos(t) - hd * math.sin(t)) * RIM_R for t in (i * math.pi / 12 for i in range(24))]
    return up, rim, max(ring, key=lambda q: q.y)  # +Y en Blender = hacia el cuerpo


def lip():
    """El labio con la cabeza como está ahora (si la cabeza baja, la boca baja con ella)."""
    return arm.matrix_world @ head.matrix @ head.bone.matrix_local.inverted() @ inv @ LIP


def tilt_deg():
    up = cup()[0]
    return math.degrees(math.atan2(up.y, up.z))  # > 0: la boca de la taza se inclina hacia la cara


def side_deg():
    up = cup()[0]
    return math.degrees(math.atan2(up.x, up.z))  # vuelco hacia el costado: tiene que quedar en 0


WRIST_MAX = 0.35  # lo que de verdad se dobla la muñeca hacia los costados (~20°)


def solve(targets, incl, ahead=0.0, below=0.0, **kw):
    """Mueve el brazo (no la taza) hasta que el borde del lado de la cara quede en el labio (ahead/below: cuánto antes
    y más abajo). La inclinación hacia la boca (incl, grados) la da el giro del antebrazo y la muñeca se dobla un poco
    para que la taza no se vuelque hacia el costado. Se corrige con el eje de la taza (x, y) y no con ángulos, que
    saltan cuando la taza da la vuelta."""
    want = Vector((0, math.sin(math.radians(incl)), math.cos(math.radians(incl))))  # eje pedido (+Y: hacia la cara)
    shift, wrist = Vector(), 0.0

    def measure(r, w):
        pose(targets, shift=shift, roll=r, wrist=w, **kw)
        up = cup()[0]
        return up.x, up.y - want.y, up

    # giro inicial: búsqueda gruesa (de 20° en 20°) del que deja el eje más cerca del pedido
    roll = min((math.radians(d) for d in range(-160, 181, 20)), key=lambda r: -measure(r, 0.0)[2].dot(want))
    clamp = lambda v, m: max(-m, min(m, v))
    for i in range(8):
        x0, y0, _ = measure(roll, wrist)
        if ahead is not None:
            step = (lip() + FWD * ahead - Vector((0, 0, below))) - cup()[2]
            shift += step * min(1, 0.05 / max(step.length, 1e-6))  # pasos cortos: si no, se pasa y diverge
        xr, yr, _ = measure(roll + 0.05, wrist)
        xw, yw, _ = measure(roll, wrist + 0.05)
        a, b, c, d = (xr - x0) / 0.05, (xw - x0) / 0.05, (yr - y0) / 0.05, (yw - y0) / 0.05
        det = a * d - b * c or 1e-6
        roll += clamp((-x0 * d + y0 * b) / det, 0.3)
        wrist = clamp(wrist + clamp((-y0 * a + x0 * c) / det, 0.15), WRIST_MAX)
    measure(roll, wrist)
    t, sd = tilt_deg(), side_deg()
    err = 0 if ahead is None else ((lip() + FWD * ahead - Vector((0, 0, below))) - cup()[2]).length
    print(f"  inclinación {t:5.1f}° (pedida {incl}), vuelco {sd:5.1f}°, borde a {err * 100:.1f} cm, "
          f"antebrazo {math.degrees(roll):.0f}°, muñeca {math.degrees(wrist):.0f}°")
    return shift, roll


scene = bpy.context.scene
scene.render.fps = FPS

# Tiempo (30 cuadros/s). La taza se inclina con la mano: al sostenerla queda derecha (0°) y en el sorbo el borde del
# lado de la cara toca el labio con 25° a 50° de inclinación.
HELD = dict(palm=PALM_HOLD, grip_k=1.0, l_fwd=0.03)
LIPS = dict(palm=PALM_SIP, grip_k=1.0, l_fwd=0.03)
pose(REST); key(1)
pose(REST, lift=-0.04, fwd=-0.01, grip_k=0.2); key(5)                                     # amago: el hombro baja antes de subir
solve(VIA, 0, ahead=None, lift=0.02, twist=0.03, palm=PALM_HOLD, grip_k=0.8, l_fwd=0.015); key(14, finger_lag=1)
solve(HOLD, -6, ahead=None, lift=0.04, twist=0.06, fwd=0.01, **HELD); key(26, finger_lag=1, body_lag=1)  # llega un poco pasada...
solve(HOLD, 0, ahead=None, lift=0.03, twist=0.05, **HELD); key(33, finger_lag=1, body_lag=1)  # ...y se acomoda
solve(HOLD, 4, ahead=None, lift=0.03, twist=0.05, nod=0.03, **HELD); key(42, body_lag=1)    # mira la taza
print("acercando:"); solve(NEAR, 12, ahead=0.03, below=0.015, lift=0.06, twist=0.08, fwd=0.02, nod=0.04, **LIPS); key(50, body_lag=1)
print("sorbo:"); solve(SIP, 28, lift=0.07, twist=0.08, fwd=0.03, nod=0.05, tilt=0.02, **LIPS); key(58, finger_lag=1, body_lag=1)
print("sorbo largo:"); solve(SIP_DEEP, 48, lift=0.08, twist=0.08, fwd=0.02, nod=-0.03, tilt=0.03, **LIPS)
key(72, finger_lag=1, body_lag=1)                                     # inclina más y la cabeza va apenas para atrás
print("termina:"); solve(SIP, 30, lift=0.08, twist=0.08, fwd=0.02, nod=0.02, tilt=0.03, **LIPS); key(84, body_lag=1)
print("separa:"); solve(NEAR, 8, ahead=0.035, below=0.02, lift=0.05, twist=0.07, fwd=0.01, nod=-0.02, **LIPS); key(94, body_lag=1)
solve(HOLD, 0, ahead=None, lift=0.03, twist=0.05, fwd=-0.02, nod=-0.07, tilt=-0.03, **HELD); key(103, finger_lag=1, body_lag=1)  # "ahh"
solve(HOLD, 0, ahead=None, lift=0.03, twist=0.05, **HELD); key(112, body_lag=1)
solve(VIA, 0, ahead=None, lift=0.01, twist=0.02, palm=PALM_HOLD, grip_k=0.8, l_fwd=0.01); key(122, finger_lag=1, body_lag=1)
pose(REST, lift=-0.02, fwd=-0.008, grip_k=0.15); key(131)                                  # se pasa apenas al bajar...
pose(REST); key(138)                                                                         # ...y se acomoda (4,6 s)
scene.frame_start, scene.frame_end = 1, 138

OUT.parent.mkdir(parents=True, exist_ok=True)
print("export:", bpy.ops.export_scene.vrma(filepath=str(OUT), armature_object_name=arm.name))

if RENDERS:
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
    target = Vector((0, -0.02, 1.2))
    views = {"frente": Vector((0, -2.2, 1.25)), "perfil": Vector((-2.2, -0.02, 1.25)), "arriba": Vector((0, -0.4, 3.4))}
    for frame in (1, 14, 33, 50, 72, 103, 122, 138):
        scene.frame_set(frame)
        for name, pos in views.items():
            cam.location = pos
            cam.rotation_euler = (target - pos).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = str(RENDERS / f"{frame:03d}_{name}.png")
            bpy.ops.render.render(write_still=True)
    print("renders en", RENDERS)
