"""Animación del gesto meme "six seven" (6-7) para NYRA en Blender -> six_seven.vrma.

Referencia: Fortnite "Six Seven" emote (Rumi):
  - Codos bajos cerca de los costados, antebrazos adelante-afuera alternando arriba/abajo
    como una balanza rítmica (~7 cuadros por cambio a 30 fps).
  - Palmas hacia arriba-cámara (supinación desde el antebrazo, nunca retorciendo la muñeca),
    dedos curvados en cuenco suave con cascada y retraso en las puntas.
  - Todo el cuerpo acompaña: balanceo de hombros, pecho y ladeos de cabeza con actitud.

Uso:
  blender --background --python scripts/blender/six_seven.py -- \
    public/avatar3d/NYRA.vrm public/avatar3d/six_seven.vrma [carpeta_renders]
"""
import math
import os
import sys

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


FPS = 30
FRAME_END = 80

REST_EULER = {
    "J_Bip_L_UpperArm": (0.0, 0.0, -1.25),
    "J_Bip_R_UpperArm": (0.0, 0.0, 1.25),
    "J_Bip_L_LowerArm": (0.0, 0.0, -0.15),
    "J_Bip_R_LowerArm": (0.0, 0.0, 0.15),
}

FWD = Vector((0.0, -1.0, 0.0))
UP = Vector((0.0, 0.0, 1.0))
LEFT = Vector((1.0, 0.0, 0.0))
RIGHT = Vector((-1.0, 0.0, 0.0))


def clear_scene():
    """Limpia los objetos iniciales sin resetear addons habilitados."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.actions):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def refresh():
    bpy.context.view_layer.update()


def world_pos(arm, pb, tail=False):
    return arm.matrix_world @ (pb.tail if tail else pb.head)


def rotate_about(arm, pb, q, pivot_arm):
    pb.matrix = Matrix.Translation(pivot_arm) @ q.to_matrix().to_4x4() @ Matrix.Translation(-pivot_arm) @ pb.matrix
    refresh()


def aim_bone(arm, pb, target_world):
    a, b = pb.head.copy(), pb.tail.copy()
    inv = arm.matrix_world.inverted()
    d_cur = (b - a).normalized()
    d_want = (inv @ Vector(target_world) - a).normalized()
    if d_cur.length < 1e-6 or d_want.length < 1e-6:
        return
    rotate_about(arm, pb, d_cur.rotation_difference(d_want), a)


def bend_signed(arm, pb, angle, direction_world, probe=None):
    """Gira pb para mover la punta de probe hacia direction_world con signo comprobado."""
    if abs(angle) < 1e-6:
        return
    probe = probe or pb
    inv3 = arm.matrix_world.inverted().to_3x3()
    d = (probe.tail - pb.head).normalized()
    want = direction_world if angle > 0 else -direction_world
    axis = d.cross(inv3 @ want)
    if axis.length < 1e-6:
        return
    mag = abs(angle)
    before = world_pos(arm, probe, tail=True)
    rotate_about(arm, pb, Quaternion(axis.normalized(), mag), pb.head.copy())
    if (world_pos(arm, probe, tail=True) - before).dot(want) < 0:
        rotate_about(arm, pb, Quaternion(axis.normalized(), -2.0 * mag), pb.head.copy())


def palm_normal(arm, side):
    """Normal saliendo de la palma en mundo (verificada para R y L)."""
    P = arm.pose.bones
    h = world_pos(arm, P[f"J_Bip_{side}_Hand"])
    idx = world_pos(arm, P[f"J_Bip_{side}_Index1"])
    lit = world_pos(arm, P[f"J_Bip_{side}_Little1"])
    if side == "R":
        return (idx - h).cross(lit - h).normalized()
    return (lit - h).cross(idx - h).normalized()


def palm_toward(arm, side, want_world):
    """Supina/prona girando el antebrazo (nunca retuerce la muñeca)."""
    P = arm.pose.bones
    low = P[f"J_Bip_{side}_LowerArm"]
    axis_w = (arm.matrix_world.to_3x3() @ (low.tail - low.head)).normalized()
    axis_l = (low.tail - low.head).normalized()
    cur = palm_normal(arm, side)
    want = Vector(want_world).normalized()
    a = cur - axis_w * cur.dot(axis_w)
    b = want - axis_w * want.dot(axis_w)
    if a.length < 1e-5 or b.length < 1e-5:
        return
    a.normalize()
    b.normalize()
    ang = math.atan2(a.cross(b).dot(axis_w), max(-1.0, min(1.0, a.dot(b))))
    rotate_about(arm, low, Quaternion(axis_l, ang), low.head.copy())
    if palm_normal(arm, side).dot(want) < cur.dot(want) - 1e-5:
        rotate_about(arm, low, Quaternion(axis_l, -2.0 * ang), low.head.copy())


def pose_fingers(arm, side, curl, thumb_curl=0.22, thumb_open=0.14, splay=0.0):
    """Cierra los dedos con una bisagra única por dedo (eje sacado del primer tramo)."""
    P = arm.pose.bones
    n_world = palm_normal(arm, side)
    n_arm = (arm.matrix_world.inverted().to_3x3() @ n_world).normalized()
    cascade = {"Index": 0.62, "Middle": 0.84, "Ring": 1.08, "Little": 1.32}
    splay_mul = {"Index": 1.0, "Middle": 0.25, "Ring": -0.45, "Little": -1.10}

    for fname, k in cascade.items():
        b1, b2, b3 = (P[f"J_Bip_{side}_{fname}{j}"] for j in (1, 2, 3))
        d1 = (b1.tail - b1.head).normalized()
        hinge = d1.cross(n_arm).normalized()
        # Splay suave en el nudillo (alrededor de la normal de la palma)
        sp = splay * splay_mul[fname]
        if abs(sp) > 1e-5:
            before_sp = world_pos(arm, b3, tail=True)
            rotate_about(arm, b1, Quaternion(n_arm, sp), b1.head.copy())
            # Signo comprobado: splay > 0 separa el índice hacia el pulgar (y meñique al lado opuesto)
            thumb_dir = (world_pos(arm, P[f"J_Bip_{side}_Thumb1"]) - world_pos(arm, P[f"J_Bip_{side}_Little1"])).normalized()
            want_sp = thumb_dir if splay_mul[fname] * sp > 0 else -thumb_dir
            if (world_pos(arm, b3, tail=True) - before_sp).dot(want_sp) < 0:
                rotate_about(arm, b1, Quaternion(n_arm, -2.0 * sp), b1.head.copy())

        c = curl * k
        for pb, ck in ((b1, c), (b2, c * 0.92), (b3, c * 0.68)):
            if abs(ck) < 1e-5:
                continue
            before = world_pos(arm, pb, tail=True)
            rotate_about(arm, pb, Quaternion(hinge, ck), pb.head.copy())
            if (world_pos(arm, pb, tail=True) - before).dot(n_world) < 0:
                rotate_about(arm, pb, Quaternion(hinge, -2.0 * ck), pb.head.copy())

    # Pulgar relajado: se acerca naturalmente al costado del índice (como en tomar_te.py), nunca en pose T
    th1, th2, th3 = (P[f"J_Bip_{side}_Thumb{j}"] for j in (1, 2, 3))
    idx2_pos = world_pos(arm, P[f"J_Bip_{side}_Index2"])
    inv3 = arm.matrix_world.inverted().to_3x3()
    toward = (inv3 @ (idx2_pos - world_pos(arm, th1))).normalized()
    d = (th1.tail - th1.head).normalized()
    blend_t = max(0.0, min(0.75, 0.44 - thumb_open))
    rotate_about(arm, th1, d.rotation_difference(d.lerp(toward, blend_t).normalized()), th1.head.copy())
    bend_signed(arm, th1, thumb_curl * 0.25, n_world, probe=th3)
    for pb, ck in ((th2, thumb_curl * 0.45), (th3, thumb_curl * 0.40)):
        bend_signed(arm, pb, ck, n_world)


def reset_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        pb.location = Vector((0.0, 0.0, 0.0))
    refresh()


def apply_rest(arm):
    reset_pose(arm)
    P = arm.pose.bones
    for name, xyz in REST_EULER.items():
        P[name].rotation_quaternion = Euler(xyz, "XYZ").to_quaternion()
    refresh()
    for side in ("R", "L"):
        pose_fingers(arm, side, curl=0.25, thumb_curl=0.16, thumb_open=-0.05, splay=0.0)
    refresh()


def pose_body(arm, lean_left=0.0, pitch_fwd=0.0, yaw_left=0.0,
              head_tilt_left=0.0, head_pitch_fwd=0.0, head_yaw_left=0.0):
    P = arm.pose.bones
    spine, chest, uchest = P["J_Bip_C_Spine"], P["J_Bip_C_Chest"], P["J_Bip_C_UpperChest"]
    neck, head = P["J_Bip_C_Neck"], P["J_Bip_C_Head"]

    for pb, w in ((spine, 0.28), (chest, 0.36), (uchest, 0.36)):
        bend_signed(arm, pb, lean_left * w, LEFT, probe=neck)
        bend_signed(arm, pb, pitch_fwd * w, FWD, probe=neck)
        if abs(yaw_left) > 1e-5:
            before = world_pos(arm, P["J_Bip_R_UpperArm"])
            rotate_about(arm, pb, Quaternion(Vector((0.0, 1.0, 0.0)), yaw_left * w), pb.head.copy())
            want = FWD if yaw_left > 0 else -FWD
            if (world_pos(arm, P["J_Bip_R_UpperArm"]) - before).dot(want) < 0:
                rotate_about(arm, pb, Quaternion(Vector((0.0, 1.0, 0.0)), -2.0 * yaw_left * w), pb.head.copy())

    # Cuello compensa mitad de la inclinación del torso; cabeza se ladea con estilo
    net_tilt = head_tilt_left - lean_left * 0.45
    for pb, w in ((neck, 0.40), (head, 0.60)):
        bend_signed(arm, pb, net_tilt * w, LEFT)
        bend_signed(arm, pb, head_pitch_fwd * w, FWD)
        if abs(head_yaw_left) > 1e-5:
            # Gira cabeza hacia la izquierda (+X) usando eje vertical local
            before = world_pos(arm, pb, tail=True)
            nose_probe = world_pos(arm, head) + FWD * 0.1
            rotate_about(arm, pb, Quaternion(Vector((0.0, 1.0, 0.0)), head_yaw_left * w), pb.head.copy())
            # Verificar con el borde del ojo/frente usando eje Y local de cabeza
            eye_dir = (arm.matrix_world.to_3x3() @ head.matrix.to_3x3() @ Vector((0.0, 0.0, 1.0)))
            if eye_dir.dot(LEFT) * head_yaw_left < 0:
                rotate_about(arm, pb, Quaternion(Vector((0.0, 1.0, 0.0)), -2.0 * head_yaw_left * w), pb.head.copy())


def pose_arm_weigh(arm, side, chest_delta, arm_spec):
    """Posa un brazo para el gesto 6-7 con supinación en antebrazo y dedos en cuenco."""
    P = arm.pose.bones
    sx = -1.0 if side == "R" else 1.0
    sh = P[f"J_Bip_{side}_Shoulder"]
    up = P[f"J_Bip_{side}_UpperArm"]
    low = P[f"J_Bip_{side}_LowerArm"]
    hand = P[f"J_Bip_{side}_Hand"]

    # Clavícula acompaña altura del brazo
    bend_signed(arm, sh, arm_spec.get("clav_lift", 0.0), UP, probe=up)

    # Codo sigue parcialmente al torso
    follow = arm_spec.get("chest_follow", 0.45)
    elb_v = Vector(arm_spec["elbow"])
    elb_t = elb_v.lerp(chest_delta @ elb_v, follow)
    aim_bone(arm, up, elb_t)

    # Orientar el giro de UpperArm para que el codo apunte atrás-afuera (pliegue natural al frente)
    low_dir = Vector(arm_spec["forearm_dir"]).normalized()
    low_rot = (chest_delta.to_3x3() @ low_dir).normalized()
    fdir = low_dir.lerp(low_rot, follow).normalized()
    wri_t = world_pos(arm, low) + fdir * 0.215
    aim_bone(arm, low, wri_t)

    # Supinación del antebrazo para poner la palma arriba y ligeramente a cámara
    palm_want = Vector(arm_spec["palm_dir"]).normalized()
    palm_toward(arm, side, palm_want)

    # Mano continúa al antebrazo + ligera flexión/extensión de muñeca (retraso de inercia, <= 18°)
    aim_bone(arm, hand, world_pos(arm, hand) + fdir * 0.13)
    wrist_cup = arm_spec.get("wrist_cup", 0.0)
    if abs(wrist_cup) > 1e-5:
        bend_signed(arm, hand, wrist_cup, palm_normal(arm, side))

    pose_fingers(
        arm, side,
        curl=arm_spec.get("finger_curl", 0.24),
        thumb_curl=arm_spec.get("thumb_curl", 0.20),
        thumb_open=arm_spec.get("thumb_open", 0.14),
        splay=arm_spec.get("finger_splay", 0.05),
    )


def pose_frame(arm, rest_uchest_mat, spec):
    apply_rest(arm)
    if spec.get("exact_rest"):
        return

    pose_body(
        arm,
        lean_left=spec.get("lean_left", 0.0),
        pitch_fwd=spec.get("pitch_fwd", 0.0),
        yaw_left=spec.get("yaw_left", 0.0),
        head_tilt_left=spec.get("head_tilt_left", 0.0),
        head_pitch_fwd=spec.get("head_pitch_fwd", 0.0),
        head_yaw_left=spec.get("head_yaw_left", 0.0),
    )
    chest_delta = arm.pose.bones["J_Bip_C_UpperChest"].matrix.copy() @ rest_uchest_mat.inverted()

    for side in ("R", "L"):
        if side in spec:
            pose_arm_weigh(arm, side, chest_delta, spec[side])


def make_arm_state(side, level, phase_amp=1.0):
    """Genera los parámetros del brazo para estado 'up', 'down', 'prep', o 'drop'."""
    sx = -1.0 if side == "R" else 1.0
    if level == "up":
        # Mano arriba a altura pecho medio-bajo (z ~ 1.18-1.20), bien adentro del encuadre OBS
        return dict(
            clav_lift=0.045,
            elbow=(sx * 0.200, -0.015, 1.12),
            forearm_dir=(sx * 0.05, -0.93, 0.36),
            palm_dir=(sx * -0.06, -0.34, 0.94),
            wrist_cup=-0.30 * phase_amp,
            finger_curl=0.18,
            thumb_curl=0.20,
            thumb_open=0.0,
            finger_splay=0.04,
        )
    if level == "down":
        # Mano abajo a altura cintura/bolsillo alto (z ~ 0.98-1.00, x ~ +-0.22), con margen amplio en OBS
        return dict(
            clav_lift=-0.015,
            elbow=(sx * 0.212, -0.015, 1.075),
            forearm_dir=(sx * 0.10, -0.90, -0.42),
            palm_dir=(sx * -0.08, -0.32, 0.94),
            wrist_cup=0.26 * phase_amp,
            finger_curl=0.16,
            thumb_curl=0.18,
            thumb_open=-0.02,
            finger_splay=0.05,
        )
    if level == "prep":
        return dict(
            clav_lift=-0.015,
            elbow=(sx * 0.215, 0.005, 1.06),
            forearm_dir=(sx * 0.14, -0.50, -0.85),
            palm_dir=(sx * -0.45, -0.65, 0.61),
            wrist_cup=0.05,
            finger_curl=0.28,
            thumb_curl=0.18,
            thumb_open=-0.04,
            finger_splay=0.015,
        )
    # drop (bajada intermedia limpia al frente-costado antes de volver a reposo)
    return dict(
        clav_lift=0.0,
        elbow=(sx * 0.215, -0.005, 1.06),
        forearm_dir=(sx * 0.14, -0.42, -0.89),
        palm_dir=(sx * -0.65, -0.58, 0.49),
        wrist_cup=0.04,
        finger_curl=0.28,
        thumb_curl=0.18,
        thumb_open=-0.04,
        finger_splay=0.01,
    )


def build_timeline_specs():
    """Secuencia completa de 80 cuadros (2.67s @ 30fps) basada en el GIF de referencia."""
    specs = [
        (0, dict(exact_rest=True)),
        # 4: amago inicial sutil (pequeña carga hacia abajo, codos empiezan a despegar)
        (4, dict(
            lean_left=0.0, pitch_fwd=0.025, yaw_left=0.0,
            head_tilt_left=-0.025, head_pitch_fwd=0.018, head_yaw_left=-0.015,
            R=make_arm_state("R", "prep"),
            L=make_arm_state("L", "prep"),
        )),
    ]

    # 8 golpes rítmicos alternando cada 7 cuadros (~0.23s por cambio, igual que el GIF)
    beats = [
        (10, "R_up", 0.95),
        (17, "L_up", 1.00),
        (24, "R_up", 1.05),
        (31, "L_up", 1.05),
        (38, "R_up", 1.00),
        (45, "L_up", 1.00),
        (52, "R_up", 0.92),
        (59, "L_up", 0.85),
    ]
    for fr, mode, amp in beats:
        r_up = mode == "R_up"
        side_sign = -1.0 if r_up else 1.0  # -1 = derecha arriba, +1 = izquierda arriba
        specs.append((fr, dict(
            # Acompañamiento contenido de torso y cabeza (vivo, sin exagerar)
            lean_left=-side_sign * 0.042 * amp,
            pitch_fwd=0.012 + 0.016 * amp,
            yaw_left=side_sign * 0.035 * amp,
            head_tilt_left=side_sign * 0.070 * amp,
            head_pitch_fwd=-0.008 + 0.018 * amp,
            head_yaw_left=side_sign * 0.030 * amp,
            R=make_arm_state("R", "up" if r_up else "down", amp),
            L=make_arm_state("L", "down" if r_up else "up", amp),
        )))

    specs.extend([
        # 66: bajada intermedia controlada frente-costado
        (66, dict(
            lean_left=0.015, pitch_fwd=0.015, yaw_left=0.01,
            head_tilt_left=-0.02, head_pitch_fwd=0.012, head_yaw_left=-0.01,
            R=make_arm_state("R", "drop"),
            L=make_arm_state("L", "drop"),
        )),
        # 73: rebote suave de asentamiento antes de reposo
        (73, dict(
            lean_left=-0.018, pitch_fwd=-0.01, yaw_left=-0.01,
            head_tilt_left=0.02, head_pitch_fwd=-0.01, head_yaw_left=0.01,
            R=dict(
                clav_lift=-0.008, elbow=(-0.185, 0.015, 1.06),
                forearm_dir=(-0.14, -0.10, -0.985), palm_dir=(0.95, -0.25, 0.0),
                wrist_cup=0.0, finger_curl=0.27, thumb_curl=0.17, thumb_open=-0.05, finger_splay=0.0,
            ),
            L=dict(
                clav_lift=-0.008, elbow=(0.185, 0.015, 1.06),
                forearm_dir=(0.14, -0.10, -0.985), palm_dir=(-0.95, -0.25, 0.0),
                wrist_cup=0.0, finger_curl=0.27, thumb_curl=0.17, thumb_open=-0.05, finger_splay=0.0,
            ),
        )),
        # 80: reposo exacto
        (80, dict(exact_rest=True)),
    ])
    return specs


BODY_CHEST = ("J_Bip_C_Spine", "J_Bip_C_Chest", "J_Bip_C_UpperChest")
BODY_HEAD = ("J_Bip_C_Neck", "J_Bip_C_Head")
ARMS_LEAD = (
    "J_Bip_R_Shoulder", "J_Bip_R_UpperArm", "J_Bip_R_LowerArm",
    "J_Bip_L_Shoulder", "J_Bip_L_UpperArm", "J_Bip_L_LowerArm",
)
WRISTS = ("J_Bip_R_Hand", "J_Bip_L_Hand")
FINGERS_BASE = tuple(
    f"J_Bip_{s}_{f}1" for s in ("R", "L") for f in ("Index", "Middle", "Ring", "Little", "Thumb")
)
FINGERS_TIPS = tuple(
    f"J_Bip_{s}_{f}{j}" for s in ("R", "L") for f in ("Index", "Middle", "Ring", "Little", "Thumb") for j in (2, 3)
)
ALL_ANIMATED = BODY_CHEST + BODY_HEAD + ARMS_LEAD + WRISTS + FINGERS_BASE + FINGERS_TIPS


def lag_for_bone(bone_name, frame):
    if frame in (0, FRAME_END):
        return frame
    lag = 0
    if bone_name in BODY_CHEST or bone_name in WRISTS or bone_name in FINGERS_BASE:
        lag = 1
    elif bone_name in BODY_HEAD or bone_name in FINGERS_TIPS:
        lag = 2
    return min(FRAME_END - 1, frame + lag)


def insert_all_keys(arm):
    apply_rest(arm)
    rest_uchest_mat = arm.pose.bones["J_Bip_C_UpperChest"].matrix.copy()
    specs = build_timeline_specs()
    per_bone_keys = {b: [] for b in ALL_ANIMATED}

    for frame, spec in specs:
        pose_frame(arm, rest_uchest_mat, spec)
        for bname in ALL_ANIMATED:
            q = arm.pose.bones[bname].rotation_quaternion.copy()
            per_bone_keys[bname].append((lag_for_bone(bname, frame), q))

        P = arm.pose.bones
        rw = world_pos(arm, P["J_Bip_R_Hand"])
        lw = world_pos(arm, P["J_Bip_L_Hand"])
        re = world_pos(arm, P["J_Bip_R_LowerArm"])
        le = world_pos(arm, P["J_Bip_L_LowerArm"])
        rp = palm_normal(arm, "R")
        lp = palm_normal(arm, "L")
        hd = world_pos(arm, P["J_Bip_C_Head"])
        print(
            f"[POSE f={frame:2d}] Head=({hd.x:+.3f},{hd.y:+.3f},{hd.z:+.3f}) "
            f"R_Elb=({re.x:+.3f},{re.y:+.3f},{re.z:+.3f}) R_Wri=({rw.x:+.3f},{rw.y:+.3f},{rw.z:+.3f}) R_P=({rp.x:+.2f},{rp.y:+.2f},{rp.z:+.2f}) | "
            f"L_Elb=({le.x:+.3f},{le.y:+.3f},{le.z:+.3f}) L_Wri=({lw.x:+.3f},{lw.y:+.3f},{lw.z:+.3f}) L_P=({lp.x:+.2f},{lp.y:+.2f},{lp.z:+.2f})"
        )

    for bname, items in per_bone_keys.items():
        pb = arm.pose.bones[bname]
        prev_q = None
        for f, q in sorted(items, key=lambda x: x[0]):
            if prev_q is not None and q.dot(prev_q) < 0.0:
                q = -q
            prev_q = q
            pb.rotation_quaternion = q
            pb.keyframe_insert("rotation_quaternion", frame=f)


def setup_render(scene):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 420
    scene.render.resolution_y = 420
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    bg = world.node_tree.nodes.get("Background") if world.node_tree else None
    if bg:
        bg.inputs[0].default_value = (0.78, 0.80, 0.84, 1.0)
        bg.inputs[1].default_value = 0.7

    def add_sun(name, rot_deg, energy):
        ld = bpy.data.lights.new(name=name, type="SUN")
        ld.energy = energy
        lo = bpy.data.objects.new(name, ld)
        lo.rotation_euler = tuple(math.radians(a) for a in rot_deg)
        scene.collection.objects.link(lo)

    add_sun("KeySun", (55, 0, -25), 2.5)
    add_sun("FillSun", (75, 0, 145), 1.0)

    def make_cam(name, loc, target, ortho_scale, top=False):
        cd = bpy.data.cameras.new(name)
        cd.type = "ORTHO"
        cd.ortho_scale = ortho_scale
        co = bpy.data.objects.new(name, cd)
        co.location = Vector(loc)
        if top:
            co.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
        else:
            direction = (Vector(target) - Vector(loc)).normalized()
            co.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(co)
        return co

    return {
        "frente": make_cam("CamFrente", (0.0, -2.5, 1.18), (0.0, 0.0, 1.18), 1.05),
        "perfil": make_cam("CamPerfil", (-2.5, 0.0, 1.18), (0.0, 0.0, 1.18), 1.05),
        "arriba": make_cam("CamArriba", (0.0, -0.10, 2.8), (0.0, -0.10, 1.18), 1.00, top=True),
        "manos": make_cam("CamManos", (0.0, -2.2, 1.06), (0.0, -0.15, 1.06), 0.72),
    }


def render_grid_frames(arm, scene, cams, render_dir, frames_to_render, closeup_frames):
    os.makedirs(render_dir, exist_ok=True)
    P = arm.pose.bones
    for f in frames_to_render:
        scene.frame_set(f)
        refresh()
        rw = world_pos(arm, P["J_Bip_R_Hand"])
        lw = world_pos(arm, P["J_Bip_L_Hand"])
        rt = world_pos(arm, P["J_Bip_R_Middle3"], tail=True)
        lt = world_pos(arm, P["J_Bip_L_Middle3"], tail=True)
        rp = palm_normal(arm, "R")
        lp = palm_normal(arm, "L")
        hd = world_pos(arm, P["J_Bip_C_Head"])
        print(
            f"[EVAL f={f:03d}] Head=({hd.x:+.3f},{hd.y:+.3f},{hd.z:+.3f}) "
            f"R_Wri=({rw.x:+.3f},{rw.y:+.3f},{rw.z:+.3f}) R_Tip=({rt.x:+.3f},{rt.y:+.3f},{rt.z:+.3f}) R_P=({rp.x:+.2f},{rp.y:+.2f},{rp.z:+.2f}) | "
            f"L_Wri=({lw.x:+.3f},{lw.y:+.3f},{lw.z:+.3f}) L_Tip=({lt.x:+.3f},{lt.y:+.3f},{lt.z:+.3f}) L_P=({lp.x:+.2f},{lp.y:+.2f},{lp.z:+.2f})"
        )
        for vname in ("frente", "perfil", "arriba"):
            scene.camera = cams[vname]
            scene.render.filepath = os.path.join(render_dir, f"{f:03d}_{vname}.png")
            bpy.ops.render.render(write_still=True)
        if f in closeup_frames:
            scene.camera = cams["manos"]
            scene.render.filepath = os.path.join(render_dir, f"{f:03d}_manos.png")
            bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    vrm_in = argv[0] if len(argv) > 0 else "public/avatar3d/NYRA.vrm"
    vrma_out = argv[1] if len(argv) > 1 else "public/avatar3d/six_seven.vrma"
    render_dir = argv[2] if len(argv) > 2 else None

    clear_scene()
    bpy.ops.import_scene.vrm(filepath=vrm_in)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    bpy.context.view_layer.objects.active = arm

    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 0
    scene.frame_end = FRAME_END

    bpy.ops.object.mode_set(mode="POSE")
    insert_all_keys(arm)
    bpy.ops.object.mode_set(mode="OBJECT")

    os.makedirs(os.path.dirname(os.path.abspath(vrma_out)), exist_ok=True)
    bpy.ops.export_scene.vrma(filepath=vrma_out, armature_object_name=arm.name)
    print(f"Exportado VRMA: {vrma_out} ({FRAME_END} cuadros @ {FPS} fps = {FRAME_END / FPS:.2f}s)")

    if render_dir:
        cams = setup_render(scene)
        frames_to_render = [4, 10, 17, 24, 31, 66, 73]
        render_grid_frames(arm, scene, cams, render_dir, frames_to_render, closeup_frames=(10, 17))


if __name__ == "__main__":
    main()
