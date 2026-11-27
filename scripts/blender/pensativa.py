"""Animación pensativa (mano en el mentón) para NYRA en Blender -> pensativa.vrma.

Referencia: GIF "Thinking Pondering" (Echosmith, 41 cuadros @ 40ms = 1.64s):
  - Brazo izquierdo doblado horizontalmente frente a la cintura/pecho bajo formando apoyo
    para el codo derecho.
  - Brazo derecho sube desde ese apoyo hasta el mentón: palma hacia la mandíbula/cuello,
    índice extendido suave siguiendo la línea derecha de la quijada, resto de dedos
    curvados bajo la barbilla y pulgar suave al costado del mentón.
  - Cabeza ladeada ~9° hacia su derecha apoyándose sutilmente en la mano, con respiración
    y micro-movimientos pensativos en el sostén; entrada y salida limpias por delante.

Uso:
  blender --background --python scripts/blender/pensativa.py -- \
    public/avatar3d/NYRA.vrm public/avatar3d/pensativa.vrma [carpeta_renders]
"""
import math
import os
import sys

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloth_collision import ClothCollider, audit_clip, fix_interpolated_frames, resolve_arm_pose


FPS = 30
FRAME_END = 96  # 3.20s @ 30fps

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
    pb.scale = Vector((1.0, 1.0, 1.0))
    pb.location = Vector((0.0, 0.0, 0.0))
    pb.rotation_quaternion.normalize()
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


def pose_fingers(arm, side, curl, thumb_curl=0.22, thumb_open=0.14, splay=0.0, finger_k=None):
    """Cierra los dedos con una bisagra única por dedo (permite índice extendido al pensar)."""
    P = arm.pose.bones
    n_world = palm_normal(arm, side)
    n_arm = (arm.matrix_world.inverted().to_3x3() @ n_world).normalized()
    cascade = finger_k or {"Index": 0.70, "Middle": 0.90, "Ring": 1.10, "Little": 1.28}
    splay_mul = {"Index": 1.0, "Middle": 0.25, "Ring": -0.45, "Little": -1.10}

    for fname, k in cascade.items():
        b1, b2, b3 = (P[f"J_Bip_{side}_{fname}{j}"] for j in (1, 2, 3))
        d1 = (b1.tail - b1.head).normalized()
        hinge = d1.cross(n_arm).normalized()
        sp = splay * splay_mul[fname]
        if abs(sp) > 1e-5:
            before_sp = world_pos(arm, b3, tail=True)
            rotate_about(arm, b1, Quaternion(n_arm, sp), b1.head.copy())
            thumb_dir = (world_pos(arm, P[f"J_Bip_{side}_Thumb1"]) - world_pos(arm, P[f"J_Bip_{side}_Little1"])).normalized()
            want_sp = thumb_dir if splay_mul[fname] * sp > 0 else -thumb_dir
            if (world_pos(arm, b3, tail=True) - before_sp).dot(want_sp) < 0:
                rotate_about(arm, b1, Quaternion(n_arm, -2.0 * sp), b1.head.copy())

        c = curl * k
        if abs(c) > 1e-5:
            # Comprobar el signo de cierre una sola vez en b1 (si b1 pasa 90°, b2/b3 no deben invertirse)
            before = world_pos(arm, b1, tail=True)
            rotate_about(arm, b1, Quaternion(hinge, c), b1.head.copy())
            if (world_pos(arm, b1, tail=True) - before).dot(n_world) < 0:
                hinge = -hinge
                rotate_about(arm, b1, Quaternion(hinge, 2.0 * c), b1.head.copy())
            for pb, ck in ((b2, c * 0.92), (b3, c * 0.72)):
                rotate_about(arm, pb, Quaternion(hinge, ck), pb.head.copy())

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
        pb.scale = Vector((1.0, 1.0, 1.0))
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

    net_tilt = head_tilt_left - lean_left * 0.45
    net_pitch = head_pitch_fwd - pitch_fwd * 0.35
    for pb, w in ((neck, 0.40), (head, 0.60)):
        bend_signed(arm, pb, net_tilt * w, LEFT)
        bend_signed(arm, pb, net_pitch * w, FWD)
        if abs(head_yaw_left) > 1e-5:
            rotate_about(arm, pb, Quaternion(Vector((0.0, 1.0, 0.0)), head_yaw_left * w), pb.head.copy())
            eye_dir = (arm.matrix_world.to_3x3() @ head.matrix.to_3x3() @ Vector((0.0, 0.0, 1.0)))
            if eye_dir.dot(LEFT) * head_yaw_left < 0:
                rotate_about(arm, pb, Quaternion(Vector((0.0, 1.0, 0.0)), -2.0 * head_yaw_left * w), pb.head.copy())


def pose_arm(arm, side, delta_mat, spec):
    """Posa un brazo transformando objetivos con delta_mat (pecho o cabeza)."""
    P = arm.pose.bones
    sh = P[f"J_Bip_{side}_Shoulder"]
    up = P[f"J_Bip_{side}_UpperArm"]
    low = P[f"J_Bip_{side}_LowerArm"]
    hand = P[f"J_Bip_{side}_Hand"]

    bend_signed(arm, sh, spec.get("clav_lift", 0.0), UP, probe=up)
    bend_signed(arm, sh, spec.get("clav_fwd", 0.0), FWD, probe=up)

    follow = spec.get("follow", 0.8)
    elb_v = Vector(spec["elbow"])
    elb_t = elb_v.lerp(delta_mat @ elb_v, follow)
    aim_bone(arm, up, elb_t)

    wri_v = Vector(spec["wrist"])
    wri_follow = spec.get("wrist_follow", follow)
    wri_t = wri_v.lerp(delta_mat @ wri_v, wri_follow)
    elb_w = world_pos(arm, low)
    w_dir = wri_t - elb_w
    if w_dir.length < 1e-5:
        w_dir = Vector((0.0, -1.0, 0.0))
    aim_bone(arm, low, elb_w + w_dir.normalized() * 0.22)

    palm_w = Vector(spec["palm_dir"]).normalized()
    palm_w = palm_w.lerp((delta_mat.to_3x3() @ palm_w).normalized(), wri_follow).normalized()
    palm_toward(arm, side, palm_w)

    hand_target = spec.get("hand_aim")
    if hand_target is not None:
        ht = Vector(hand_target)
        ht = ht.lerp(delta_mat @ ht, wri_follow)
        h_dir = ht - world_pos(arm, hand)
        if h_dir.length > 1e-4:
            aim_bone(arm, hand, world_pos(arm, hand) + h_dir.normalized() * 0.12)
            palm_toward(arm, side, palm_w)
    else:
        cur_fdir = (world_pos(arm, low, tail=True) - world_pos(arm, low)).normalized()
        aim_bone(arm, hand, world_pos(arm, hand) + cur_fdir * 0.12)

    wrist_cup = spec.get("wrist_cup", 0.0)
    if abs(wrist_cup) > 1e-5:
        bend_signed(arm, hand, wrist_cup, palm_normal(arm, side))

    pose_fingers(
        arm, side,
        curl=spec.get("finger_curl", 0.35),
        thumb_curl=spec.get("thumb_curl", 0.25),
        thumb_open=spec.get("thumb_open", 0.05),
        splay=spec.get("finger_splay", 0.02),
        finger_k=spec.get("finger_k"),
    )


def pose_frame(arm, rest_uchest_mat, rest_head_mat, spec, collider=None):
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
    head_delta = arm.pose.bones["J_Bip_C_Head"].matrix.copy() @ rest_head_mat.inverted()

    clean_rest = collider.capture_arm_quats(("L",)) if collider is not None else None
    if "L" in spec:
        pose_arm(arm, "L", chest_delta, spec["L"])
    if "R" in spec:
        delta = head_delta if spec["R"].get("track_head") else chest_delta
        pose_arm(arm, "R", delta, spec["R"])
    if collider is not None and "L" in spec:
        resolve_arm_pose(collider, arm, "L", pose_arm, chest_delta, spec["L"], margin=0.0025, allow_uncurl=False, init_quats=clean_rest["L"])


CHIN_FINGER_K = {"Index": 0.28, "Middle": 1.15, "Ring": 1.25, "Little": 1.32}


def thinking_pose(jaw_dx=0.0, tilt=-0.16, pitch=0.015, yaw=0.045, lean=0.025):
    """Pose pensativa con brazo L horizontal de apoyo y mano R bajo el mentón (boca siempre libre)."""
    return dict(
        lean_left=lean,
        pitch_fwd=0.015,
        yaw_left=-0.02,
        head_tilt_left=tilt,
        head_pitch_fwd=pitch,
        head_yaw_left=yaw,
        L=dict(
            clav_lift=0.0,
            clav_fwd=0.0,
            elbow=(0.162, -0.095, 1.066),
            wrist=(-0.050, -0.204, 1.026),
            hand_aim=(-0.122, -0.188, 1.028),
            palm_dir=(0.08, 0.72, 0.68),
            wrist_cup=0.20,
            finger_curl=0.82,
            thumb_curl=0.60,
            thumb_open=-0.12,
            finger_splay=0.0,
            finger_k={"Index": 0.95, "Middle": 1.00, "Ring": 1.05, "Little": 1.10},
        ),
        R=dict(
            track_head=True,
            follow=0.20,       # codo apoyado sobre la mano/antebrazo izquierdo
            wrist_follow=0.20, # evitar que el giro de cabeza arrastre la muñeca hacia el centro
            clav_lift=0.0,
            clav_fwd=0.0,
            elbow=(-0.128, -0.122, 1.102),
            wrist=(-0.026 + jaw_dx, -0.122, 1.266),
            hand_aim=(-0.055 + jaw_dx, -0.112, 1.358),
            # Palma mira hacia la quijada derecha (+X) y apenas atrás: índice visible sobre la quijada y resto bajo barbilla
            palm_dir=(0.88, 0.44, 0.12),
            wrist_cup=0.02,
            finger_curl=0.85,
            thumb_curl=0.42,
            thumb_open=0.0,
            finger_splay=0.0,
            finger_k=CHIN_FINGER_K,
        ),
    )


def build_timeline_specs():
    """96 cuadros (3.20s @ 30fps) basados en Echosmith Thinking Pondering."""
    specs = [
        (0, dict(exact_rest=True)),
        # 5: amago sutil, brazos despegan hacia adelante
        (5, dict(
            lean_left=-0.01, pitch_fwd=0.015, yaw_left=0.0,
            head_tilt_left=0.02, head_pitch_fwd=0.012, head_yaw_left=-0.01,
            L=dict(
                clav_lift=0.0, clav_fwd=0.0,
                elbow=(0.20, -0.025, 1.06), wrist=(0.18, -0.13, 0.90),
                palm_dir=(-0.80, 0.55, 0.15), finger_curl=0.35,
            ),
            R=dict(
                clav_lift=0.0, clav_fwd=0.0,
                elbow=(-0.19, -0.05, 1.075), wrist=(-0.15, -0.18, 0.95),
                palm_dir=(0.80, 0.50, 0.25), finger_curl=0.40,
            ),
        )),
        # 14: vía limpia por delante del pecho (L cruza abajo, R sube separado frente al cuello)
        (14, dict(
            lean_left=0.015, pitch_fwd=0.018, yaw_left=-0.015,
            head_tilt_left=-0.08, head_pitch_fwd=0.018, head_yaw_left=0.02,
            L=dict(
                clav_lift=0.0, clav_fwd=0.0,
                elbow=(0.174, -0.075, 1.070), wrist=(0.010, -0.205, 1.055),
                palm_dir=(-0.20, 0.80, -0.25), finger_curl=0.42,
            ),
            R=dict(
                clav_lift=0.01, clav_fwd=0.0,
                elbow=(-0.138, -0.145, 1.112), wrist=(-0.060, -0.235, 1.185),
                hand_aim=(-0.025, -0.260, 1.265),
                palm_dir=(0.74, 0.65, 0.18), finger_curl=0.55, thumb_open=0.10,
            ),
        )),
        # 25: llegada bajo el mentón con cabeza ladeada
        (25, thinking_pose(jaw_dx=0.0, tilt=-0.16, pitch=0.015, yaw=0.045, lean=0.025)),
        # 42: micro-matiz pensativo 1 (inclina apenas más reflexiva, como cuadros #16-#24 del GIF)
        (42, thinking_pose(jaw_dx=0.002, tilt=-0.19, pitch=0.035, yaw=0.065, lean=0.028)),
        # 56: micro-matiz 2 (levanta leve el mentón como evaluando)
        (56, thinking_pose(jaw_dx=0.0, tilt=-0.14, pitch=0.002, yaw=0.030, lean=0.022)),
        # 67: cierre de sostén estable antes de bajar
        (67, thinking_pose(jaw_dx=0.0, tilt=-0.15, pitch=0.012, yaw=0.035, lean=0.024)),
        # 76: despega mano del mentón hacia adelante y suelta apoyo
        (76, dict(
            lean_left=0.012, pitch_fwd=0.012, yaw_left=-0.01,
            head_tilt_left=-0.05, head_pitch_fwd=0.005, head_yaw_left=0.015,
            L=dict(
                clav_lift=0.0, clav_fwd=0.0,
                elbow=(0.176, -0.065, 1.068), wrist=(0.038, -0.200, 1.018),
                palm_dir=(-0.35, 0.75, -0.10), finger_curl=0.35,
            ),
            R=dict(
                clav_lift=0.005, clav_fwd=0.0,
                elbow=(-0.145, -0.125, 1.102), wrist=(-0.090, -0.235, 1.165),
                hand_aim=(-0.040, -0.265, 1.235),
                palm_dir=(0.76, 0.60, 0.15), finger_curl=0.50,
            ),
        )),
        # 84: bajada por adelante-costado
        (84, dict(
            lean_left=-0.005, pitch_fwd=0.010, yaw_left=0.005,
            head_tilt_left=0.015, head_pitch_fwd=0.010, head_yaw_left=-0.005,
            L=dict(
                clav_lift=-0.004, clav_fwd=0.0,
                elbow=(0.195, -0.020, 1.065), wrist=(0.215, -0.110, 0.890),
                palm_dir=(-0.88, 0.40, 0.10), finger_curl=0.28,
            ),
            R=dict(
                clav_lift=-0.004, clav_fwd=0.0,
                elbow=(-0.195, -0.020, 1.065), wrist=(-0.215, -0.115, 0.895),
                palm_dir=(0.88, 0.40, 0.10), finger_curl=0.28,
            ),
        )),
        # 91: rebote suave de asentamiento
        (91, dict(
            lean_left=-0.008, pitch_fwd=-0.005, yaw_left=0.004,
            head_tilt_left=-0.010, head_pitch_fwd=-0.006, head_yaw_left=-0.005,
            L=dict(
                clav_lift=-0.003, clav_fwd=0.0,
                elbow=(0.182, 0.018, 1.068), wrist=(0.220, -0.010, 0.855),
                palm_dir=(-0.95, 0.20, 0.0), finger_curl=0.25,
            ),
            R=dict(
                clav_lift=-0.003, clav_fwd=0.0,
                elbow=(-0.182, 0.018, 1.068), wrist=(-0.220, -0.010, 0.855),
                palm_dir=(0.95, 0.20, 0.0), finger_curl=0.25,
            ),
        )),
        (96, dict(exact_rest=True)),
    ]
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


def insert_all_keys(arm, collider=None):
    apply_rest(arm)
    rest_uchest_mat = arm.pose.bones["J_Bip_C_UpperChest"].matrix.copy()
    rest_head_mat = arm.pose.bones["J_Bip_C_Head"].matrix.copy()
    specs = build_timeline_specs()
    per_bone_keys = {b: [] for b in ALL_ANIMATED}

    for frame, spec in specs:
        pose_frame(arm, rest_uchest_mat, rest_head_mat, spec, collider=collider)
        for bname in ALL_ANIMATED:
            q = arm.pose.bones[bname].rotation_quaternion.copy()
            per_bone_keys[bname].append((lag_for_bone(bname, frame), q))

        P = arm.pose.bones
        rw = world_pos(arm, P["J_Bip_R_Hand"])
        ri = world_pos(arm, P["J_Bip_R_Index3"], tail=True)
        lw = world_pos(arm, P["J_Bip_L_Hand"])
        re = world_pos(arm, P["J_Bip_R_LowerArm"])
        rp = palm_normal(arm, "R")
        hd = world_pos(arm, P["J_Bip_C_Head"])
        print(
            f"[POSE f={frame:2d}] Head=({hd.x:+.3f},{hd.y:+.3f},{hd.z:+.3f}) "
            f"R_Elb=({re.x:+.3f},{re.y:+.3f},{re.z:+.3f}) R_Wri=({rw.x:+.3f},{rw.y:+.3f},{rw.z:+.3f}) "
            f"R_Idx=({ri.x:+.3f},{ri.y:+.3f},{ri.z:+.3f}) R_P=({rp.x:+.2f},{rp.y:+.2f},{rp.z:+.2f}) | "
            f"L_Wri=({lw.x:+.3f},{lw.y:+.3f},{lw.z:+.3f})"
        )

    reset_pose(arm)
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
        "frente": make_cam("CamFrente", (0.0, -2.4, 1.24), (0.0, 0.0, 1.24), 0.92),
        "perfil": make_cam("CamPerfil", (-2.4, 0.0, 1.24), (0.0, 0.0, 1.24), 0.92),
        "arriba": make_cam("CamArriba", (0.0, -0.10, 2.8), (0.0, -0.10, 1.24), 0.85, top=True),
        "cara": make_cam("CamCara", (0.0, -2.1, 1.38), (0.0, -0.07, 1.38), 0.42),
    }


def render_grid_frames(arm, scene, cams, render_dir, frames_to_render, closeup_frames):
    os.makedirs(render_dir, exist_ok=True)
    P = arm.pose.bones
    for f in frames_to_render:
        scene.frame_set(f)
        refresh()
        rw = world_pos(arm, P["J_Bip_R_Hand"])
        ri = world_pos(arm, P["J_Bip_R_Index3"], tail=True)
        rm = world_pos(arm, P["J_Bip_R_Middle3"], tail=True)
        lw = world_pos(arm, P["J_Bip_L_Hand"])
        re = world_pos(arm, P["J_Bip_R_LowerArm"])
        hd = world_pos(arm, P["J_Bip_C_Head"])
        print(
            f"[EVAL f={f:03d}] Head=({hd.x:+.3f},{hd.y:+.3f},{hd.z:+.3f}) "
            f"R_Elb=({re.x:+.3f},{re.y:+.3f},{re.z:+.3f}) R_Wri=({rw.x:+.3f},{rw.y:+.3f},{rw.z:+.3f}) "
            f"R_Idx=({ri.x:+.3f},{ri.y:+.3f},{ri.z:+.3f}) R_Mid=({rm.x:+.3f},{rm.y:+.3f},{rm.z:+.3f}) | "
            f"L_Wri=({lw.x:+.3f},{lw.y:+.3f},{lw.z:+.3f})"
        )
        for vname in ("frente", "perfil", "arriba"):
            scene.camera = cams[vname]
            scene.render.filepath = os.path.join(render_dir, f"{f:03d}_{vname}.png")
            bpy.ops.render.render(write_still=True)
        if f in closeup_frames:
            scene.camera = cams["cara"]
            scene.render.filepath = os.path.join(render_dir, f"{f:03d}_cara.png")
            bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    vrm_in = argv[0] if len(argv) > 0 else "public/avatar3d/NYRA.vrm"
    vrma_out = argv[1] if len(argv) > 1 else "public/avatar3d/pensativa.vrma"
    render_dir = argv[2] if len(argv) > 2 else None

    clear_scene()
    bpy.ops.import_scene.vrm(filepath=vrm_in)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    bpy.context.view_layer.objects.active = arm

    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 0
    scene.frame_end = FRAME_END

    collider = ClothCollider(arm)
    bpy.ops.object.mode_set(mode="POSE")
    insert_all_keys(arm, collider=collider)
    fixed = fix_interpolated_frames(collider, arm, range(12, 80, 3), margin=0.0015, uncurl_sides=())
    if fixed:
        print(f"[CLOTH_FIX] keyframes intermedios ajustados en cuadros: {fixed}", flush=True)
    bpy.ops.object.mode_set(mode="OBJECT")

    hits = audit_clip(collider, arm, range(0, FRAME_END + 1, 3), margin=0.0)
    if hits:
        for fr, side, items in hits:
            print(f"[CLOTH_WARN f={fr:02d} {side}]", ", ".join(f"{k}={v*1000:.1f}mm" for v, k in items), flush=True)
    else:
        print("[CLOTH_OK] 0 penetraciones piel-buzo en toda la animación", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(vrma_out)), exist_ok=True)
    bpy.ops.export_scene.vrma(filepath=vrma_out, armature_object_name=arm.name)
    print(f"Exportado VRMA: {vrma_out} ({FRAME_END} cuadros @ {FPS} fps = {FRAME_END / FPS:.2f}s)")

    if render_dir:
        cams = setup_render(scene)
        frames_to_render = [5, 14, 26, 43, 57, 76, 84]
        render_grid_frames(arm, scene, cams, render_dir, frames_to_render, closeup_frames=(26, 43, 57))


if __name__ == "__main__":
    main()
