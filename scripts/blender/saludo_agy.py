"""Animación de saludo natural para NYRA en Blender -> saludo_agy.vrma.

Uso:
  blender --background --python scripts/blender/saludo_agy.py -- \
    public/avatar3d/NYRA.vrm public/avatar3d/saludo_agy.vrma [carpeta_renders]
"""
import math
import os
import sys

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


FPS = 30
FRAME_END = 78

# Pose de reposo equivalente a REST en public/avatar3d/vrm.js
REST_EULER = {
    "J_Bip_L_UpperArm": (0.0, 0.0, -1.25),
    "J_Bip_R_UpperArm": (0.0, 0.0, 1.25),
    "J_Bip_L_LowerArm": (0.0, 0.0, -0.15),
    "J_Bip_R_LowerArm": (0.0, 0.0, 0.15),
}


def finger_curl_eulers(side_prefix, curl, thumb_abduct=0.05, thumb_curl=0.25, splay=0.0):
    """Devuelve rotaciones Euler locales para dedos basadas en ejes medidos de NYRA.vrm."""
    out = {}
    cascade = {"Index": 0.72, "Middle": 0.92, "Ring": 1.15, "Little": 1.38}
    splay_mul = {"Index": 1.0, "Middle": 0.25, "Ring": -0.45, "Little": -1.10}
    sign = 1.0 if side_prefix == "J_Bip_R_" else -1.0
    for f, k in cascade.items():
        c = curl * k
        sp = sign * splay * splay_mul[f]
        out[f"{side_prefix}{f}1"] = (sp, 0.0, sign * c)
        out[f"{side_prefix}{f}2"] = (0.0, 0.0, sign * c * 0.95)
        out[f"{side_prefix}{f}3"] = (0.0, 0.0, sign * c * 0.75)

    if side_prefix == "J_Bip_R_":
        # Medido en pose R: +X lleva el pulgar adelante/adentro; -Z lo abre; +Z lo cruza a la palma
        out[f"{side_prefix}Thumb1"] = (thumb_curl * 0.65, 0.0, -thumb_abduct)
        out[f"{side_prefix}Thumb2"] = (thumb_curl * 0.85, 0.0, thumb_curl * 0.75)
        out[f"{side_prefix}Thumb3"] = (thumb_curl * 0.75, 0.0, thumb_curl * 0.65)
    else:
        # Medido en L: -Z baja el pulgar hacia el índice (-0.038 Z); +X lo acerca a la palma
        out[f"{side_prefix}Thumb1"] = (thumb_curl * 0.45, 0.0, -0.45)
        out[f"{side_prefix}Thumb2"] = (thumb_curl * 0.55, 0.0, -0.18)
        out[f"{side_prefix}Thumb3"] = (thumb_curl * 0.45, 0.0, -0.12)
    return out


def clear_scene():
    """Limpia los objetos de inicio sin resetear addons habilitados."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.actions):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def reset_pose(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        pb.location = Vector((0.0, 0.0, 0.0))
    bpy.context.view_layer.update()


def set_local_euler(arm, name, xyz):
    pb = arm.pose.bones[name]
    pb.rotation_quaternion = Euler(xyz, "XYZ").to_quaternion()


def apply_rest(arm):
    reset_pose(arm)
    for name, xyz in REST_EULER.items():
        set_local_euler(arm, name, xyz)
    for name, xyz in finger_curl_eulers("J_Bip_R_", 0.25, thumb_abduct=-0.04, thumb_curl=0.18, splay=0.0).items():
        set_local_euler(arm, name, xyz)
    for name, xyz in finger_curl_eulers("J_Bip_L_", 0.25, thumb_curl=0.18, splay=0.0).items():
        set_local_euler(arm, name, xyz)
    bpy.context.view_layer.update()


def aim_bone(arm, name, target_world, twist_local_axis=None, twist_world_dir=None):
    """Apunta el eje Y del hueso hacia target_world y opcionalmente alinea su giro."""
    pb = arm.pose.bones[name]
    mat = pb.matrix.copy()
    head = mat.translation.copy()
    cur_dir = mat.col[1].xyz.normalized()
    want_dir = (Vector(target_world) - head).normalized()
    if cur_dir.length < 1e-6 or want_dir.length < 1e-6:
        return
    rot = cur_dir.rotation_difference(want_dir).to_matrix().to_4x4()
    mat_aim = Matrix.Translation(head) @ rot @ Matrix.Translation(-head) @ mat

    if twist_local_axis is not None and twist_world_dir is not None:
        cur_n = (mat_aim.to_3x3() @ Vector(twist_local_axis)).normalized()
        want_n = Vector(twist_world_dir).normalized()
        p_cur = cur_n - cur_n.dot(want_dir) * want_dir
        p_want = want_n - want_n.dot(want_dir) * want_dir
        if p_cur.length > 1e-5 and p_want.length > 1e-5:
            p_cur.normalize()
            p_want.normalize()
            sin_a = want_dir.dot(p_cur.cross(p_want))
            cos_a = max(-1.0, min(1.0, p_cur.dot(p_want)))
            ang = math.atan2(sin_a, cos_a)
            twist_rot = Matrix.Rotation(ang, 4, want_dir)
            mat_aim = Matrix.Translation(head) @ twist_rot @ Matrix.Translation(-head) @ mat_aim

    pb.matrix = mat_aim
    bpy.context.view_layer.update()


def bone_world_head(arm, name):
    return arm.matrix_world @ arm.pose.bones[name].head


def bone_world_tail(arm, name):
    return arm.matrix_world @ arm.pose.bones[name].tail


def palm_forward_world(arm, side="R"):
    """Normal de la palma en mundo (-X local en mano derecha, +X local en izquierda)."""
    pb = arm.pose.bones[f"J_Bip_{side}_Hand"]
    local_palm = Vector((-1.0, 0.0, 0.0)) if side == "R" else Vector((1.0, 0.0, 0.0))
    return ((arm.matrix_world.to_3x3() @ pb.matrix.to_3x3()) @ local_palm).normalized()


def compute_chest_delta(arm, rest_upper_chest_mat):
    """Transformación del UpperChest respecto al reposo para que los objetivos acompañen el torso."""
    cur = arm.pose.bones["J_Bip_C_UpperChest"].matrix.copy()
    return cur @ rest_upper_chest_mat.inverted()


def pose_frame(arm, rest_upper_chest_mat, spec):
    """Construye una pose completa a partir de objetivos espaciales."""
    apply_rest(arm)

    # 1. Torso y cabeza: torso compensa al lado contrario (+X) y cabeza se inclina hacia la mano (-X)
    t_lean = spec.get("torso_lean_left", 0.0)
    t_pitch = spec.get("torso_pitch_fwd", 0.0)
    t_yaw = spec.get("torso_yaw_left", 0.0)
    set_local_euler(arm, "J_Bip_C_Spine", (t_pitch * 0.25, t_yaw * 0.25, -t_lean * 0.25))
    set_local_euler(arm, "J_Bip_C_Chest", (t_pitch * 0.35, t_yaw * 0.35, -t_lean * 0.38))
    set_local_euler(arm, "J_Bip_C_UpperChest", (t_pitch * 0.40, t_yaw * 0.40, -t_lean * 0.37))

    h_tilt = spec.get("head_tilt_right", 0.0)
    h_pitch = spec.get("head_pitch_fwd", 0.0)
    h_yaw = spec.get("head_yaw_right", 0.0)
    set_local_euler(arm, "J_Bip_C_Neck", (h_pitch * 0.35, -h_yaw * 0.35, (t_lean * 0.45 + h_tilt * 0.35)))
    set_local_euler(arm, "J_Bip_C_Head", (h_pitch * 0.65, -h_yaw * 0.65, (t_lean * 0.55 + h_tilt * 0.65)))
    bpy.context.view_layer.update()

    # 2. Hombro derecho
    clav_lift = spec.get("clav_lift", 0.0)
    clav_fwd = spec.get("clav_fwd", 0.0)
    set_local_euler(arm, "J_Bip_R_Shoulder", (0.0, clav_fwd, -clav_lift))
    bpy.context.view_layer.update()

    # 3. Brazo derecho: objetivos siguen el movimiento del pecho
    if not spec.get("right_rest"):
        chest_follow = spec.get("chest_follow", 0.80)
        delta = compute_chest_delta(arm, rest_upper_chest_mat)

        def xform(pt):
            v = Vector(pt)
            return v.lerp(delta @ v, chest_follow)

        elbow_t = xform(spec["elbow"])
        aim_bone(arm, "J_Bip_R_UpperArm", elbow_t,
                 twist_local_axis=(0.0, 0.0, 1.0), twist_world_dir=(0.25, -1.0, 0.25))

        # Si se pasa forearm_dir, garantiza distancia física exacta (0.215m) sin quebrar la muñeca
        actual_elbow = bone_world_head(arm, "J_Bip_R_LowerArm")
        if "forearm_dir" in spec:
            fdir = Vector(spec["forearm_dir"]).normalized()
            fdir_rot = (delta.to_3x3() @ fdir).normalized()
            wrist_t = actual_elbow + fdir.lerp(fdir_rot, chest_follow).normalized() * 0.215
        else:
            wrist_t = xform(spec["wrist"])

        palm_dir = Vector(spec.get("palm_dir", (0.0, -1.0, 0.0))).normalized()
        aim_bone(arm, "J_Bip_R_LowerArm", wrist_t,
                 twist_local_axis=(-1.0, 0.0, 0.0), twist_world_dir=palm_dir)

        actual_wrist = bone_world_head(arm, "J_Bip_R_Hand")
        if "hand_dir" in spec:
            hdir = Vector(spec["hand_dir"]).normalized()
            hdir_rot = (delta.to_3x3() @ hdir).normalized()
            hand_t = actual_wrist + hdir.lerp(hdir_rot, chest_follow).normalized() * 0.13
        else:
            hand_t = xform(spec["hand_tip"])

        aim_bone(arm, "J_Bip_R_Hand", hand_t,
                 twist_local_axis=(-1.0, 0.0, 0.0), twist_world_dir=palm_dir)

        for name, xyz in finger_curl_eulers(
            "J_Bip_R_",
            curl=spec.get("finger_curl", 0.18),
            thumb_abduct=spec.get("thumb_abduct", 0.05),
            thumb_curl=spec.get("thumb_curl", 0.26),
            splay=spec.get("finger_splay", 0.08),
        ).items():
            set_local_euler(arm, name, xyz)
        bpy.context.view_layer.update()

    # 4. Brazo izquierdo libre: cuelga por gravedad hacia posiciones fijas del mundo
    if not spec.get("left_rest"):
        l_sh_lift = spec.get("left_clav_drop", 0.0)
        set_local_euler(arm, "J_Bip_L_Shoulder", (0.0, 0.0, l_sh_lift))
        bpy.context.view_layer.update()
        aim_bone(arm, "J_Bip_L_UpperArm", spec["l_elbow"],
                 twist_local_axis=(0.0, 0.0, 1.0), twist_world_dir=(0.0, -1.0, 0.1))
        # Palma izquierda (+X local) mirando hacia la pierna (-X) y levemente atrás (+Y)
        aim_bone(arm, "J_Bip_L_LowerArm", spec["l_wrist"],
                 twist_local_axis=(1.0, 0.0, 0.0), twist_world_dir=(-0.78, 0.62, 0.0))
        l_actual_wrist = bone_world_head(arm, "J_Bip_L_Hand")
        l_hand_t = l_actual_wrist + Vector(spec.get("l_hand_dir", (-0.05, -0.10, -1.0))).normalized() * 0.13
        aim_bone(arm, "J_Bip_L_Hand", l_hand_t,
                 twist_local_axis=(1.0, 0.0, 0.0), twist_world_dir=(-0.78, 0.62, 0.0))
        for name, xyz in finger_curl_eulers(
            "J_Bip_L_", curl=0.45, thumb_curl=0.22, splay=-0.02
        ).items():
            set_local_euler(arm, name, xyz)
        bpy.context.view_layer.update()


def build_timeline_specs():
    """Define cuadros clave calibrados contra el GIF de referencia (30 fps)."""
    specs = [
        # 0: reposo exacto
        (0, dict(
            right_rest=True, left_rest=True,
            torso_lean_left=0.0, torso_pitch_fwd=0.0, torso_yaw_left=0.0,
            head_tilt_right=0.0, head_pitch_fwd=0.0, head_yaw_right=0.0,
            clav_lift=0.0, clav_fwd=0.0,
        )),
        # 4: amago (carga visible hacia su derecha antes de subir el brazo)
        (4, dict(
            torso_lean_left=-0.06, torso_pitch_fwd=0.025, torso_yaw_left=-0.045,
            head_tilt_right=-0.05, head_pitch_fwd=0.02, head_yaw_right=-0.03,
            clav_lift=-0.07, clav_fwd=0.0,
            elbow=(-0.215, -0.01, 1.065), forearm_dir=(-0.10, -0.10, -0.99), hand_dir=(-0.05, -0.12, -0.99),
            palm_dir=(0.78, -0.62, 0.0), finger_curl=0.35, thumb_abduct=-0.02, thumb_curl=0.22, finger_splay=0.0,
            l_elbow=(0.185, 0.015, 1.07), l_wrist=(0.20, -0.02, 0.855), l_hand_dir=(-0.05, -0.08, -1.0),
            left_clav_drop=0.04,
        )),
        # 11: pose intermedia de subida (codo doblado pasando frente al costado)
        (11, dict(
            torso_lean_left=0.075, torso_pitch_fwd=-0.015, torso_yaw_left=0.04,
            head_tilt_right=0.11, head_pitch_fwd=-0.02, head_yaw_right=0.05,
            clav_lift=0.10, clav_fwd=0.06,
            elbow=(-0.25, -0.085, 1.13), wrist=(-0.25, -0.175, 1.275), hand_tip=(-0.27, -0.19, 1.395),
            palm_dir=(0.15, -0.97, -0.10), finger_curl=0.25, thumb_abduct=0.04, thumb_curl=0.26, finger_splay=0.04,
            l_elbow=(0.21, 0.02, 1.08), l_wrist=(0.225, -0.025, 0.865), l_hand_dir=(-0.06, -0.10, -1.0),
            left_clav_drop=-0.03,
        )),
    ]

    wave_frames = [
        (17, "out", 0.95),
        (23, "in",  0.98),
        (29, "out", 1.05),
        (35, "in",  1.05),
        (41, "out", 0.95),
        (47, "in",  0.92),
        (53, "out", 0.80),
    ]
    for fr, side, amp in wave_frames:
        if side == "out":
            # Out: codo ligeramente más bajo (1.175), muñeca a altura mentón/mejilla (~1.38-1.40), puntas altura frente (~1.52)
            specs.append((fr, dict(
                torso_lean_left=0.13 + 0.02 * amp, torso_pitch_fwd=-0.025, torso_yaw_left=0.07,
                head_tilt_right=0.21 + 0.025 * amp, head_pitch_fwd=-0.03, head_yaw_right=0.08,
                clav_lift=0.14, clav_fwd=0.06,
                elbow=(-0.28, -0.08, 1.175),
                wrist=(-0.295 - 0.015 * amp, -0.15, 1.37),
                hand_tip=(-0.315 - 0.025 * amp, -0.16, 1.485),
                palm_dir=(0.08, -0.99, -0.04),
                finger_curl=0.19, thumb_abduct=0.045, thumb_curl=0.28, finger_splay=0.09,
                l_elbow=(0.22, 0.02, 1.08), l_wrist=(0.235, -0.03, 0.87), l_hand_dir=(-0.06, -0.10, -1.0),
                left_clav_drop=-0.04,
            )))
        else:
            # In: limpiaparabrisas hacia la sien/frente, separando apenas puntas del flequillo
            specs.append((fr, dict(
                torso_lean_left=0.105, torso_pitch_fwd=-0.015, torso_yaw_left=0.045,
                head_tilt_right=0.27 + 0.025 * amp, head_pitch_fwd=-0.04, head_yaw_right=0.10,
                clav_lift=0.15, clav_fwd=0.07,
                elbow=(-0.288, -0.075, 1.175),
                wrist=(-0.222, -0.165, 1.365),
                hand_tip=(-0.152 + 0.01 * (1.0 - amp), -0.18, 1.468),
                palm_dir=(-0.16, -0.98, -0.04),
                finger_curl=0.29, thumb_abduct=0.02, thumb_curl=0.31, finger_splay=0.015,
                l_elbow=(0.215, 0.025, 1.08), l_wrist=(0.228, -0.025, 0.868), l_hand_dir=(-0.06, -0.10, -1.0),
                left_clav_drop=-0.03,
            )))

    # 61: bajada limpia usando direcciones exactas desde el codo y la muñeca (imposible quebrar atrás)
    specs.extend([
        (61, dict(
            torso_lean_left=0.03, torso_pitch_fwd=0.0, torso_yaw_left=0.015,
            head_tilt_right=0.06, head_pitch_fwd=-0.01, head_yaw_right=0.03,
            clav_lift=0.04, clav_fwd=0.02,
            elbow=(-0.22, -0.035, 1.10),
            forearm_dir=(-0.12, -0.35, -0.93),
            hand_dir=(-0.06, -0.25, -0.96),
            palm_dir=(0.82, -0.55, -0.08),
            finger_curl=0.32, thumb_abduct=0.0, thumb_curl=0.24, finger_splay=0.01,
            l_elbow=(0.20, 0.02, 1.075), l_wrist=(0.215, -0.025, 0.86), l_hand_dir=(-0.06, -0.10, -1.0),
            left_clav_drop=-0.01,
        )),
        # 68: rebote suave pasando del centro antes de reposo
        (68, dict(
            torso_lean_left=-0.05, torso_pitch_fwd=0.02, torso_yaw_left=-0.035,
            head_tilt_right=-0.05, head_pitch_fwd=0.015, head_yaw_right=-0.03,
            clav_lift=-0.03, clav_fwd=0.0,
            elbow=(-0.185, 0.01, 1.06),
            forearm_dir=(-0.16, -0.12, -0.98),
            hand_dir=(-0.05, -0.08, -0.99),
            palm_dir=(0.95, -0.25, 0.0),
            finger_curl=0.28, thumb_abduct=-0.03, thumb_curl=0.20, finger_splay=0.0,
            l_elbow=(0.185, 0.015, 1.07), l_wrist=(0.20, -0.02, 0.852), l_hand_dir=(-0.05, -0.08, -1.0),
            left_clav_drop=0.02,
        )),
        # 78: reposo exacto
        (78, dict(
            right_rest=True, left_rest=True,
            torso_lean_left=0.0, torso_pitch_fwd=0.0, torso_yaw_left=0.0,
            head_tilt_right=0.0, head_pitch_fwd=0.0, head_yaw_right=0.0,
            clav_lift=0.0, clav_fwd=0.0,
        )),
    ])
    return specs


BODY_CHEST = ("J_Bip_C_Spine", "J_Bip_C_Chest", "J_Bip_C_UpperChest")
BODY_HEAD = ("J_Bip_C_Neck", "J_Bip_C_Head")
RIGHT_ARM_LEAD = ("J_Bip_R_Shoulder", "J_Bip_R_UpperArm", "J_Bip_R_LowerArm")
RIGHT_WRIST = ("J_Bip_R_Hand",)
RIGHT_FINGERS_BASE = (
    "J_Bip_R_Index1", "J_Bip_R_Middle1", "J_Bip_R_Ring1", "J_Bip_R_Little1", "J_Bip_R_Thumb1"
)
RIGHT_FINGERS_TIPS = (
    "J_Bip_R_Index2", "J_Bip_R_Index3",
    "J_Bip_R_Middle2", "J_Bip_R_Middle3",
    "J_Bip_R_Ring2", "J_Bip_R_Ring3",
    "J_Bip_R_Little2", "J_Bip_R_Little3",
    "J_Bip_R_Thumb2", "J_Bip_R_Thumb3",
)
LEFT_ARM_BONES = (
    "J_Bip_L_Shoulder", "J_Bip_L_UpperArm", "J_Bip_L_LowerArm", "J_Bip_L_Hand",
    "J_Bip_L_Index1", "J_Bip_L_Index2", "J_Bip_L_Index3",
    "J_Bip_L_Middle1", "J_Bip_L_Middle2", "J_Bip_L_Middle3",
    "J_Bip_L_Ring1", "J_Bip_L_Ring2", "J_Bip_L_Ring3",
    "J_Bip_L_Little1", "J_Bip_L_Little2", "J_Bip_L_Little3",
    "J_Bip_L_Thumb1", "J_Bip_L_Thumb2", "J_Bip_L_Thumb3",
)

ALL_ANIMATED = (
    BODY_CHEST + BODY_HEAD + RIGHT_ARM_LEAD + RIGHT_WRIST
    + RIGHT_FINGERS_BASE + RIGHT_FINGERS_TIPS + LEFT_ARM_BONES
)


def lag_for_bone(bone_name, frame):
    if frame in (0, FRAME_END):
        return frame
    lag = 0
    if bone_name in BODY_CHEST or bone_name in RIGHT_WRIST or bone_name in RIGHT_FINGERS_BASE:
        lag = 1
    elif bone_name in BODY_HEAD or bone_name in RIGHT_FINGERS_TIPS:
        lag = 2
    elif bone_name in LEFT_ARM_BONES:
        lag = 1
    return min(FRAME_END - 1, frame + lag)


def insert_all_keys(arm):
    apply_rest(arm)
    rest_upper_chest_mat = arm.pose.bones["J_Bip_C_UpperChest"].matrix.copy()

    specs = build_timeline_specs()
    per_bone_keys = {b: [] for b in ALL_ANIMATED}

    for frame, spec in specs:
        pose_frame(arm, rest_upper_chest_mat, spec)
        for bname in ALL_ANIMATED:
            q = arm.pose.bones[bname].rotation_quaternion.copy()
            per_bone_keys[bname].append((lag_for_bone(bname, frame), q))

        r_elb = bone_world_head(arm, "J_Bip_R_LowerArm")
        r_wri = bone_world_head(arm, "J_Bip_R_Hand")
        r_tip = bone_world_tail(arm, "J_Bip_R_Middle3")
        r_thb = bone_world_tail(arm, "J_Bip_R_Thumb3")
        head = bone_world_head(arm, "J_Bip_C_Head")
        pf = palm_forward_world(arm, "R")
        print(
            f"[POSE f={frame:2d}] Head=({head.x:+.3f},{head.y:+.3f},{head.z:+.3f}) "
            f"Elb=({r_elb.x:+.3f},{r_elb.y:+.3f},{r_elb.z:+.3f}) "
            f"Wri=({r_wri.x:+.3f},{r_wri.y:+.3f},{r_wri.z:+.3f}) "
            f"Tip=({r_tip.x:+.3f},{r_tip.y:+.3f},{r_tip.z:+.3f}) "
            f"Thb=({r_thb.x:+.3f},{r_thb.y:+.3f},{r_thb.z:+.3f}) "
            f"PalmF=({pf.x:+.2f},{pf.y:+.2f},{pf.z:+.2f})"
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
        "frente": make_cam("CamFrente", (0.0, -2.5, 1.24), (0.0, 0.0, 1.24), 1.10),
        "perfil": make_cam("CamPerfil", (-2.5, 0.0, 1.24), (0.0, 0.0, 1.24), 1.10),
        "arriba": make_cam("CamArriba", (0.0, -0.08, 2.8), (0.0, -0.08, 1.24), 0.95, top=True),
        "mano": make_cam("CamMano", (-0.18, -2.2, 1.43), (-0.18, 0.0, 1.43), 0.42),
    }


def render_grid_frames(arm, scene, cams, render_dir, frames_to_render, closeup_frames):
    os.makedirs(render_dir, exist_ok=True)
    for f in frames_to_render:
        scene.frame_set(f)
        bpy.context.view_layer.update()
        r_elb = bone_world_head(arm, "J_Bip_R_LowerArm")
        r_wri = bone_world_head(arm, "J_Bip_R_Hand")
        r_m1 = bone_world_head(arm, "J_Bip_R_Middle1")
        r_tip = bone_world_tail(arm, "J_Bip_R_Middle3")
        head = bone_world_head(arm, "J_Bip_C_Head")
        pf = palm_forward_world(arm, "R")
        print(
            f"[EVAL f={f:03d}] Head=({head.x:+.3f},{head.y:+.3f},{head.z:+.3f}) "
            f"Elb=({r_elb.x:+.3f},{r_elb.y:+.3f},{r_elb.z:+.3f}) "
            f"Wri=({r_wri.x:+.3f},{r_wri.y:+.3f},{r_wri.z:+.3f}) "
            f"M1=({r_m1.x:+.3f},{r_m1.y:+.3f},{r_m1.z:+.3f}) "
            f"Tip=({r_tip.x:+.3f},{r_tip.y:+.3f},{r_tip.z:+.3f}) "
            f"PalmF=({pf.x:+.2f},{pf.y:+.2f},{pf.z:+.2f})"
        )
        for vname in ("frente", "perfil", "arriba"):
            scene.camera = cams[vname]
            scene.render.filepath = os.path.join(render_dir, f"{f:03d}_{vname}.png")
            bpy.ops.render.render(write_still=True)
        if f in closeup_frames:
            scene.camera = cams["mano"]
            scene.render.filepath = os.path.join(render_dir, f"{f:03d}_mano.png")
            bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    vrm_in = argv[0] if len(argv) > 0 else "public/avatar3d/NYRA.vrm"
    vrma_out = argv[1] if len(argv) > 1 else "public/avatar3d/saludo_agy.vrma"
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
        frames_to_render = [4, 11, 17, 23, 29, 35, 61, 68]
        render_grid_frames(arm, scene, cams, render_dir, frames_to_render, closeup_frames=(17, 23, 29))


if __name__ == "__main__":
    main()
