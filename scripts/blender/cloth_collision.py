"""Sistema reutilizable de colisión contra ropa y piel para animaciones de NYRA en Blender.

Usa las mallas deformadas reales de `NYRA.vrm` en cada cuadro (`Body`: buzo `Tops_01_CLOTH`
y piel) separadas en zonas (`TORSO`, `R_SLV`, `L_SLV`) para:
1. Medir en milímetros si vértices de piel de mano/dedos invaden el torso o la manga opuesta.
2. Ajustar muñeca/mano y abrir suavemente los dedos colisionantes para que apoyen sobre la
   superficie externa de la tela en lugar de atravesarla.
3. Auditar los cuadros interpolados de toda la línea de tiempo (incluidos cruces entre poses).
"""
import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_point_line


FINGER_NAMES = ("Index", "Middle", "Ring", "Little", "Thumb")
HAND_PART_PREFIXES = ("Hand",) + FINGER_NAMES
_RAY_DIRS_5 = (
    Vector((1.0, 0.03, 0.02)).normalized(),
    Vector((-1.0, 0.02, 0.03)).normalized(),
    Vector((0.02, 1.0, 0.03)).normalized(),
    Vector((0.03, -1.0, 0.02)).normalized(),
    Vector((0.02, 0.03, 1.0)).normalized(),
)


def _seg_closest(pt, a, b):
    _, t = intersect_point_line(pt, a, b)
    t_c = max(0.0, min(1.0, t))
    return a.lerp(b, t_c), t


def _vert_zone(v, vg_names):
    w = {"R_SLV": 0.0, "L_SLV": 0.0, "TORSO": 0.0}
    for g in v.groups:
        gn = vg_names.get(g.group, "")
        if any(f"J_Bip_R_{k}" in gn for k in ("UpperArm", "LowerArm", *HAND_PART_PREFIXES)):
            w["R_SLV"] += g.weight
        elif any(f"J_Bip_L_{k}" in gn for k in ("UpperArm", "LowerArm", *HAND_PART_PREFIXES)):
            w["L_SLV"] += g.weight
        else:
            w["TORSO"] += g.weight
    return max(w, key=w.get)


def _is_hand_bone_name(gn):
    return any(f"_{k}" in gn for k in HAND_PART_PREFIXES)


def _precompute_zone_polys(rest_mesh, poly_zones, zone, cap_holes):
    """Precalcula tuplas de índices de vértices originales (`0..N-1`) por zona una sola vez."""
    bm = bmesh.new()
    bm.from_mesh(rest_mesh)
    orig_layer = bm.verts.layers.int.new("orig_vi")
    for i, v in enumerate(bm.verts):
        v[orig_layer] = i
    bm.faces.ensure_lookup_table()
    to_del = [fc for i, fc in enumerate(bm.faces) if poly_zones[i] != zone]
    bmesh.ops.delete(bm, geom=to_del, context="FACES")
    if cap_holes:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
        bedges = [e for e in bm.edges if e.is_boundary]
        if bedges:
            bmesh.ops.holes_fill(bm, edges=bedges, sides=0)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    polys = [tuple(v[orig_layer] for v in f.verts) for f in bm.faces]
    bm.free()
    return polys


class ClothCollider:
    """Detector de penetración piel-ropa sobre la malla deformada del personaje."""

    def __init__(self, arm, body_obj=None):
        self.arm = arm
        self.body = body_obj or bpy.data.objects.get("Body")
        if self.body is None:
            raise RuntimeError("No se encontró el objeto de malla 'Body' en la escena.")

        mesh = self.body.data
        vg_names = {g.index: g.name for g in self.body.vertex_groups}

        # Vértices reales de la piel de cada mano y cada dedo (evita aproximar radios).
        self.hand_verts = {"R": [], "L": []}
        self.wrist_verts = {"R": [], "L": []}
        self.finger_verts = {s: {fn: [] for fn in FINGER_NAMES} for s in ("R", "L")}
        for vi, v in enumerate(mesh.vertices):
            dom = max(v.groups, key=lambda g: g.weight, default=None)
            if not dom:
                continue
            gn = vg_names.get(dom.group, "")
            for s in ("R", "L"):
                prefix = f"J_Bip_{s}_"
                if not gn.startswith(prefix):
                    continue
                part = gn[len(prefix):]
                if part == "Hand":
                    self.hand_verts[s].append((vi, part))
                    self.wrist_verts[s].append(vi)
                elif any(part.startswith(fn) for fn in FINGER_NAMES):
                    self.hand_verts[s].append((vi, part))
                    fn = next(f for f in FINGER_NAMES if part.startswith(f))
                    self.finger_verts[s][fn].append((vi, part))
                    if part.endswith("1"):
                        self.wrist_verts[s].append(vi)

        # Clasificación por polígono: buzo y torso/brazos (excluye piel de manos).
        poly_zones = []
        for p in mesh.polygons:
            mat = self.body.material_slots[p.material_index].material
            mname = mat.name if mat else ""
            if "Tops" not in mname and "SKIN" not in mname:
                poly_zones.append(None)
                continue
            if "SKIN" in mname:
                has_hand = False
                for vi in p.vertices:
                    dom = max(mesh.vertices[vi].groups, key=lambda g: g.weight, default=None)
                    if dom and _is_hand_bone_name(vg_names.get(dom.group, "")):
                        has_hand = True
                        break
                if has_hand:
                    poly_zones.append(None)
                    continue
            sc = {"TORSO": 0, "R_SLV": 0, "L_SLV": 0}
            for vi in p.vertices:
                sc[_vert_zone(mesh.vertices[vi], vg_names)] += 1
            poly_zones.append(max(sc, key=sc.get))

        self.zone_polys = {
            z: _precompute_zone_polys(mesh, poly_zones, z, cap_holes=(z != "TORSO"))
            for z in ("TORSO", "R_SLV", "L_SLV")
        }

    def eval_verts_world(self):
        deps = bpy.context.evaluated_depsgraph_get()
        deps.update()
        ev = self.body.evaluated_get(deps)
        me = ev.to_mesh()
        mw = ev.matrix_world
        verts_w = [mw @ v.co for v in me.vertices]
        ev.to_mesh_clear()
        return verts_w

    def capture_arm_quats(self, sides=("R", "L")):
        """Guarda rotaciones base limpias antes de posar los brazos (evita doble acumulación)."""
        P = self.arm.pose.bones
        return {
            s: {n: P[n].rotation_quaternion.copy() for n in _side_bone_names(s) if n in P}
            for s in sides
        }

    def snapshot(self, reuse_bvhs=None):
        """Evalúa vértices deformados y construye BVHs rápidos por lista de polígonos."""
        verts_w = self.eval_verts_world()
        if reuse_bvhs is not None:
            return {"verts_w": verts_w, "bvhs": reuse_bvhs["bvhs"], "spine": reuse_bvhs["spine"]}

        bvhs = {
            z: BVHTree.FromPolygons(verts_w, self.zone_polys[z], all_triangles=False)
            for z in ("TORSO", "R_SLV", "L_SLV")
        }
        P = self.arm.pose.bones
        spine_a = self.arm.matrix_world @ P["J_Bip_C_Hips"].head
        spine_b = self.arm.matrix_world @ P["J_Bip_C_UpperChest"].tail
        return {"verts_w": verts_w, "bvhs": bvhs, "spine": (spine_a, spine_b)}

    def _point_clearance(self, pt, snap, check_zones, margin=0.0):
        """Devuelve [(clearance_m, zone, push_vec)] con valores negativos si penetra."""
        hits = []
        bvhs = snap["bvhs"]
        spine_a, spine_b = snap["spine"]

        if "TORSO" in check_zones:
            ori, tc = _seg_closest(pt, spine_a, spine_b)
            if -0.05 <= tc <= 1.03:
                ori_z = Vector((ori.x, ori.y, pt.z))
                d = pt - ori_z
                if d.length > 1e-4:
                    dn = d.normalized()
                    max_r = 0.0
                    o = ori_z
                    for _ in range(6):
                        hl, _, _, _ = bvhs["TORSO"].ray_cast(o, dn, 0.35)
                        if hl is None:
                            break
                        r = (hl - ori_z).length
                        if r > max_r:
                            max_r = r
                        o = hl + dn * 1e-4
                    if max_r > 0.0:
                        clr = (d.length - max_r) - margin
                        if clr < 0.0:
                            hits.append((clr, "TORSO", dn))

        for z in check_zones:
            if z == "TORSO":
                continue
            bvh = bvhs[z]
            loc, nor, _, dist = bvh.find_nearest(pt)
            if loc is None or dist > 0.07:
                continue
            inside = 0
            for d in _RAY_DIRS_5:
                cnt = 0
                o = pt + d * 5e-4
                for _ in range(8):
                    hl, _, _, _ = bvh.ray_cast(o, d)
                    if hl is None:
                        break
                    cnt += 1
                    o = hl + d * 5e-4
                if cnt % 2 == 1:
                    inside += 1
            sd = -dist if inside >= 3 else dist
            clr = sd - margin
            if clr < 0.0:
                push = (loc - pt).normalized() if inside >= 3 and dist > 1e-5 else nor.normalized()
                if inside >= 3 and nor.dot(push) < 0:
                    push = nor.normalized()
                hits.append((clr, z, push))

        return hits

    def inspect_side(self, side, snap=None, margin=0.0, stride=1):
        """Lista de (clearance_m, zone, part, push_vec) para toda la mano de `side`."""
        snap = snap or self.snapshot()
        opp = "L_SLV" if side == "R" else "R_SLV"
        zones = ("TORSO", opp)
        vw = snap["verts_w"]
        bad = []
        for vi, part in self.hand_verts[side][::stride]:
            for clr, z, push in self._point_clearance(vw[vi], snap, zones, margin=margin):
                bad.append((clr, z, part, push))
        bad.sort(key=lambda x: x[0])
        return bad

    def worst_by_part(self, side, snap=None, margin=0.0, stride=2):
        """Resumen { 'ZONA:Parte': clearance_m } ordenado de peor a mejor."""
        worst = {}
        for clr, z, part, _ in self.inspect_side(side, snap=snap, margin=margin, stride=stride):
            key = f"{z}:{part}"
            if key not in worst or clr < worst[key]:
                worst[key] = clr
        return sorted((v, k) for k, v in worst.items())


def _palm_normal(arm, side):
    P = arm.pose.bones
    h = arm.matrix_world @ P[f"J_Bip_{side}_Hand"].head
    idx = arm.matrix_world @ P[f"J_Bip_{side}_Index1"].head
    lit = arm.matrix_world @ P[f"J_Bip_{side}_Little1"].head
    if side == "R":
        return (idx - h).cross(lit - h).normalized()
    return (lit - h).cross(idx - h).normalized()


def _rotate_about(pb, q, pivot_arm):
    pb.matrix = Matrix.Translation(pivot_arm) @ q.to_matrix().to_4x4() @ Matrix.Translation(-pivot_arm) @ pb.matrix
    pb.scale = Vector((1.0, 1.0, 1.0))
    pb.location = Vector((0.0, 0.0, 0.0))
    pb.rotation_quaternion.normalize()
    bpy.context.view_layer.update()


def _uncurl_finger_outward(collider, arm, side, fname, opp_zones, base_snap, margin, max_steps=8):
    """Abre de a poco un dedo que penetra la tela hasta que su piel quede libre."""
    P = arm.pose.bones
    b1, b2, b3 = (P[f"J_Bip_{side}_{fname}{j}"] for j in (1, 2, 3))
    inv3 = arm.matrix_world.inverted().to_3x3()
    fverts = collider.finger_verts[side][fname][::3]

    # Salir rápido si este dedo ya está 100% libre en base_snap
    vw0 = base_snap["verts_w"]
    if not any(collider._point_clearance(vw0[vi], base_snap, opp_zones, margin=margin) for vi, _ in fverts):
        return True

    for _ in range(max_steps):
        snap = collider.snapshot(reuse_bvhs=base_snap)
        vw = snap["verts_w"]
        worst_clr = 0.0
        worst_push = None
        for vi, _ in fverts:
            for clr, _, push in collider._point_clearance(vw[vi], snap, opp_zones, margin=margin):
                if clr < worst_clr:
                    worst_clr = clr
                    worst_push = push
        if worst_push is None:
            return True

        step_ang = min(0.22, max(0.06, -worst_clr * 8.5))
        if fname == "Thumb":
            for pb, k in ((b1, 0.55), (b2, 0.70), (b3, 0.50)):
                d_loc = (pb.tail - pb.head).normalized()
                axis = d_loc.cross(inv3 @ worst_push)
                if axis.length > 1e-5:
                    before = arm.matrix_world @ b3.tail
                    _rotate_about(pb, Quaternion(axis.normalized(), step_ang * k), pb.head.copy())
                    if (arm.matrix_world @ b3.tail - before).dot(worst_push) < 0:
                        _rotate_about(pb, Quaternion(axis.normalized(), -2.0 * step_ang * k), pb.head.copy())
        else:
            n_arm = (inv3 @ _palm_normal(arm, side)).normalized()
            d1 = (b1.tail - b1.head).normalized()
            hinge = d1.cross(n_arm)
            if hinge.length < 1e-5:
                break
            hinge.normalize()
            for pb, k in ((b1, 0.55), (b2, 0.65), (b3, 0.45)):
                before = arm.matrix_world @ b3.tail
                _rotate_about(pb, Quaternion(hinge, step_ang * k), pb.head.copy())
                if (arm.matrix_world @ b3.tail - before).dot(worst_push) < 0:
                    _rotate_about(pb, Quaternion(hinge, -2.0 * step_ang * k), pb.head.copy())
    return False


def _side_bone_names(side):
    names = [f"J_Bip_{side}_Shoulder", f"J_Bip_{side}_UpperArm", f"J_Bip_{side}_LowerArm", f"J_Bip_{side}_Hand"]
    for fn in FINGER_NAMES:
        for j in (1, 2, 3):
            names.append(f"J_Bip_{side}_{fn}{j}")
    return names


def resolve_arm_pose(collider, arm, side, pose_arm_fn, chest_delta, spec, margin=0.0025, allow_uncurl=True, init_quats=None):
    """Posa un brazo y empuja muñeca/mano/dedos fuera de la tela deformada."""
    P = arm.pose.bones
    side_pbs = [P[n] for n in _side_bone_names(side) if n in P]
    rest_quats = {n: q.copy() for n, q in init_quats.items()} if init_quats is not None else {
        pb.name: pb.rotation_quaternion.copy() for pb in side_pbs
    }

    def reset_side():
        for pb in side_pbs:
            pb.rotation_quaternion = rest_quats[pb.name].copy()
            pb.location = Vector((0.0, 0.0, 0.0))
            pb.scale = Vector((1.0, 1.0, 1.0))
        bpy.context.view_layer.update()

    work = dict(spec)
    if init_quats is not None:
        reset_side()
    pose_arm_fn(arm, side, chest_delta, work)
    opp_zones = ("TORSO", "L_SLV" if side == "R" else "R_SLV")

    sh_head = arm.matrix_world @ P[f"J_Bip_{side}_UpperArm"].head
    max_reach = 0.435
    check_verts = collider.wrist_verts[side][::4] if allow_uncurl else [vi for vi, _ in collider.hand_verts[side][::4]]

    # Fase 1: sacar muñeca/palma (y puño completo cuando allow_uncurl=False).
    for _ in range(8):
        snap = collider.snapshot()
        vw = snap["verts_w"]
        worst_clr = 0.0
        push_sum = Vector((0.0, 0.0, 0.0))
        cnt = 0
        for vi in check_verts:
            for clr, _, push in collider._point_clearance(vw[vi], snap, opp_zones, margin=margin):
                if clr < 0.0:
                    push_sum += push * (-clr)
                    cnt += 1
                    if clr < worst_clr:
                        worst_clr = clr
        if cnt == 0 or push_sum.length < 1e-5:
            break
        dir_w = push_sum.normalized()
        shift = min(0.028, (-worst_clr) * 0.95 + 0.002)
        delta_w = dir_w * shift

        inv_d = chest_delta.to_3x3().inverted() @ delta_w
        follow = work.get("chest_follow", work.get("follow", 0.75))
        adj = delta_w.lerp(inv_d, follow)
        new_wri = Vector(work["wrist"]) + adj
        if (new_wri - sh_head).length > max_reach:
            new_wri = sh_head + (new_wri - sh_head).normalized() * max_reach
        work["wrist"] = tuple(new_wri)
        if work.get("hand_aim") is not None:
            work["hand_aim"] = tuple(Vector(work["hand_aim"]) + adj)
        reset_side()
        pose_arm_fn(arm, side, chest_delta, work)

    # Fase 2: desplegar suavemente dedos que rocen (solo si el gesto admite mano apoyada abierta).
    base_snap = collider.snapshot()
    if allow_uncurl:
        for fn in FINGER_NAMES:
            _uncurl_finger_outward(collider, arm, side, fn, opp_zones, base_snap, margin=margin)

    # Fase 3: si alguna punta aún roza, inclinar muñeca afuera sin desarmar la mano.
    hand_pb = P[f"J_Bip_{side}_Hand"]
    inv3 = arm.matrix_world.inverted().to_3x3()
    for _ in range(5):
        snap = collider.snapshot(reuse_bvhs=base_snap)
        bad = collider.inspect_side(side, snap=snap, margin=margin, stride=2)
        if not bad:
            break
        worst_clr, _, _, push = bad[0]
        h_axis = (hand_pb.tail - hand_pb.head).normalized()
        rot_axis = h_axis.cross(inv3 @ push)
        if rot_axis.length < 1e-5:
            break
        ang = min(0.14, max(0.03, -worst_clr * 6.0))
        before = arm.matrix_world @ P[f"J_Bip_{side}_Middle3"].tail
        _rotate_about(hand_pb, Quaternion(rot_axis.normalized(), ang), hand_pb.head.copy())
        if (arm.matrix_world @ P[f"J_Bip_{side}_Middle3"].tail - before).dot(push) < 0:
            _rotate_about(hand_pb, Quaternion(rot_axis.normalized(), -2.0 * ang), hand_pb.head.copy())

    return work


def _batch_insert_aligned_keys(bones, frame):
    """Guarda e inserta keyframes para varios huesos sin perder rotaciones al cambiar de cuadro."""
    scene = bpy.context.scene
    saved_q = {pb.name: pb.rotation_quaternion.copy() for pb in bones}
    scene.frame_set(max(0, frame - 1))
    prev_q = {pb.name: pb.rotation_quaternion.copy() for pb in bones}
    scene.frame_set(frame)
    for pb in bones:
        q = saved_q[pb.name]
        if q.dot(prev_q[pb.name]) < 0.0:
            q = -q
        pb.rotation_quaternion = q
        pb.keyframe_insert("rotation_quaternion", frame=frame)


def fix_interpolated_frames(collider, arm, frames, margin=0.0015, uncurl_sides=("R",)):
    """Corrige e inserta keyframes en cualquier cuadro intermedio donde la interpolación roce ropa."""
    scene = bpy.context.scene
    P = arm.pose.bones
    inv3 = arm.matrix_world.inverted().to_3x3()
    fixed_frames = []

    for fr in frames:
        scene.frame_set(fr)
        snap = collider.snapshot()
        changed_any = False
        touched_all = {}
        for side in ("L", "R"):
            opp_zones = ("TORSO", "L_SLV" if side == "R" else "R_SLV")
            low_pb = P[f"J_Bip_{side}_LowerArm"]
            hand_pb = P[f"J_Bip_{side}_Hand"]

            for _ in range(6):
                cur_snap = collider.snapshot() if _ > 0 else snap
                bad = collider.inspect_side(side, snap=cur_snap, margin=margin, stride=2)
                if not bad:
                    break
                changed_any = True
                worst_clr, _, _, push = bad[0]
                if side in uncurl_sides:
                    for fn in FINGER_NAMES:
                        _uncurl_finger_outward(collider, arm, side, fn, opp_zones, cur_snap, margin=margin, max_steps=4)
                        for j in (1, 2, 3):
                            pb = P[f"J_Bip_{side}_{fn}{j}"]
                            touched_all[pb.name] = pb
                    cur_snap = collider.snapshot()
                    bad = collider.inspect_side(side, snap=cur_snap, margin=margin, stride=2)
                    if not bad:
                        break
                    worst_clr, _, _, push = bad[0]

                d_low = (low_pb.tail - low_pb.head).normalized()
                ax = d_low.cross(inv3 @ push)
                if ax.length > 1e-5:
                    ang = min(0.09, max(0.02, -worst_clr * 3.0))
                    before = arm.matrix_world @ low_pb.tail
                    _rotate_about(low_pb, Quaternion(ax.normalized(), ang), low_pb.head.copy())
                    if (arm.matrix_world @ low_pb.tail - before).dot(push) < 0:
                        _rotate_about(low_pb, Quaternion(ax.normalized(), -2.0 * ang), low_pb.head.copy())
                    touched_all[low_pb.name] = low_pb
                    touched_all[hand_pb.name] = hand_pb

        if changed_any and touched_all:
            _batch_insert_aligned_keys(list(touched_all.values()), fr)
            fixed_frames.append(fr)
    return fixed_frames


def audit_clip(collider, arm, frames, margin=0.0):
    """Recorre todos los cuadros indicados de la animación ya keyframeada y reporta roces."""
    scene = bpy.context.scene
    report = []
    for f in frames:
        scene.frame_set(f)
        snap = collider.snapshot()
        for side in ("R", "L"):
            worst = collider.worst_by_part(side, snap=snap, margin=margin, stride=2)
            if worst:
                report.append((f, side, worst[:4]))
    return report
