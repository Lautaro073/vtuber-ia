"""Sistema reutilizable de colisión contra ropa y piel para animaciones de NYRA en Blender.

Usa las mallas deformadas reales de `NYRA.vrm` en cada cuadro para detectar y resolver
colisiones piel-buzo (y opcionalmente piel-rostro):
- Cobertura por defecto (`Body`): vértices reales de piel de mano/dedos (`Hand`, `Index..Little`,
  `Thumb`) frente a las zonas `TORSO`, `R_SLV` y `L_SLV` (`Tops_01_CLOTH` y piel no manual).
- Cobertura facial opcional (`include_face=True`): malla `Face` filtrada únicamente al material
  `Face_00_SKIN` y cerrada con `holes_fill` (la malla `Face` cruda contiene múltiples subcapas y
  1035 aristas de borde abiertas; al aislar `Face_00_SKIN` y tapar sus 296 aristas de borde queda
  cerrada a 0 aristas abiertas para evaluar paridad de 5 rayos de forma fiable).
- Regiones NO cubiertas automáticamente por este módulo:
  * `Hair001` (cabello).
  * Subcapas internas/superficiales de ojos y boca en `Face` (`FaceMouth`, `Eye*`, `FaceBrow`,
    `FaceEyelash`, `FaceEyeline`).
  * Borde superior del cuello de la capucha por encima de `UpperChest.tail` (`tc > 1.03`),
    aunque puede medirse por separado como proximidad al cuello (`neck_collar`).
  * Pantalones y calzado (`Bottoms_01_CLOTH`, `Shoes_01_CLOTH`).
  * Autocolisión entre dedos/palma de la misma mano y colisión física tela-contra-tela
    (`R_SLV` vs `L_SLV`, `SLV` vs `TORSO`).

Unidades y convenciones:
- Posiciones y holguras internas en metros (`m`) en coordenadas de mundo de Blender (`-Y` frente,
  `+Z` arriba); los reportes convierten a milímetros (`mm`).
- Distinción en auditoría:
  * Penetración geométrica: distancia firmada `< 0.0 mm` (vértice dentro de la malla).
  * Contacto superficial legítimo: `0.0 mm <= dist < contact_margin` (fuera, apoyado o rasante).
  * Margen preventivo violado: `0.0 mm <= dist < safety_margin` (invade el colchón, sin penetrar).
"""
import math
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


def _precompute_face_skin_polys(face_obj):
    """Aísla polígonos `Face_00_SKIN` y rellena bordes abiertos manteniendo índices originales."""
    mesh = face_obj.data
    skin_indices = {
        i for i, sl in enumerate(face_obj.material_slots)
        if sl.material and "Face_00_SKIN" in sl.material.name
    }
    if not skin_indices:
        skin_indices = {
            i for i, sl in enumerate(face_obj.material_slots)
            if sl.material and "SKIN" in sl.material.name
        }
    poly_zones = ["FACE" if p.material_index in skin_indices else None for p in mesh.polygons]
    polys = _precompute_zone_polys(mesh, poly_zones, "FACE", cap_holes=True)

    # Verifica estanqueidad residual sobre la lista cerrada para documentar fiabilidad.
    bm = bmesh.new()
    bm.from_mesh(mesh)
    to_del = [fc for i, fc in enumerate(bm.faces) if poly_zones[i] != "FACE"]
    bmesh.ops.delete(bm, geom=to_del, context="FACES")
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bedges = [e for e in bm.edges if e.is_boundary]
    if bedges:
        bmesh.ops.holes_fill(bm, edges=bedges, sides=0)
    open_after = sum(1 for e in bm.edges if e.is_boundary)
    bm.free()
    return polys, open_after


def _parity_inside(pt, bvh):
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
    return inside >= 3


class ClothCollider:
    """Detector de penetración piel-ropa (y rostro opcional) sobre mallas deformadas."""

    def __init__(self, arm, body_obj=None, face_obj=None, include_face=False):
        self.arm = arm
        self.body = body_obj or bpy.data.objects.get("Body")
        if self.body is None:
            raise RuntimeError("No se encontró el objeto de malla 'Body' en la escena.")
        self.face = face_obj or (bpy.data.objects.get("Face") if include_face else None)

        mesh = self.body.data
        vg_names = {g.index: g.name for g in self.body.vertex_groups}

        # Vértices reales de piel de cada mano y cada dedo (evita aproximar radios).
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

        # Clasificación por polígono de Body: buzo y torso/brazos (excluye piel de manos).
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

        # Zona frontal del cuello del buzo (para auditoría de proximidad al cuello).
        top_mats = {
            i for i, sl in enumerate(self.body.material_slots)
            if sl.material and "Tops" in sl.material.name
        }
        neck_z = (self.arm.matrix_world @ self.arm.pose.bones["J_Bip_C_UpperChest"].tail).z - 0.05
        self.collar_polys = [
            tuple(p.vertices) for p in mesh.polygons
            if p.material_index in top_mats and any(
                (self.body.matrix_world @ mesh.vertices[vi].co).z >= neck_z
                and (self.body.matrix_world @ mesh.vertices[vi].co).y < -0.01
                and abs((self.body.matrix_world @ mesh.vertices[vi].co).x) < 0.14
                for vi in p.vertices
            )
        ]

        self.face_open_edges = None
        if self.face is not None:
            fpolys, open_after = _precompute_face_skin_polys(self.face)
            self.zone_polys["FACE"] = fpolys
            self.face_open_edges = open_after

    def slv_collar_min_dist(self, side, snap=None, stride=2):
        """Distancia mínima (m) entre los vértices del antebrazo/puño de manga y el cuello frontal."""
        snap = snap or self.snapshot()
        cbvh = snap.get("collar_bvh")
        if cbvh is None:
            return None
        mesh = self.body.data
        vg = {g.index: g.name for g in self.body.vertex_groups}
        vw = snap["verts_w"]
        best = None
        slv_vis = sorted({vi for poly in self.zone_polys[f"{side}_SLV"] for vi in poly})[::stride]
        for vi in slv_vis:
            dom = max(mesh.vertices[vi].groups, key=lambda g: g.weight, default=None)
            if not dom or f"J_Bip_{side}_LowerArm" not in vg.get(dom.group, ""):
                continue
            loc, _, _, dist = cbvh.find_nearest(vw[vi])
            if loc is not None and (best is None or dist < best):
                best = dist
        return best

    def eval_verts_world(self, obj=None):
        target = obj or self.body
        deps = bpy.context.evaluated_depsgraph_get()
        deps.update()
        ev = target.evaluated_get(deps)
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
        verts_w = self.eval_verts_world(self.body)
        if reuse_bvhs is not None:
            return {
                "verts_w": verts_w,
                "bvhs": reuse_bvhs["bvhs"],
                "spine": reuse_bvhs["spine"],
                "collar_bvh": reuse_bvhs.get("collar_bvh"),
            }

        bvhs = {
            z: BVHTree.FromPolygons(verts_w, self.zone_polys[z], all_triangles=False)
            for z in ("TORSO", "R_SLV", "L_SLV")
        }
        if self.face is not None and "FACE" in self.zone_polys:
            fverts_w = self.eval_verts_world(self.face)
            bvhs["FACE"] = BVHTree.FromPolygons(fverts_w, self.zone_polys["FACE"], all_triangles=False)

        collar_bvh = (
            BVHTree.FromPolygons(verts_w, self.collar_polys, all_triangles=False)
            if self.collar_polys else None
        )
        P = self.arm.pose.bones
        spine_a = self.arm.matrix_world @ P["J_Bip_C_Hips"].head
        spine_b = self.arm.matrix_world @ P["J_Bip_C_UpperChest"].tail
        return {"verts_w": verts_w, "bvhs": bvhs, "spine": (spine_a, spine_b), "collar_bvh": collar_bvh}

    def _signed_zone_distance(self, pt, snap, zone):
        """Distancia firmada (`< 0` dentro) y vector de empuje para una zona de `snap`."""
        bvhs = snap["bvhs"]
        if zone not in bvhs:
            return None, None

        if zone == "TORSO":
            spine_a, spine_b = snap["spine"]
            ori, tc = _seg_closest(pt, spine_a, spine_b)
            if not (-0.05 <= tc <= 1.03):
                return None, None
            ori_z = Vector((ori.x, ori.y, pt.z))
            d = pt - ori_z
            if d.length <= 1e-4:
                return None, None
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
            if max_r <= 0.0:
                return None, None
            return (d.length - max_r), dn

        bvh = bvhs[zone]
        loc, nor, _, dist = bvh.find_nearest(pt)
        max_d = 0.07 if zone != "FACE" else 0.055
        if loc is None or dist > max_d:
            return None, None
        inside = _parity_inside(pt, bvh)
        sd = -dist if inside else dist
        push = (loc - pt).normalized() if inside and dist > 1e-5 else nor.normalized()
        if inside and nor.dot(push) < 0:
            push = nor.normalized()
        return sd, push

    def _point_clearance(self, pt, snap, check_zones, margin=0.0, zone_margins=None):
        """Devuelve [(clearance_m, zone, push_vec)] con valores negativos si viola `margin`."""
        hits = []
        zm = zone_margins or {}
        for z in check_zones:
            sd, push = self._signed_zone_distance(pt, snap, z)
            if sd is None:
                continue
            m = zm.get(z, margin)
            clr = sd - m
            if clr < 0.0:
                hits.append((clr, z, push))
        return hits

    def inspect_side(self, side, snap=None, margin=0.0, stride=1, zones=None, zone_margins=None):
        """Lista de (clearance_m, zone, part, push_vec) para toda la mano de `side`."""
        snap = snap or self.snapshot()
        opp = "L_SLV" if side == "R" else "R_SLV"
        check_zones = zones or ("TORSO", opp)
        vw = snap["verts_w"]
        bad = []
        for vi, part in self.hand_verts[side][::stride]:
            for clr, z, push in self._point_clearance(vw[vi], snap, check_zones, margin=margin, zone_margins=zone_margins):
                bad.append((clr, z, part, push))
        bad.sort(key=lambda x: x[0])
        return bad

    def inspect_side_bands(self, side, snap=None, safety_margin=0.0025, contact_tol=0.0025, stride=1, zones=None):
        """Distingue penetración (`sd < 0`), contacto (`0 <= sd < contact_tol`) y margen (`0 <= sd < safety_margin`)."""
        snap = snap or self.snapshot()
        opp = "L_SLV" if side == "R" else "R_SLV"
        check_zones = zones or ("TORSO", opp)
        vw = snap["verts_w"]
        pen = {}
        contact = {}
        preventive = {}
        min_signed = None

        for vi, part in self.hand_verts[side][::stride]:
            for z in check_zones:
                sd, _ = self._signed_zone_distance(vw[vi], snap, z)
                if sd is None:
                    continue
                if min_signed is None or sd < min_signed[0]:
                    min_signed = (sd, f"{z}:{part}")
                key = f"{z}:{part}"
                if sd < 0.0:
                    if key not in pen or sd < pen[key]:
                        pen[key] = sd
                else:
                    if sd < contact_tol and (key not in contact or sd < contact[key]):
                        contact[key] = sd
                    if safety_margin > 0.0 and sd < safety_margin and (key not in preventive or sd < preventive[key]):
                        preventive[key] = sd

        return {
            "min_signed": min_signed,
            "penetrations": sorted((v, k) for k, v in pen.items()),
            "contacts": sorted((v, k) for k, v in contact.items()),
            "preventive": sorted((v, k) for k, v in preventive.items()),
        }

    def collar_min_dist(self, side, snap=None, stride=1):
        """Distancia mínima (m) de la mano/muñeca `side` al borde frontal del cuello del buzo."""
        snap = snap or self.snapshot()
        cbvh = snap.get("collar_bvh")
        if cbvh is None:
            return None
        vw = snap["verts_w"]
        best = None
        for vi, part in self.hand_verts[side][::stride]:
            loc, _, _, dist = cbvh.find_nearest(vw[vi])
            if loc is not None and (best is None or dist < best[0]):
                best = (dist, part)
        return best

    def evaluate_contact_regions(self, side, target_zone, snap=None, contact_max_mm=4.5, pen_tol_mm=-0.01):
        """Evalúa regiones anatómicas de la mano frente a `target_zone` (penetración / contacto / separación)."""
        snap = snap or self.snapshot()
        bvh = snap["bvhs"].get(target_zone)
        if bvh is None:
            return None
        P = self.arm.pose.bones
        hand_pb = P[f"J_Bip_{side}_Hand"]
        low_pb = P[f"J_Bip_{side}_LowerArm"]
        palm_n = _palm_normal(self.arm, side)
        hand_origin = self.arm.matrix_world @ ((hand_pb.head + hand_pb.tail) * 0.5)
        fdir = (self.arm.matrix_world @ low_pb.tail - self.arm.matrix_world @ low_pb.head).normalized()
        hdir = (self.arm.matrix_world @ hand_pb.tail - self.arm.matrix_world @ hand_pb.head).normalized()
        wrist_bend_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, fdir.dot(hdir))))), 1)

        vw = snap["verts_w"]
        regions = {
            "Palma": [],
            "DorsoHand": [],
            "BaseDedos": [],
            "FalangesInternas": [],
            "YemaPulgar": [],
            "YemaIndice": [],
        }
        all_signed = []
        for vi, part in self.hand_verts[side]:
            pt = vw[vi]
            sd, _ = self._signed_zone_distance(pt, snap, target_zone)
            if sd is None:
                loc, nor, _, dist = bvh.find_nearest(pt)
                if loc is None:
                    continue
                sd_mm = dist * 1000.0
                surf_nor = nor.normalized()
            else:
                sd_mm = sd * 1000.0
                _, nor, _, _ = bvh.find_nearest(pt)
                surf_nor = nor.normalized() if nor is not None else Vector((0.0, -1.0, 0.0))

            all_signed.append((sd_mm, part, surf_nor, pt))
            side_dot = (pt - hand_origin).dot(palm_n)
            entry = (sd_mm, surf_nor, pt)
            if part == "Hand":
                (regions["Palma"] if side_dot >= -0.003 else regions["DorsoHand"]).append(entry)
            elif part in ("Index1", "Middle1", "Ring1", "Little1"):
                regions["BaseDedos"].append(entry)
            elif part in ("Index2", "Middle2", "Middle3", "Ring2", "Ring3", "Little2", "Little3"):
                regions["FalangesInternas"].append(entry)
            elif part in ("Thumb2", "Thumb3"):
                regions["YemaPulgar"].append(entry)
            if part == "Index3":
                regions["YemaIndice"].append(entry)

        out_regions = {}
        for rname, entries in regions.items():
            if not entries:
                continue
            entries.sort(key=lambda e: e[0])
            sds = [e[0] for e in entries]
            n = len(sds)
            best_sd, best_nor, _ = entries[0]
            near_cnt = sum(1 for v in sds if pen_tol_mm <= v <= contact_max_mm)
            pen_cnt = sum(1 for v in sds if v < pen_tol_mm)
            palm_face_dot = round(palm_n.dot(best_nor), 3)
            if best_sd < pen_tol_mm:
                state = "penetracion"
            elif best_sd <= contact_max_mm:
                state = "contacto"
            else:
                state = "separacion"
            out_regions[rname] = {
                "estado": state,
                "verts": n,
                "min_mm": round(best_sd, 2),
                "p25_mm": round(sds[max(0, int((n - 1) * 0.25))], 2),
                "mediana_mm": round(sds[n // 2], 2),
                "p90_mm": round(sds[min(n - 1, int((n - 1) * 0.90))], 2),
                "verts_en_contacto": near_cnt,
                "verts_penetrando": pen_cnt,
                "palm_dot_surface_normal": palm_face_dot,
            }
        return {
            "side": side,
            "target_zone": target_zone,
            "wrist_bend_deg": wrist_bend_deg,
            "palm_normal": [round(c, 3) for c in palm_n],
            "forearm_dir": [round(c, 3) for c in fdir],
            "hand_dir": [round(c, 3) for c in hdir],
            "regions": out_regions,
        }

    def worst_by_part(self, side, snap=None, margin=0.0, stride=2, zones=None):
        """Resumen `[(clearance_m, 'ZONA:Parte'), ...]` ordenado de peor a mejor."""
        worst = {}
        for clr, z, part, _ in self.inspect_side(side, snap=snap, margin=margin, stride=stride, zones=zones):
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


def _uncurl_finger_outward(collider, arm, side, fname, opp_zones, base_snap, margin, max_steps=8, stride=3, zone_margins=None):
    """Abre de a poco un dedo que penetra la tela o superficie objetivo hasta dejarlo libre."""
    P = arm.pose.bones
    b1, b2, b3 = (P[f"J_Bip_{side}_{fname}{j}"] for j in (1, 2, 3))
    inv3 = arm.matrix_world.inverted().to_3x3()
    fverts = collider.finger_verts[side][fname][::stride]

    vw0 = base_snap["verts_w"]
    if not any(collider._point_clearance(vw0[vi], base_snap, opp_zones, margin=margin, zone_margins=zone_margins) for vi, _ in fverts):
        return False

    moved = False
    for _ in range(max_steps):
        snap = collider.snapshot(reuse_bvhs=base_snap)
        vw = snap["verts_w"]
        worst_clr = 0.0
        worst_push = None
        for vi, _ in fverts:
            for clr, _, push in collider._point_clearance(vw[vi], snap, opp_zones, margin=margin, zone_margins=zone_margins):
                if clr < worst_clr:
                    worst_clr = clr
                    worst_push = push
        if worst_push is None:
            return moved

        moved = True
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
    return moved


def _side_bone_names(side):
    names = [f"J_Bip_{side}_Shoulder", f"J_Bip_{side}_UpperArm", f"J_Bip_{side}_LowerArm", f"J_Bip_{side}_Hand"]
    for fn in FINGER_NAMES:
        for j in (1, 2, 3):
            names.append(f"J_Bip_{side}_{fn}{j}")
    return names


def resolve_arm_pose(collider, arm, side, pose_arm_fn, chest_delta, spec,
                     margin=0.0025, allow_uncurl=True, init_quats=None,
                     extra_zones=(), adjust_wrist=True, solver_stride=4,
                     zone_margins=None, lock_lowerarm_on_zones=()):
    """Posa un brazo y empuja muñeca/mano/dedos fuera de la malla deformada."""
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
    opp_zones = ("TORSO", "L_SLV" if side == "R" else "R_SLV", *extra_zones)
    wrist_zones = tuple(z for z in opp_zones if z not in lock_lowerarm_on_zones) or opp_zones

    sh_head = arm.matrix_world @ P[f"J_Bip_{side}_UpperArm"].head
    max_reach = 0.435
    check_verts = (
        collider.wrist_verts[side][::solver_stride]
        if allow_uncurl else [vi for vi, _ in collider.hand_verts[side][::solver_stride]]
    )

    if adjust_wrist:
        for _ in range(8):
            snap = collider.snapshot()
            vw = snap["verts_w"]
            worst_clr = 0.0
            push_sum = Vector((0.0, 0.0, 0.0))
            cnt = 0
            for vi in check_verts:
                for clr, _, push in collider._point_clearance(vw[vi], snap, wrist_zones, margin=margin, zone_margins=zone_margins):
                    if clr < 0.0:
                        push_sum += push * (-clr)
                        cnt += 1
                        if clr < worst_clr:
                            worst_clr = clr
            if cnt == 0 or push_sum.length < 1e-5:
                break
            dir_w = push_sum.normalized()
            pad = 0.0004 if zone_margins is not None else 0.002
            shift = min(0.028, (-worst_clr) * 0.95 + pad)
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

    base_snap = collider.snapshot()
    if allow_uncurl:
        for fn in FINGER_NAMES:
            _uncurl_finger_outward(
                collider, arm, side, fn, opp_zones, base_snap,
                margin=margin, stride=max(2, solver_stride - 1), zone_margins=zone_margins,
            )

    hand_pb = P[f"J_Bip_{side}_Hand"]
    inv3 = arm.matrix_world.inverted().to_3x3()
    for _ in range(5):
        snap = collider.snapshot(reuse_bvhs=base_snap)
        bad = collider.inspect_side(side, snap=snap, margin=margin, stride=max(1, solver_stride // 2), zones=opp_zones, zone_margins=zone_margins)
        if not bad:
            break
        worst_clr, hit_zone, _, push = bad[0]
        if hit_zone in lock_lowerarm_on_zones and worst_clr >= -0.0005:
            break
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


def resolve_both_arms(collider, arm, pose_arm_fn, side_jobs, passes=2):
    """Resuelve ambos brazos y revalida ambos lados cuando cualquiera se modifica."""
    for _p in range(passes):
        changed = False
        for job in side_jobs:
            side = job["side"]
            snap = collider.snapshot()
            opp_zones = ("TORSO", "L_SLV" if side == "R" else "R_SLV", *job.get("extra_zones", ()))
            margin = job.get("margin", 0.0025)
            zm = job.get("zone_margins")
            if _p > 0 and not collider.inspect_side(side, snap=snap, margin=margin, stride=2, zones=opp_zones, zone_margins=zm):
                continue
            changed = True
            resolve_arm_pose(
                collider, arm, side, pose_arm_fn,
                job["delta"], job["spec"],
                margin=margin,
                allow_uncurl=job.get("allow_uncurl", True),
                init_quats=job.get("init_quats"),
                extra_zones=job.get("extra_zones", ()),
                adjust_wrist=job.get("adjust_wrist", True),
                solver_stride=job.get("solver_stride", 4),
                zone_margins=zm,
                lock_lowerarm_on_zones=job.get("lock_lowerarm_on_zones", ()),
            )
        if not changed:
            break


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


def fix_interpolated_frames(collider, arm, frames, margin=0.0015, uncurl_sides=("R",),
                            solver_stride=2, passes=2, extra_zones_by_side=None,
                            lowerarm_sides=("L", "R"), hand_rot_sides=("L", "R"),
                            zone_margins_by_side=None):
    """Corrige e inserta keyframes en cuadros intermedios y revalida ambos brazos tras cambios."""
    scene = bpy.context.scene
    P = arm.pose.bones
    inv3 = arm.matrix_world.inverted().to_3x3()
    fixed_frames = []
    extra_map = extra_zones_by_side or {}
    zm_map = zone_margins_by_side or {}

    for fr in frames:
        scene.frame_set(fr)
        changed_any = False
        touched_all = {}
        for _pass in range(passes):
            pass_changed = False
            for side in ("L", "R"):
                opp_zones = ("TORSO", "L_SLV" if side == "R" else "R_SLV", *extra_map.get(side, ()))
                zm = zm_map.get(side)
                low_pb = P[f"J_Bip_{side}_LowerArm"]
                hand_pb = P[f"J_Bip_{side}_Hand"]

                for it in range(6):
                    cur_snap = collider.snapshot()
                    bad = collider.inspect_side(side, snap=cur_snap, margin=margin, stride=solver_stride, zones=opp_zones, zone_margins=zm)
                    if not bad:
                        break
                    changed_any = True
                    pass_changed = True
                    worst_clr, _, _, push = bad[0]
                    if side in uncurl_sides:
                        for fn in FINGER_NAMES:
                            if _uncurl_finger_outward(
                                collider, arm, side, fn, opp_zones, cur_snap,
                                margin=margin, max_steps=4, stride=max(2, solver_stride), zone_margins=zm,
                            ):
                                for j in (1, 2, 3):
                                    pb = P[f"J_Bip_{side}_{fn}{j}"]
                                    touched_all[pb.name] = pb
                        cur_snap = collider.snapshot()
                        bad = collider.inspect_side(side, snap=cur_snap, margin=margin, stride=solver_stride, zones=opp_zones, zone_margins=zm)
                        if not bad:
                            break
                        worst_clr, _, _, push = bad[0]

                    if side not in lowerarm_sides:
                        if side not in hand_rot_sides:
                            break
                        rot_pb = hand_pb
                        d_ax = (hand_pb.tail - hand_pb.head).normalized()
                    else:
                        rot_pb = low_pb
                        d_ax = (low_pb.tail - low_pb.head).normalized()
                    ax = d_ax.cross(inv3 @ push)
                    if ax.length > 1e-5:
                        ang = min(0.09, max(0.02, -worst_clr * 3.0))
                        before = arm.matrix_world @ rot_pb.tail
                        _rotate_about(rot_pb, Quaternion(ax.normalized(), ang), rot_pb.head.copy())
                        if (arm.matrix_world @ rot_pb.tail - before).dot(push) < 0:
                            _rotate_about(rot_pb, Quaternion(ax.normalized(), -2.0 * ang), rot_pb.head.copy())
                        touched_all[rot_pb.name] = rot_pb
                        touched_all[hand_pb.name] = hand_pb
                    elif it > 0:
                        break
            if not pass_changed:
                break

        if changed_any and touched_all:
            _batch_insert_aligned_keys(list(touched_all.values()), fr)
            fixed_frames.append(fr)
    return fixed_frames


def measure_continuity(arm, frames, track_bones=None):
    """Mide desplazamientos cuadro a cuadro (`mm`) para detectar discontinuidades tras correcciones."""
    scene = bpy.context.scene
    P = arm.pose.bones
    bones = track_bones or (
        "J_Bip_C_Head", "J_Bip_C_UpperChest",
        "J_Bip_R_LowerArm", "J_Bip_R_Hand", "J_Bip_R_Index3",
        "J_Bip_L_LowerArm", "J_Bip_L_Hand", "J_Bip_L_Index3",
    )
    prev = None
    history = {b: [] for b in bones}
    for fr in frames:
        scene.frame_set(fr)
        bpy.context.view_layer.update()
        cur = {}
        for bn in bones:
            pb = P[bn]
            cur[bn] = (arm.matrix_world @ (pb.tail if bn.endswith("3") else pb.head)).copy()
            if prev is not None:
                history[bn].append((fr, round((cur[bn] - prev[bn]).length * 1000.0, 2)))
        prev = cur

    summary = {}
    for bn, seq in history.items():
        if not seq:
            continue
        max_fr, max_mm = max(seq, key=lambda x: x[1])
        idx = next(i for i, (f, _) in enumerate(seq) if f == max_fr)
        neighbors = seq[max(0, idx - 2):min(len(seq), idx + 3)]
        summary[bn] = {"max_mm": max_mm, "at_frame": max_fr, "neighbors": neighbors}
    return summary


def audit_clip_detailed(collider, arm, frames, margin=0.0, stride=1, contact_margin=0.0025,
                        include_face=False):
    """Audita todos los cuadros diferenciando penetración geométrica (<0), contacto y margen preventivo."""
    scene = bpy.context.scene
    report = []
    details = []
    for f in frames:
        scene.frame_set(f)
        snap = collider.snapshot()
        for side in ("R", "L"):
            zones = ("TORSO", "L_SLV" if side == "R" else "R_SLV")
            if include_face and side == "R" and "FACE" in snap["bvhs"]:
                zones = (*zones, "FACE")
            bands = collider.inspect_side_bands(
                side, snap=snap, safety_margin=margin, contact_tol=contact_margin,
                stride=stride, zones=zones,
            )
            eff_pen = bands["penetrations"]
            eff_prev = bands["preventive"]
            if eff_pen:
                report.append((f, side, eff_pen[:4]))
            if eff_pen or eff_prev or bands["contacts"]:
                details.append({
                    "frame": f,
                    "side": side,
                    "penetrations": eff_pen[:6],
                    "contacts": bands["contacts"][:6],
                    "preventive": eff_prev[:6],
                    "min_signed": bands["min_signed"],
                })
    return {"penetration_report": report, "details": details}


def audit_clip(collider, arm, frames, margin=0.0, stride=1, include_face=False):
    """Recorre los cuadros indicados y devuelve solo las penetraciones geométricas (o < margin si margin>0)."""
    scene = bpy.context.scene
    report = []
    for f in frames:
        scene.frame_set(f)
        snap = collider.snapshot()
        for side in ("R", "L"):
            zones = ("TORSO", "L_SLV" if side == "R" else "R_SLV")
            if include_face and side == "R" and "FACE" in snap["bvhs"]:
                zones = (*zones, "FACE")
            worst = collider.worst_by_part(side, snap=snap, margin=margin, stride=stride, zones=zones)
            if worst:
                report.append((f, side, worst[:4]))
    return report
