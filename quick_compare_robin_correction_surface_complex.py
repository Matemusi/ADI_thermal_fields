# -*- coding: utf-8 -*-
"""Compare STL-informed voxel Robin corrections on complex geometries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import adi3d_numba_coeff as adi
from voxel_bc_correction import build_corrected_robin_fields


try:  # pragma: no cover - optional dependency during type checking
    import trimesh
except Exception as exc:  # pragma: no cover - handled at runtime in main()
    trimesh = None  # type: ignore[assignment]
    _TRIMESH_IMPORT_ERROR = exc
else:  # pragma: no cover - best-effort hint for static analysis
    _TRIMESH_IMPORT_ERROR = None


Face = str


@dataclass
class FaceAreaStats:
    """Container with per-face surface information."""

    base_area: float
    corrected_area: float
    actual_projected_area: float

    def ratios(self) -> Tuple[float, float]:
        if self.actual_projected_area <= 0.0:
            return float("nan"), float("nan")
        base_ratio = self.base_area / self.actual_projected_area
        corrected_ratio = self.corrected_area / self.actual_projected_area
        return base_ratio, corrected_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Voxelises a complex geometry, builds STL-informed Robin corrections,"
            " and compares the accumulated projected areas against the STL ground"
            " truth while sweeping multiple voxel sizes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--geometry",
        choices=("tilted_cone", "stl"),
        default="tilted_cone",
        help="Geometry used for comparison."
        " `tilted_cone` is a synthetic frustum-like cone rotated off-axis",
    )
    parser.add_argument(
        "--stl_path",
        type=str,
        default=None,
        help="Path to the STL mesh when --geometry=stl. If omitted, the script"
        " exits with an error.",
    )
    parser.add_argument("--height", type=float, default=0.09, help="Cone height in metres")
    parser.add_argument(
        "--radius_bottom",
        type=float,
        default=0.035,
        help="Bottom radius of the synthetic cone geometry, metres",
    )
    parser.add_argument(
        "--radius_top",
        type=float,
        default=0.015,
        help="Top radius of the synthetic cone geometry, metres",
    )
    parser.add_argument(
        "--tilt_deg",
        type=float,
        default=28.0,
        help="Tilt angle in degrees applied around the X axis for the synthetic geometry",
    )
    parser.add_argument(
        "--sections",
        type=int,
        default=128,
        help="Number of circumferential sections for generated meshes",
    )
    parser.add_argument(
        "--dx_mm_list",
        type=str,
        default="3.0,2.2,1.6,1.2,0.9",
        help="Comma-separated list of voxel pitches in millimetres",
    )
    parser.add_argument(
        "--faces",
        type=str,
        default="x-,x+,y-,y+,z-,z+",
        help="Comma-separated subset of voxel faces to include in statistics",
    )
    parser.add_argument(
        "--max_subdiv",
        type=int,
        default=24,
        help="Maximum triangle subdivision factor when projecting STL facets",
    )

    return parser.parse_args()


def ensure_trimesh_loaded() -> None:
    if trimesh is None:  # pragma: no cover - runtime guard
        raise ImportError(
            "trimesh is required for this comparison script"
        ) from _TRIMESH_IMPORT_ERROR


def create_tilted_cone(
    height: float,
    radius_bottom: float,
    radius_top: float,
    sections: int,
    tilt_deg: float,
) -> "trimesh.Trimesh":
    ensure_trimesh_loaded()

    n_sections = max(8, int(sections))
    if hasattr(trimesh.creation, "conical_frustum"):
        mesh = trimesh.creation.conical_frustum(
            radius_top=radius_top,
            radius_bottom=radius_bottom,
            height=height,
            sections=n_sections,
        )
    else:
        mesh = _build_frustum_manual(
            radius_bottom=radius_bottom,
            radius_top=radius_top,
            height=height,
            sections=n_sections,
        )
    # Align the frustum base with z=0 and rotate off-axis.
    mesh.apply_translation([0.0, 0.0, height / 2.0])
    rot = trimesh.transformations.rotation_matrix(np.deg2rad(tilt_deg), [1.0, 0.0, 0.0])
    mesh.apply_transform(rot)
    # Shift so the minimum bound sits at the origin to ease voxelisation bookkeeping.
    bounds_min = mesh.bounds[0]
    mesh.apply_translation(-bounds_min)
    return mesh


def _build_frustum_manual(
    radius_bottom: float,
    radius_top: float,
    height: float,
    sections: int,
) -> "trimesh.Trimesh":
    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    bottom = np.column_stack(
        (
            radius_bottom * np.cos(angles),
            radius_bottom * np.sin(angles),
            np.zeros_like(angles),
        )
    )
    top = np.column_stack(
        (
            radius_top * np.cos(angles),
            radius_top * np.sin(angles),
            np.full_like(angles, height),
        )
    )
    vertices = np.vstack((bottom, top))

    faces: List[Tuple[int, int, int]] = []
    for i in range(sections):
        j = (i + 1) % sections
        # Side faces (two triangles per quad)
        faces.append((i, j, sections + i))
        faces.append((j, sections + j, sections + i))
    # Caps using fan triangulation with outward-facing normals.
    bottom_center_idx = len(vertices)
    top_center_idx = bottom_center_idx + 1
    vertices = np.vstack(
        (
            vertices,
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, height]], dtype=float),
        )
    )
    for i in range(sections):
        j = (i + 1) % sections
        # Bottom cap (normal pointing downward)
        faces.append((bottom_center_idx, j, i))
        # Top cap (normal pointing upward)
        faces.append((top_center_idx, sections + i, sections + j))
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=True)
    return mesh


def load_mesh_from_args(args: argparse.Namespace) -> "trimesh.Trimesh":
    ensure_trimesh_loaded()

    if args.geometry == "tilted_cone":
        return create_tilted_cone(
            height=args.height,
            radius_bottom=args.radius_bottom,
            radius_top=args.radius_top,
            sections=args.sections,
            tilt_deg=args.tilt_deg,
        )

    if args.geometry == "stl":
        if not args.stl_path:
            raise SystemExit("--stl_path must be provided when --geometry=stl")
        mesh = trimesh.load(args.stl_path, force="mesh")
        if getattr(mesh, "units", None) is None or str(mesh.units).lower() == "mm":
            mesh = mesh.copy()
            mesh.apply_scale(1e-3)
        bounds_min = mesh.bounds[0]
        mesh.apply_translation(-bounds_min)
        return mesh

    raise SystemExit(f"Unsupported geometry kind: {args.geometry}")


def parse_dx_list(values: str) -> List[float]:
    out: List[float] = []
    for chunk in values.split(","):
        val = chunk.strip()
        if not val:
            continue
        out.append(float(val) * 1e-3)
    if not out:
        raise SystemExit("--dx_mm_list must contain at least one positive value")
    return out


def parse_faces(values: str) -> List[Face]:
    faces = [face.strip() for face in values.split(",") if face.strip()]
    valid = {"x-", "x+", "y-", "y+", "z-", "z+"}
    for face in faces:
        if face not in valid:
            raise SystemExit(f"Unknown face '{face}'. Valid options: {sorted(valid)}")
    if not faces:
        raise SystemExit("At least one face must be specified")
    return faces


def voxelize_mesh(mesh: "trimesh.Trimesh", dx: float) -> Tuple[np.ndarray, np.ndarray]:
    vg = mesh.voxelized(pitch=dx)
    vg = vg.fill()
    mask = np.array(vg.matrix, dtype=bool, order="C")
    if vg.points.size > 0:
        min_center = np.min(vg.points, axis=0)
    else:
        min_center = mesh.bounds[0]
    origin = np.asarray(min_center - 0.5 * dx, dtype=float)
    return mask, origin


def compute_actual_projected_areas(
    mesh: "trimesh.Trimesh", faces: Sequence[Face]
) -> Dict[Face, float]:
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    results: Dict[Face, float] = {face: 0.0 for face in faces}
    axis_map = {
        "x-": (-1, 0),
        "x+": (1, 0),
        "y-": (-1, 1),
        "y+": (1, 1),
        "z-": (-1, 2),
        "z+": (1, 2),
    }
    for face in faces:
        sign, axis = axis_map[face]
        comp = normals[:, axis] * sign
        comp = np.clip(comp, 0.0, None)
        results[face] = float(np.sum(areas * comp))
    return results


def gather_face_stats(
    mesh: "trimesh.Trimesh",
    mask: np.ndarray,
    origin: np.ndarray,
    dx: float,
    faces: Sequence[Face],
    max_subdiv: int,
) -> Tuple[Dict[Face, FaceAreaStats], float]:
    base_h: Mapping[Face, float] = {face: 1.0 for face in faces}
    _, scale_fields = build_corrected_robin_fields(
        mesh,
        mask,
        origin=origin,
        dx=dx,
        base_h=base_h,
        fallback_to_base=False,
        max_subdiv=max(1, int(max_subdiv)),
    )

    actual_proj = compute_actual_projected_areas(mesh, faces)

    stats: Dict[Face, FaceAreaStats] = {}
    face_area = dx * dx
    for face in faces:
        exp = adi.exposed_mask(mask, face)
        base_area = float(np.count_nonzero(exp) * face_area)
        corrected_area = 0.0
        if face in scale_fields:
            corrected_area = float(np.sum(scale_fields[face][exp]) * face_area)
        stats[face] = FaceAreaStats(
            base_area=base_area,
            corrected_area=corrected_area,
            actual_projected_area=actual_proj.get(face, 0.0),
        )

    return stats, float(mesh.area)


def format_area(value: float) -> str:
    return f"{value:9.6f}"


def format_ratio(value: float) -> str:
    if not np.isfinite(value):
        return "   nan"
    return f"{value:6.3f}"


def main(args: argparse.Namespace) -> None:
    mesh = load_mesh_from_args(args)
    faces = parse_faces(args.faces)
    dx_values = parse_dx_list(args.dx_mm_list)

    print("# STL-based voxel Robin correction area comparison")
    if args.geometry == "tilted_cone":
        print(
            f"Geometry: tilted conical frustum (height={args.height:.3f} m, "
            f"R_bot={args.radius_bottom:.3f} m, R_top={args.radius_top:.3f} m, "
            f"tilt={args.tilt_deg:.1f}°)"
        )
    else:
        print(f"Geometry: STL file '{args.stl_path}'")
    print("Faces considered:", ", ".join(faces))
    print()

    header = ["dx_mm", "voxels", "surf_area_m2"]
    for face in faces:
        header.extend(
            [
                f"{face}:base",
                f"{face}:corr",
                f"{face}:actual",
                f"{face}:bas/act",
                f"{face}:cor/act",
            ]
        )
    header.append("sum_base")
    header.append("sum_corr")
    header.append("sum_actual")
    header.append("corr/actual")
    print(" ".join(f"{h:>11s}" for h in header))

    for dx in dx_values:
        mask, origin = voxelize_mesh(mesh, dx)
        stats, surface_area = gather_face_stats(
            mesh=mesh,
            mask=mask,
            origin=origin,
            dx=dx,
            faces=faces,
            max_subdiv=args.max_subdiv,
        )

        shape = mask.shape
        total_base = sum(s.base_area for s in stats.values())
        total_corr = sum(s.corrected_area for s in stats.values())
        total_actual = sum(s.actual_projected_area for s in stats.values())

        row: List[str] = [f"{dx * 1e3:6.3f}", f"{shape}", f"{surface_area:10.6f}"]
        for face in faces:
            s = stats[face]
            bas_ratio, cor_ratio = s.ratios()
            row.extend(
                [
                    format_area(s.base_area),
                    format_area(s.corrected_area),
                    format_area(s.actual_projected_area),
                    format_ratio(bas_ratio),
                    format_ratio(cor_ratio),
                ]
            )
        corr_vs_actual = total_corr / total_actual if total_actual > 0 else float("nan")
        row.extend(
            [
                format_area(total_base),
                format_area(total_corr),
                format_area(total_actual),
                format_ratio(corr_vs_actual),
            ]
        )
        print(" ".join(f"{val:>11s}" for val in row))


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)

