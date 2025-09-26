# -*- coding: utf-8 -*-
"""Assess Robin area corrections across voxel resolutions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import trimesh

import adi3d_numba_coeff as adi
from voxel_bc_correction import build_corrected_robin_fields


Face = str


@dataclass
class FaceAreaStats:
    """Stores surface area information for a voxel face."""

    base_area: float
    corrected_area: float
    actual_proj_area: float

    def ratios(self) -> Tuple[float, float]:
        if self.actual_proj_area <= 0:
            return float("nan"), float("nan")
        return self.base_area / self.actual_proj_area, self.corrected_area / self.actual_proj_area


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compares per-face projected areas from voxel Robin corrections "
            "against constant coefficients and STL ground truth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--R", type=float, default=0.02, help="Cylinder radius, m")
    p.add_argument("--height", type=float, default=0.12, help="Cylinder height, m")
    p.add_argument(
        "--nxr_list",
        type=str,
        default="8,12,18,24,32",
        help="Comma separated list of radial voxel counts (R/dx).",
    )
    p.add_argument("--mesh_sections", type=int, default=128, help="Number of STL circumferential sections")
    p.add_argument("--max_subdiv", type=int, default=16, help="Maximum triangle subdivision per voxel span")
    p.add_argument(
        "--faces",
        type=str,
        default="z-",
        help="Comma separated list of faces to report (subset of x±, y±, z±)",
    )

    return p.parse_args()


def build_cylinder_mask(nx: int, ny: int, nz: int, dx: float, R: float) -> np.ndarray:
    cx = nx / 2.0
    cy = ny / 2.0
    xs = (np.arange(nx) + 0.5 - cx) * dx
    ys = (np.arange(ny) + 0.5 - cy) * dx
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    disk = np.sqrt(X ** 2 + Y ** 2) <= (R + 1e-12)
    return np.repeat(disk[:, :, None], nz, axis=2)


def compute_actual_projected_areas(mesh: trimesh.Trimesh, faces: Iterable[Face]) -> Dict[Face, float]:
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    out: Dict[Face, float] = {face: 0.0 for face in faces}
    axis_map = {
        "x-": (-1, 0),
        "x+": (1, 0),
        "y-": (-1, 1),
        "y+": (1, 1),
        "z-": (-1, 2),
        "z+": (1, 2),
    }
    for face, (sign, axis) in axis_map.items():
        if face not in out:
            continue
        component = normals[:, axis] * sign
        component = np.clip(component, 0.0, None)
        out[face] = float(np.sum(areas * component))
    return out


def compute_face_areas(
    mask: np.ndarray,
    dx: float,
    base_h: Mapping[Face, float],
    mesh: trimesh.Trimesh,
    faces: Iterable[Face],
    max_subdiv: int,
    radius: float,
) -> Dict[Face, FaceAreaStats]:
    origin = np.array([-radius, -radius, 0.0], dtype=float)
    _, scale_raw = build_corrected_robin_fields(
        mesh,
        mask,
        origin=origin,
        dx=dx,
        base_h=base_h,
        fallback_to_base=False,
        max_subdiv=max(1, max_subdiv),
    )

    stats: Dict[Face, FaceAreaStats] = {}
    actual_proj = compute_actual_projected_areas(mesh, faces)
    for face in faces:
        exp = adi.exposed_mask(mask, face)
        base_area = float(np.count_nonzero(exp) * dx * dx)
        corrected_area = 0.0
        if face in scale_raw:
            corrected_area = float(np.sum(scale_raw[face][exp]) * dx * dx)
        stats[face] = FaceAreaStats(
            base_area=base_area,
            corrected_area=corrected_area,
            actual_proj_area=actual_proj.get(face, 0.0),
        )
    return stats


def format_mm(value: float) -> str:
    return f"{value * 1e3:7.3f}"


def format_area(value: float) -> str:
    return f"{value:9.6f}"


def main(args: argparse.Namespace) -> None:
    faces = [f.strip() for f in args.faces.split(",") if f.strip()]
    if not faces:
        raise SystemExit("No faces specified")

    nxr_values = [int(v) for v in args.nxr_list.split(",") if v.strip()]
    if not nxr_values:
        raise SystemExit("nxr_list must contain at least one integer")

    print("# Robin projected area comparison")
    print(f"Radius={args.R:.4f} m, height={args.height:.4f} m")
    print("Faces considered:", ", ".join(faces))
    print()

    header = (
        "nxr  dx_mm  "
        + "  ".join(
            f"{face:^40s}" for face in faces
        )
    )
    print(header)
    print(
        "    "
        + "      "
        + "  ".join("base[m^2]  corr[m^2]  actual[m^2]  bas/act  cor/act" for _ in faces)
    )

    for nxr in nxr_values:
        dx = args.R / float(nxr)
        nx = ny = int(round((2.0 * args.R) / dx))
        nz = max(1, int(round(args.height / dx)))
        mask = build_cylinder_mask(nx, ny, nz, dx, args.R)

        cyl_mesh = trimesh.creation.cylinder(
            radius=args.R, height=nz * dx, sections=max(8, args.mesh_sections)
        )
        cyl_mesh.apply_translation([0.0, 0.0, nz * dx / 2.0])

        base_h = {face: 1.0 for face in faces}
        stats = compute_face_areas(
            mask,
            dx,
            base_h,
            cyl_mesh,
            faces,
            args.max_subdiv,
            radius=args.R,
        )

        row_parts: List[str] = [f"{nxr:3d}", format_mm(dx)]
        for face in faces:
            s = stats[face]
            bas_ratio, cor_ratio = s.ratios()
            row_parts.extend(
                [
                    format_area(s.base_area),
                    format_area(s.corrected_area),
                    format_area(s.actual_proj_area),
                    f"{bas_ratio:6.3f}",
                    f"{cor_ratio:6.3f}",
                ]
            )
        print(" ".join(row_parts))


if __name__ == "__main__":
    args = parse_args()
    main(args)

