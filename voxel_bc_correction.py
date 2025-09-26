# -*- coding: utf-8 -*-
"""Utilities for correcting voxel Robin coefficients using STL geometry."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np

from adi3d_numba_coeff import exposed_mask

FaceName = str
VoxelIndex = Tuple[int, int, int]


class VoxelBoundaryData:
    """Stores projected surface areas that contribute to voxel faces."""

    __slots__ = ("voxel_index", "projected_area")

    def __init__(self, voxel_index: VoxelIndex):
        self.voxel_index = voxel_index
        self.projected_area: MutableMapping[FaceName, float] = defaultdict(float)

    def add_projected_area(self, face: FaceName, area: float) -> None:
        if area <= 0.0:
            return
        self.projected_area[face] += float(area)


class STLBoundaryCorrector:
    """Compute voxel-face projected areas from an STL mesh."""

    def __init__(
        self,
        mesh,
        mask: np.ndarray,
        origin: Iterable[float],
        dx: float,
        max_subdiv: int = 6,
        area_epsilon: float = 1e-16,
    ) -> None:
        self.mesh = mesh
        self.mask = np.asarray(mask, dtype=bool)
        self.origin = np.asarray(origin, dtype=float)
        self.dx = float(dx)
        self.shape = self.mask.shape
        self.max_subdiv = max(1, int(max_subdiv))
        self.area_epsilon = float(area_epsilon)

    def compute_voxel_projected_areas(self) -> Dict[VoxelIndex, VoxelBoundaryData]:
        """Return projected areas per voxel face derived from the STL mesh."""

        centers = np.asarray(self.mesh.triangles_center, dtype=float)
        normals = np.asarray(self.mesh.face_normals, dtype=float)
        areas = np.asarray(self.mesh.area_faces, dtype=float)
        vertices = np.asarray(self.mesh.triangles, dtype=float)

        data: Dict[VoxelIndex, VoxelBoundaryData] = {}
        for tri_idx in range(len(vertices)):
            area = float(areas[tri_idx])
            if area <= self.area_epsilon:
                continue

            tri_vertices = vertices[tri_idx]
            normal = normals[tri_idx]
            # Determine a reasonable subdivision so the triangle footprint fits into voxels.
            bbox_min = tri_vertices.min(axis=0)
            bbox_max = tri_vertices.max(axis=0)
            span = (bbox_max - bbox_min) / self.dx
            span_max = float(np.max(span))
            n_sub = int(math.ceil(span_max)) if span_max > 1.0 else 1
            n_sub = max(1, min(n_sub, self.max_subdiv))

            if n_sub == 1:
                sub_tris = (tri_vertices,)
                sub_area = area
            else:
                sub_tris = _subdivide_triangle(tri_vertices, n_sub)
                sub_area = area / (n_sub * n_sub)

            for sub_tri in sub_tris:
                centroid = np.mean(sub_tri, axis=0)

                voxel_idx = np.floor((centroid - self.origin) / self.dx)
                if np.any(voxel_idx < -1) or np.any(voxel_idx > (np.array(self.shape) + 1)):
                    continue
                voxel_idx = np.clip(voxel_idx, 0, np.array(self.shape) - 1).astype(int)
                
                voxel_idx = np.floor((centroid - self.origin) / self.dx).astype(int)
                if np.any(voxel_idx < 0) or np.any(voxel_idx >= self.shape):
                    continue


                idx_tuple = (int(voxel_idx[0]), int(voxel_idx[1]), int(voxel_idx[2]))
                if not self.mask[idx_tuple]:
                    continue

                vdata = data.get(idx_tuple)
                if vdata is None:
                    vdata = VoxelBoundaryData(idx_tuple)
                    data[idx_tuple] = vdata

                _accumulate_face_projection(vdata, normal, sub_area)

        return data

    def build_corrected_fields(
        self,
        base_h: Mapping[FaceName, float],
        fallback_to_base: bool = True,
    ) -> Tuple[Dict[FaceName, np.ndarray], Dict[FaceName, np.ndarray]]:
        """Create corrected robin_h arrays and area scaling fields.

        Parameters
        ----------
        base_h:
            Mapping from face name (e.g. 'x-', 'y+', 'z-') to base convective
            coefficient in W/m^2/K.
        fallback_to_base:
            If True, ensure that every exposed voxel face receives at least the
            base coefficient (useful when mesh discretisation misses cells).

        Returns
        -------
        robin_h_fields, area_scale_fields
            Dictionaries keyed by face names containing corrected fields.
        """

        projected = self.compute_voxel_projected_areas()
        shape = self.shape
        face_area = self.dx * self.dx

        robin_fields: Dict[FaceName, np.ndarray] = {}
        scale_fields: Dict[FaceName, np.ndarray] = {}

        for face, base_val in base_h.items():
            arr = np.zeros(shape, dtype=np.float64)
            scl = np.zeros(shape, dtype=np.float64)
            robin_fields[face] = arr
            scale_fields[face] = scl

        for idx, vdata in projected.items():
            for face, proj_area in vdata.projected_area.items():
                if face not in base_h:
                    continue
                base_val = float(base_h[face])
                if base_val == 0.0:
                    continue
                scale = proj_area / face_area
                robin_fields[face][idx] += base_val * scale
                scale_fields[face][idx] += scale

        if fallback_to_base:
            for face, base_val in base_h.items():
                if base_val == 0.0:
                    continue
                exp = exposed_mask(self.mask, face)
                arr = robin_fields[face]
                missing = exp & (arr <= 0.0)
                arr[missing] = float(base_val)
                scl = scale_fields[face]
                scl[missing] = 1.0

        return robin_fields, scale_fields


def _accumulate_face_projection(vdata: VoxelBoundaryData, normal: np.ndarray, area: float) -> None:
    tol = 1e-12
    comps = (
        (normal[0], "x-", "x+"),
        (normal[1], "y-", "y+"),
        (normal[2], "z-", "z+"),
    )
    abs_area = float(area)
    for comp, face_neg, face_pos in comps:
        if comp > tol:
            vdata.add_projected_area(face_pos, abs_area * comp)
        elif comp < -tol:
            vdata.add_projected_area(face_neg, abs_area * (-comp))


def _subdivide_triangle(vertices: np.ndarray, n: int) -> List[np.ndarray]:
    v0, v1, v2 = vertices

    def bary(i: int, j: int) -> np.ndarray:
        a = i / float(n)
        b = j / float(n)
        c = 1.0 - a - b
        return c * v0 + a * v1 + b * v2

    tris: List[np.ndarray] = []
    for i in range(n):
        for j in range(n - i):
            p0 = bary(i, j)
            p1 = bary(i + 1, j)
            p2 = bary(i, j + 1)
            tris.append(np.array((p0, p1, p2), dtype=float))
            if i + j < n - 1:
                p3 = bary(i + 1, j + 1)
                tris.append(np.array((p1, p3, p2), dtype=float))
    return tris


def build_corrected_robin_fields(
    mesh,
    mask: np.ndarray,
    origin: Iterable[float],
    dx: float,
    base_h: Mapping[FaceName, float],
    fallback_to_base: bool = True,
    max_subdiv: int = 6,
) -> Tuple[Dict[FaceName, np.ndarray], Dict[FaceName, np.ndarray]]:
    """Helper wrapper to create corrected Robin coefficient fields."""

    corrector = STLBoundaryCorrector(
        mesh=mesh,
        mask=mask,
        origin=origin,
        dx=dx,
        max_subdiv=max_subdiv,
    )
    return corrector.build_corrected_fields(base_h=base_h, fallback_to_base=fallback_to_base)

