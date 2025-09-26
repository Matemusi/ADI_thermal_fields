
# -*- coding: utf-8 -*-
import numpy as np

def _write_scalars(f, name, arr_flat):
    f.write(f"SCALARS {name} float 1\n")
    f.write("LOOKUP_TABLE default\n")
    for i in range(0, arr_flat.size, 9):
        chunk = arr_flat[i:i+9]
        f.write(" ".join(f"{float(v):.6e}" for v in chunk) + "\n")

def write_vtk_structured_points(path, T, dx, origin=(0.0,0.0,0.0), field_name="Temperature", mask=None):
    T = np.asarray(T)
    nx,ny,nz = T.shape
    ox,oy,oz = origin
    origin_center = (ox+dx*0.5, oy+dx*0.5, oz+dx*0.5)
    T_flat = T.reshape(-1, order='F')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Uniform grid with Temperature and mask\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {origin_center[0]:.9e} {origin_center[1]:.9e} {origin_center[2]:.9e}\n")
        f.write(f"SPACING {dx:.9e} {dx:.9e} {dx:.9e}\n")
        f.write(f"POINT_DATA {nx*ny*nz}\n")
        _write_scalars(f, field_name, T_flat)
        if mask is not None:
            mask_flat = np.asarray(mask, dtype=np.float32).reshape(-1, order='F')
            _write_scalars(f, "mask", mask_flat)
