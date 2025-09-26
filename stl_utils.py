
# -*- coding: utf-8 -*-
import numpy as np, math, multiprocessing, concurrent.futures

def load_stl_mesh(stl_path):
    try:
        import trimesh
    except Exception as e:
        raise ImportError("Need 'trimesh' to voxelize STL: pip install trimesh") from e
    mesh = trimesh.load(stl_path, force='mesh')
    if getattr(mesh, "units", None) is None or str(mesh.units).lower() == 'mm':
        mesh.apply_scale(1e-3)  # mm -> m
    return mesh

def load_voxel_from_stl(stl_path, dx_m, pad_mm=0.0, fill_solid=True):
    import numpy as np
    mesh = load_stl_mesh(stl_path)
    pad_m = pad_mm * 1e-3
    bmin, bmax = mesh.bounds
    bmin = bmin - pad_m; bmax = bmax + pad_m
    vg = mesh.voxelized(pitch=dx_m)
    if fill_solid:
        vg = vg.fill()
    mask = np.array(vg.matrix, dtype=bool, order='C')
    nx, ny, nz = mask.shape
    if len(vg.points) > 0:
        min_center = np.min(vg.points, axis=0)
        origin = (min_center - 0.5*dx_m).astype(float)
    else:
        origin = (bmin - 0.5*dx_m).astype(float)
    return mask, tuple(origin.tolist()), dx_m, (nx,ny,nz), mesh

def _section_perimeter_area(mesh, z):
    section = mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1.0])
    if section is None:
        return 0.0, 0.0
    planar, _ = section.to_planar()
    return float(planar.length), float(planar.area)

def per_slice_geom_from_stl(mesh, dz_m, nz, origin_z_m, parallel=True):
    z_levels = [origin_z_m + (k+0.5)*dz_m for k in range(nz)]
    per = np.zeros(nz, float); area = np.zeros(nz, float)
    if parallel and nz > 16:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as ex:
            for k, (p,a) in enumerate(ex.map(lambda z: _section_perimeter_area(mesh, z), z_levels)):
                per[k]=p; area[k]=a
    else:
        for k, z in enumerate(z_levels):
            p,a = _section_perimeter_area(mesh, z)
            per[k]=p; area[k]=a
    return per, area

def per_slice_scale_from_mesh_or_vox(mesh, mask, dx_m, origin, use_mesh=True, parallel=True):
    from adi3d_numba_coeff import exposed_mask
    nx,ny,nz = mask.shape
    dx = dx_m
    exp_faces = [exposed_mask(mask, f) for f in ("x-","x+","y-","y+")]
    voxel_area = np.zeros(nz, float)
    for k in range(nz):
        cnt = sum(np.count_nonzero(e[:,:,k]) for e in exp_faces)
        voxel_area[k] = cnt * (dx*dx)
    true_area = np.zeros(nz, float)
    if use_mesh and mesh is not None:
        try:
            per, _area = per_slice_geom_from_stl(mesh, dx, nz, origin[2], parallel=parallel)
            true_area = per * dx
        except Exception:
            true_area[:] = 0.0
    scale = np.ones(nz, float)
    for k in range(nz):
        a_vox = voxel_area[k]; a_true = true_area[k]
        if a_true>0 and a_vox>0:
            scale[k] = a_true / a_vox
        else:
            scale[k] = 1.0
    return scale

def slab_area_from_mesh_or_vox(mesh, mask_full, dx_m, origin, ks, ke, use_mesh=True):
    nx,ny,nz = mask_full.shape
    dx = dx_m
    mesh_area = 0.0; n_ok=0
    if use_mesh and mesh is not None:
        for k in range(ks, ke):
            z = origin[2] + (k+0.5)*dx
            try:
                section = mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1.0])
                if section is None: continue
                planar, _ = section.to_planar()
                a = float(planar.area)
                if a>0: mesh_area += a; n_ok += 1
            except Exception:
                pass
    if n_ok>0:
        return mesh_area / n_ok
    vox_areas = [np.count_nonzero(mask_full[:,:,k]) * (dx*dx) for k in range(ks, ke)]
    if len(vox_areas)==0:
        return 0.0
    return float(np.mean(vox_areas))
