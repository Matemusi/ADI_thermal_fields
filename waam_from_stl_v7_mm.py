
# -*- coding: utf-8 -*-
"""
waam_from_stl_v7_mm.py — WAAM по STL (мм наружу, СИ внутри) с РОБАСТНОЙ заливкой объёма
Новое в v7:
- Надёжная «солидизация» даже для негерметичных STL:
  * --solidify auto|fill|flood|close_flood|off (по умолчанию auto)
  * auto: пытается VoxelGrid.fill(); если похоже на «скорлупу» — делает
          морфологическое закрытие (итераций задаётся --solid_close_iters) + flood-fill
- Морфология без SciPy: бинарная дилатация/эрозия 6-связности на numpy
- Flood-fill «снаружи» без циклов Python (итерационная дилатация воздуха)
- Подробные логи степени заполнения и выбранной стратегии
- Сохранены: --cfl, --mpl_backend, --precision, ray-voxelizer, авто-dx
"""
import os, math, argparse, time, sys
import numpy as np

# ---------- Утилиты ----------
def fmt_bytes(n):
    units = ['B','KB','MB','GB','TB']
    i = 0
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0; i += 1
    return f"{n:.2f} {units[i]}"

def log(s):
    print(s, flush=True)

def mm_to_m(x_mm): return float(x_mm) / 1000.0

def setup_matplotlib_backend(choice='auto'):
    import matplotlib
    chosen = None
    try:
        if choice == 'auto':
            in_jupyter = ('ipykernel' in sys.modules)
            if in_jupyter:
                try:
                    import ipympl  # noqa: F401
                    matplotlib.use('module://ipympl.backend_nbagg', force=True)
                    chosen = 'ipympl'
                except Exception:
                    try:
                        matplotlib.use('nbAgg', force=True)
                        chosen = 'nbagg'
                    except Exception:
                        chosen = matplotlib.get_backend()
            else:
                chosen = matplotlib.get_backend()
        elif choice == 'ipympl':
            import ipympl  # noqa: F401
            matplotlib.use('module://ipympl.backend_nbagg', force=True)
            chosen = 'ipympl'
        elif choice == 'nbagg':
            matplotlib.use('nbAgg', force=True)
            chosen = 'nbagg'
        elif choice == 'tk':
            matplotlib.use('TkAgg', force=True)
            chosen = 'TkAgg'
        elif choice == 'qt':
            matplotlib.use('Qt5Agg', force=True)
            chosen = 'Qt5Agg'
        elif choice == 'inline':
            from matplotlib_inline.backend_inline import set_matplotlib_formats  # noqa: F401
            chosen = matplotlib.get_backend()
        else:
            chosen = matplotlib.get_backend()
    except Exception as e:
        chosen = f"fallback ({e})"
    return chosen

# ---------- Морфология 6-связности на numpy ----------
def dilate6(a: np.ndarray) -> np.ndarray:
    a = a.astype(bool, copy=False)
    b = a.copy()
    b[1:,:,:] |= a[:-1,:,:]
    b[:-1,:,:] |= a[1:,:,:]
    b[:,1:,:] |= a[:,:-1,:]
    b[:,:-1,:] |= a[:,1:,:]
    b[:,:,1:] |= a[:,:,:-1]
    b[:,:,:-1] |= a[:,:,1:]
    return b

def erode6(a: np.ndarray) -> np.ndarray:
    a = a.astype(bool, copy=False)
    b = np.zeros_like(a, dtype=bool)
    # внутри куба, чтобы не выходить за границы
    core = (
        a[1:-1,1:-1,1:-1] &
        a[:-2,1:-1,1:-1] & a[2:,1:-1,1:-1] &
        a[1:-1,:-2,1:-1] & a[1:-1,2:,1:-1] &
        a[1:-1,1:-1,:-2] & a[1:-1,1:-1,2:]
    )
    b[1:-1,1:-1,1:-1] = core
    return b

def closing6(a: np.ndarray, iters: int=1) -> np.ndarray:
    x = a.astype(bool, copy=False)
    for _ in range(max(0, iters)):
        x = dilate6(x)
    for _ in range(max(0, iters)):
        x = erode6(x)
    return x

def flood_fill_outside(solid: np.ndarray, max_iters: int=None) -> np.ndarray:
    """
    Возвращает outside (True там, где "внешний воздух"), используя итеративную дилатацию.
    solid — булева маска объекта (True=твёрдое).
    """
    s = solid.astype(bool, copy=False)
    # Паддинг на 1 по всем осям
    air = ~np.pad(s, 1, mode='constant', constant_values=True)  # всё вне padded-бокса считаем воздухом
    outside = np.zeros_like(air, dtype=bool)
    # старт: любые ячейки на границе, где air=True
    outside[0,:,:] |= air[0,:,:]
    outside[-1,:,:] |= air[-1,:,:]
    outside[:,0,:] |= air[:,0,:]
    outside[:,-1,:] |= air[:,-1,:]
    outside[:,:,0] |= air[:,:,0]
    outside[:,:,-1] |= air[:,:,-1]
    # итеративное расширение внешего воздуха
    if max_iters is None:
        max_iters = sum(s.shape) + 10
    for _ in range(max_iters):
        grown = dilate6(outside) & air
        new = outside | grown
        if new.sum() == outside.sum():
            break
        outside = new
    # возвращаем в размер без паддинга
    return outside[1:-1,1:-1,1:-1]

def solidify_mask(mask_surface: np.ndarray, mode: str='auto', close_iters: int=2, verbose: bool=True) -> np.ndarray:
    """
    Превращает "скорлупу" в массив "полный объём".
    mode:
      - 'fill'         : оставить как есть (предполагаем, что уже vg.fill() применили выше);
      - 'flood'        : заливка внутренностей через flood-fill от внешних границ;
      - 'close_flood'  : морфологическое закрытие скорлупы + flood-fill;
      - 'auto'         : эвристика: если похоже на скорлупу -> close_flood, иначе оставляем.
      - 'off'          : без изменений.
    """
    m = mask_surface.astype(bool, copy=False)

    def is_shell_like(a: np.ndarray) -> bool:
        if a.sum() == 0:
            return True
        er = erode6(a)
        ratio = er.sum() / float(a.sum())
        # если после эрозии почти всё исчезло и заполненность маленькая — это скорлупа
        fill_frac = a.mean()
        shell = (ratio < 0.25) or (fill_frac < 0.02)
        if verbose:
            log(f"[solidify] shell-test: erosion_ratio={ratio:.3f}, fill_frac={fill_frac:.3f} -> {'SHELL' if shell else 'SOLID'}")
        return shell

    if mode == 'off':
        return m
    if mode == 'fill':
        # предполагаем, что VoxelGrid.fill() уже применён выше; просто возвращаем
        return m
    if mode == 'flood':
        outside = flood_fill_outside(m)
        inside_air = (~m) & (~outside)
        solid = m | inside_air
        if verbose:
            log(f"[solidify] flood: +{int(inside_air.sum()):,} вокс. добавлено внутрь (итого {int(solid.sum()):,})")
        return solid
    if mode == 'close_flood':
        closed = closing6(m, iters=int(close_iters))
        outside = flood_fill_outside(closed)
        inside_air = (~closed) & (~outside)
        solid = closed | inside_air
        if verbose:
            log(f"[solidify] close_flood(iters={close_iters}): +{int(inside_air.sum()):,} вокс. добавлено (итого {int(solid.sum()):,})")
        return solid
    if mode == 'auto':
        if is_shell_like(m):
            return solidify_mask(m, mode='close_flood', close_iters=close_iters, verbose=verbose)
        else:
            if verbose: log("[solidify] auto: маска выглядит объёмной — оставляю как есть.")
            return m
    return m

# ---------- VTK writer ----------
def write_vtk_structured_points(path, T, dx_mm, origin_mm=(0.0,0.0,0.0), field_name="Temperature", mask=None):
    T = np.asarray(T)
    assert T.ndim == 3
    nx, ny, nz = T.shape
    ox, oy, oz = map(float, origin_mm)
    dx = float(dx_mm)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("WAAM Structured Points (mm)\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {ox:.9g} {oy:.9g} {oz:.9g}\n")
        f.write(f"SPACING {dx:.9g} {dx:.9g} {dx:.9g}\n")
        f.write(f"POINT_DATA {nx*ny*nz}\n")
        f.write(f"SCALARS {field_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for k in range(nz):
            for j in range(ny):
                row = " ".join(f"{float(T[i,j,k]):.6g}" for i in range(nx))
                f.write(row + "\n")
        if mask is not None:
            M = np.asarray(mask, dtype=np.float32)
            assert M.shape == T.shape
            f.write("SCALARS Mask float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nz):
                for j in range(ny):
                    row = " ".join(f"{float(M[i,j,k]):.6g}" for i in range(nx))
                    f.write(row + "\n")

# ---------- STL → voxel (мм) ----------
def load_voxel_from_stl_mm(stl_path, dx_mm, pad_mm=0.0, voxel_method='ray',
                           auto_dx=True, max_voxels=12_000_000, solidify='auto', solid_close_iters=2):
    import trimesh
    from trimesh.voxel import creation as vx

    t0 = time.perf_counter()
    log(f"[I/O] Загрузка STL: {stl_path}")
    mesh = trimesh.load(stl_path, force='mesh')
    if mesh.is_empty or mesh.vertices.shape[0] == 0:
        raise RuntimeError("Пустой или некорректный STL")
    t1 = time.perf_counter()
    bounds = mesh.bounds
    extents_mm = mesh.extents
    log(f"[geom] Габариты (мм): ex={extents_mm[0]:.3f}, ey={extents_mm[1]:.3f}, ez={extents_mm[2]:.3f}")
    log(f"[geom] Bounds min={bounds[0]}, max={bounds[1]} (ед. STL=мм)")

    pad = float(pad_mm)
    ex,ey,ez = (extents_mm[0]+2*pad, extents_mm[1]+2*pad, extents_mm[2]+2*pad)

    def estimate_dims(dxx_mm):
        nx = int(math.ceil(ex / dxx_mm))
        ny = int(math.ceil(ey / dxx_mm))
        nz = int(math.ceil(ez / dxx_mm))
        return nx,ny,nz, nx*ny*nz

    nx,ny,nz,N = estimate_dims(dx_mm)
    log(f"[vox] Оценка: dx={dx_mm:.3g} мм ⇒ {nx}×{ny}×{nz} = {N:,} вокселей")

    if auto_dx and N > max_voxels and N>0:
        scale = (N / float(max_voxels)) ** (1.0/3.0)
        dx2 = dx_mm * scale
        nx2,ny2,nz2,N2 = estimate_dims(dx2)
        log(f"[vox] Авто-укрупнение шага: dx {dx_mm:.3g}→{dx2:.3g} мм (≈{scale:.3g}×) ⇒"
            f" {nx2}×{ny2}×{nz2} = {N2:,}")
        dx_mm = dx2; nx,ny,nz,N = nx2,ny2,nz2,N2

    bytes_T = N * 8
    bytes_M = N * 1
    log(f"[mem] Оценка памяти T+mask: {fmt_bytes(bytes_T+bytes_M)} (T: {fmt_bytes(bytes_T)}, mask: {fmt_bytes(bytes_M)})")

    log(f"[vox] Вокселизация: method={voxel_method}, solidify={solidify}")
    pitch = float(dx_mm)
    if voxel_method == 'ray':
        vg = vx.voxelize_ray(mesh, pitch=pitch)
    elif voxel_method == 'subdivide':
        vg = mesh.voxelized(pitch=pitch, method='subdivide')
    else:
        vg = vx.voxelize_ray(mesh, pitch=pitch)

    mask = vg.matrix.copy()
    shape = mask.shape
    origin_mm = (0.0, 0.0, 0.0)
    try:
        if hasattr(vg, 'transform') and vg.transform is not None:
            tr = np.asarray(vg.transform, dtype=float)
            c0 = tr[:3, 3]
            origin_mm = (float(c0[0] - pitch/2.0), float(c0[1] - pitch/2.0), float(c0[2] - pitch/2.0))
            log(f"[vox] ORIGIN из transform (мм): {origin_mm}")
        else:
            log("[vox] transform отсутствует; ORIGIN=0,0,0 (мм)")
    except Exception as e:
        log(f"[vox] Не удалось получить transform: {e}; ORIGIN=0,0,0 (мм)")

    # БЫЛО: vg.fill(); ТЕПЕРЬ: более робастная стратегия
    mask_before = mask.copy()
    vox_before = int(mask_before.sum())

    if solidify in ('fill','off'):
        if solidify == 'fill':
            try:
                vg2 = vg.fill()
                mask = vg2.matrix.copy()
                log(f"[vox] VoxelGrid.fill(): +{int(mask.sum()-vox_before):,} вокс.")
            except Exception as e:
                log(f"[vox] fill() не удалось: {e} — остаюсь на surface.")
        # else: off — оставляем surface
    elif solidify in ('flood','close_flood','auto'):
        # сначала пробуем fill() как «дешёвый» шаг
        tried_fill = False
        try:
            vg2 = vg.fill()
            mask = vg2.matrix.copy()
            tried_fill = True
            log(f"[vox] VoxelGrid.fill() пробовали: +{int(mask.sum()-vox_before):,} вокс.")
        except Exception as e:
            log(f"[vox] fill() не удалось: {e} — переключаюсь на {solidify}.")
        # если auto — проверим, не скорлупа ли
        if solidify == 'auto':
            mode = 'auto'
        else:
            mode = solidify
        mask = solidify_mask(mask, mode=mode, close_iters=int(solid_close_iters), verbose=True)
    else:
        log(f"[vox] Неизвестный solidify={solidify}, оставляю surface.")

    N_real = int(mask.size)
    log(f"[vox] Готово: решётка {shape[0]}×{shape[1]}×{shape[2]} = {N_real:,} (dx={dx_mm:.6g} мм). "
        f"Заполнено: {int(mask.sum()):,} ({100.0*mask.mean():.2f}%)")
    t2 = time.perf_counter()
    log(f"[time] STL: {(t1-t0):.3f} c, вокселизация: {(t2-t1):.3f} c")
    return mask, origin_mm, float(dx_mm), shape, mesh

# ---------- ADI backend ----------
def import_backend(name):
    try:
        if name == "gpu":
            import adi3d_gpu_coeff as backend
            return backend, True
        else:
            import adi3d_numba_coeff as backend
            return backend, False
    except Exception as e:
        log(f"[warn] Не удалось импортировать {name} бэкенд: {e}\n[warn] Пробую CPU...")
        try:
            import adi3d_numba_coeff as backend
            return backend, False
        except Exception as e2:
            raise ImportError("Положите adi3d_numba_coeff(.py) рядом со скриптом") from e2

# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser(description="WAAM по STL (мм наружу, СИ внутри), надёжная заливка объёма.")
    p.add_argument("--stl", type=str, required=True, help="Путь к STL (в мм).")
    p.add_argument("--dx_mm", type=float, default=2.0, help="Шаг вокселя (мм).")
    p.add_argument("--pad_mm", type=float, default=0.0, help="Отступ (мм) вокруг модели при вокселизации.")
    p.add_argument("--voxel_method", choices=["ray","subdivide"], default="ray", help="Метод вокселизации.")
    p.add_argument("--solidify", choices=["auto","fill","flood","close_flood","off"], default="auto",
                   help="Стратегия превращения скорлупы в объём.")
    p.add_argument("--solid_close_iters", type=int, default=2, help="Итераций морфологического закрытия при close_flood/auto.")
    p.add_argument("--auto_dx", type=int, default=1, help="1 — авто-укрупнять dx при превышении лимита.")
    p.add_argument("--max_voxels", type=float, default=12_000_000, help="Лимит N = nx*ny*nz.")
    # WAAM-параметры (мм/мм/мм/с)
    p.add_argument("--bead_height_mm", type=float, default=1.0, help="Высота слоя (мм).")
    p.add_argument("--bead_width_mm",  type=float, default=3.0, help="Ширина валика (мм).")
    p.add_argument("--scan_speed_mm_s",type=float, default=15.0, help="Скорость сопла (мм/с).")
    p.add_argument("--eta_fill",    type=float, default=1.05,    help="Коэфф. на развороты/перехлёсты.")
    # Материал / ГУ (СИ)
    p.add_argument("--k",      type=float, default=54.0,  help="Теплопроводность, Вт/м/К.")
    p.add_argument("--rho",    type=float, default=7800.0,help="Плотность, кг/м^3.")
    p.add_argument("--cp",     type=float, default=490.0, help="Теплоёмкость, Дж/кг/К.")
    p.add_argument("--h_side", type=float, default=40.0,  help="Коэфф. теплоотдачи, Вт/м^2/К.")
    p.add_argument("--T_inf",  type=float, default=20.0,  help="Температура среды, °C.")
    p.add_argument("--Ts",     type=float, default=1000.0,help="Температура рождения слоя, °C.")
    # Число
    p.add_argument("--theta",  type=float, default=0.5,   help="θ-схема ADI.")
    p.add_argument("--cfl",    type=float, default=2000.0,help="dt_cap ~ cfl*dx^2/alpha.")
    p.add_argument("--backend",choices=["cpu","gpu"], default="cpu", help="Выбор бэкенда.")
    p.add_argument("--precision", choices=["float64","float32"], default="float64", help="Тип числа для T.")
    # Viewer
    p.add_argument("--viewer",  action="store_true", help="Интерактивный просмотрщик.")
    p.add_argument("--mpl_backend", choices=["auto","ipympl","nbagg","tk","qt","inline"], default="auto", help="Backend Matplotlib.")
    # Вывод
    p.add_argument("--nframes", type=int, default=20, help="Сколько кадров сохранить.")
    p.add_argument("--save_vtk",type=int, default=1,  help="1 — писать VTK.")
    p.add_argument("--outdir",  type=str, default="out_waam", help="Папка результатов.")
    return p

def main():
    args = build_argparser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) STL → vox (в мм, с solidify)
    mask_full, origin_mm, dx_mm, (nx,ny,nz), mesh = load_voxel_from_stl_mm(
        args.stl, args.dx_mm, pad_mm=args.pad_mm,
        voxel_method=args.voxel_method,
        auto_dx=bool(args.auto_dx),
        max_voxels=int(args.max_voxels),
        solidify=args.solidify,
        solid_close_iters=args.solid_close_iters
    )
    # 2) СИ внутри
    dx_m = mm_to_m(dx_mm)
    bead_height_m = mm_to_m(args.bead_height_mm)
    bead_width_m  = mm_to_m(args.bead_width_mm)
    scan_speed_m_s= mm_to_m(args.scan_speed_mm_s)
    alpha = args.k/(args.rho*args.cp)  # м^2/с
    log(f"[units] dx={dx_mm:.6g} мм ({dx_m:.6g} м), bead_height={args.bead_height_mm:.6g} мм, bead_width={args.bead_width_mm:.6g} мм, "
        f"scan_speed={args.scan_speed_mm_s:.6g} мм/с ({scan_speed_m_s:.6g} м/с)")

    # 3) ADI backend
    backend, on_gpu = import_backend(args.backend)
    Grid3D = backend.Grid3D; Material = backend.Material; Params = backend.Params
    mask_act = np.zeros_like(mask_full, dtype=bool)
    grid = Grid3D(nx,ny,nz,dx_m, mask_act)
    mat  = Material(args.rho, args.cp, args.k)
    params = Params(dt=1e-3, theta=args.theta)
    dt_cap = args.cfl * dx_m*dx_m / alpha
    log(f"[num] alpha={alpha:.3e} м^2/с, dt_cap≈{dt_cap:.3e} с (cfl={args.cfl})")

    # 4) Поле температур
    dtype = np.float64 if args.precision == "float64" else np.float32
    if on_gpu:
        import cupy as cp
        T = cp.full((nx,ny,nz), args.T_inf, dtype=dtype)
    else:
        T = np.full((nx,ny,nz), args.T_inf, dtype=dtype)

    # 5) Robin одинаковый везде
    def build_packs():
        return backend.precompute_coeff_packs_unified(
            grid, mat,
            dir_mask=None, dir_value=None,
            neumann=None,
            robin_h={'x-': args.h_side, 'x+': args.h_side,
                     'y-': args.h_side, 'y+': args.h_side,
                     'z-': args.h_side, 'z+': args.h_side},
            robin_Tinf=args.T_inf
        )

    def adi_step_local(T, params, packs):
        if on_gpu:
            return backend.adi_step_gpu_coeff(T, grid, mat, params, packs, Tinf=args.T_inf)
        else:
            try:
                return backend.adi_step_numba_coeff(T, grid, mat, params, packs, Tinf=args.T_inf)
            except TypeError:
                return backend.adi_step_numba_coeff(T, grid, mat, params, packs)

    # 6) Слои
    k_indices = np.where(mask_full.any(axis=(0,1)))[0]
    if k_indices.size == 0:
        raise RuntimeError("Пустая вокселизованная модель.")
    kmin, kmax = int(k_indices.min()), int(k_indices.max())
    n_per_layer = max(1, int(round(args.bead_height_mm / dx_mm)))
    layers = []
    ks = kmin
    while ks <= kmax:
        while ks <= kmax and not mask_full[:,:,ks].any():
            ks += 1
        if ks > kmax: break
        ke = min(kmax, ks + n_per_layer - 1)
        while ke >= ks and not mask_full[:,:,ke].any():
            ke -= 1
        if ke < ks:
            ks += 1
            continue
        layers.append((ks,ke))
        ks = ke + 1
    log(f"[layers] {len(layers)} слоёв; n_per_layer≈{n_per_layer}, k∈[{kmin},{kmax}]")

    # 7) Время слоёв
    def slab_area_from_vox(mask, dx_mm, ks, ke):
        areas_m2 = []
        a_pix_m2 = (mm_to_m(dx_mm) * mm_to_m(dx_mm))
        for k in range(ks, ke+1):
            if 0 <= k < mask.shape[2]:
                areas_m2.append(float(mask[:,:,k].sum()) * a_pix_m2)
        return float(np.mean(areas_m2)) if areas_m2 else 0.0

    times_birth = []
    t_cursor = 0.0
    for (ks,ke) in layers:
        A_layer_m2 = slab_area_from_vox(mask_full, dx_mm, ks, ke)
        L_est_m = (A_layer_m2 / max(bead_width_m, 1e-12)) * max(args.eta_fill, 1.0)
        t_layer = L_est_m / max(scan_speed_m_s, 1e-12)
        t_cursor += float(t_layer)
        times_birth.append(t_cursor)
    total_time = times_birth[-1] if times_birth else 0.0
    log(f"[time] Σ время печати ≈ {total_time:.3f} с")

    # 8) Времена вывода
    times_out = (np.linspace(0.0, total_time, args.nframes).tolist()
                 if (args.nframes > 1 and total_time > 0) else [0.0])

    packs = build_packs()
    saved_fields = []
    next_birth = 0
    t_now = 0.0

    def activate_layer(ks, ke):
        nonlocal T, grid, mask_act
        born = np.zeros_like(mask_full, dtype=bool)
        born[:,:,ks:ke+1] = mask_full[:,:,ks:ke+1]
        idx = np.where(born & (~mask_act))
        if len(idx[0]) > 0:
            T[idx] = dtype.type(args.Ts) if hasattr(dtype, 'type') else args.Ts
        mask_act |= born
        grid.mask = mask_act

    def save_frame(tstamp):
        if on_gpu:
            import cupy as cp
            T_cpu = cp.asnumpy(T)
        else:
            T_cpu = T.copy()
        Tmin = float(np.nanmin(T_cpu))
        Tmax = float(np.nanmax(T_cpu))
        if not np.isfinite(Tmin) or not np.isfinite(Tmax) or Tmax > 1e5 or Tmin < -1e5:
            log(f"[warn] Подозрительные значения поля: Tmin={Tmin:.3g}, Tmax={Tmax:.3g}")
        saved_fields.append((float(tstamp), T_cpu, mask_act.copy()))
        if args.save_vtk:
            fn = os.path.join(args.outdir, f"waam_{tstamp:010.3f}.vtk")
            try:
                write_vtk_structured_points(fn, T_cpu, dx_mm, origin_mm=origin_mm, field_name="Temperature", mask=mask_act.astype(np.float32))
            except Exception as e:
                log(f"[VTK] ошибка: {e}")

    events = sorted(set(times_out + times_birth))
    log(f"[time] событий: {len(events)} (рождения+кадры)")

    # ВАЖНО: не делаем шагов ADI, пока нет активных вокселей
    for te in events:
        # рождения
        while next_birth < len(times_birth) and times_birth[next_birth] <= te + 1e-15:
            t_b = times_birth[next_birth]
            seg = max(0.0, t_b - t_now)
            if seg > 1e-15 and mask_act.any():
                nsub = max(1, int(math.ceil(seg / dt_cap)))
                params.dt = max(seg / nsub, 1e-15)
                for _ in range(nsub):
                    T = adi_step_local(T, params, packs)
                t_now = t_b
            else:
                t_now = t_b
            ks, ke = layers[next_birth]
            activate_layer(ks, ke)
            packs = build_packs()
            next_birth += 1

        # досчёт до te
        seg = max(0.0, te - t_now)
        if seg > 1e-15 and mask_act.any():
            nsub = max(1, int(math.ceil(seg / dt_cap)))
            params.dt = max(seg / nsub, 1e-15)
            for _ in range(nsub):
                T = adi_step_local(T, params, packs)
            t_now = te
        else:
            t_now = te

        if any(abs(te - to) <= 1e-12 for to in times_out):
            log(f"[frame] t={t_now:.3f} s, активных: {int(mask_act.sum())}")
            save_frame(t_now)

    log(f"[done] Кадров: {len(saved_fields)}; outdir={args.outdir}")

    # 9) Просмотрщик
    if args.viewer and len(saved_fields) > 0:
        backend_name = setup_matplotlib_backend(args.mpl_backend)
        log(f"[viewer] Matplotlib backend: {backend_name}")
        if backend_name not in ('ipympl','nbagg','TkAgg','Qt5Agg'):
            log("[viewer] Внимание: текущий backend может быть НЕинтерактивным в Jupyter (inline). "
                "Совет: pip install ipympl && используйте --mpl_backend ipympl")

        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons
        times = [t for (t,_,_) in saved_fields]
        data0 = saved_fields[0][1]
        mask0 = saved_fields[0][2]
        k_valid = np.where(mask0.any(axis=(0,1)))[0]
        iz = int(round((k_valid.min()+k_valid.max())/2)) if len(k_valid)>0 else data0.shape[2]//2

        fig, ax = plt.subplots(figsize=(7.5,6))
        plt.subplots_adjust(left=0.25, bottom=0.25)
        frame0 = np.where(mask0[:,:,iz], data0[:,:,iz], np.nan).T
        if np.all(np.isnan(frame0)):
            frame0 = np.zeros_like(frame0); empty_note = True
        else:
            empty_note = False
        im = ax.imshow(frame0, origin='lower', aspect='equal')
        cbar = plt.colorbar(im, ax=ax); cbar.set_label("T (°C)")
        title = f"t={times[0]:.3f} s | XY @ Z={iz}" + (" (пустой срез)" if empty_note else "")
        ax.set_title(title)
        ax.set_xlabel("i (x)"); ax.set_ylabel("j (y)")

        ax_time = plt.axes([0.25, 0.12, 0.65, 0.03])
        s_time = Slider(ax_time, 't (s)', times[0], times[-1], valinit=times[0])
        ax_radio = plt.axes([0.02, 0.5, 0.18, 0.25])
        radio = RadioButtons(ax_radio, ('XY@Z', 'XZ@Y', 'YZ@X'))
        ax_idx = plt.axes([0.25, 0.06, 0.65, 0.03])
        s_idx = Slider(ax_idx, 'index', 0, max(nx,ny,nz)-1, valinit=iz, valstep=1)

        def redraw(val=None):
            tsel = s_time.val
            kf = int(np.argmin([abs(t - tsel) for t,_,_ in saved_fields]))
            t, Tnp, Mnp = saved_fields[kf]
            mode = radio.value_selected
            idx = int(s_idx.val)
            if mode.startswith("XY"):
                idx = np.clip(idx, 0, nz-1)
                frame = np.where(Mnp[:,:,idx], Tnp[:,:,idx], np.nan).T
                title = f"t={t:.3f} s | XY @ Z={idx}"
            elif mode.startswith("XZ"):
                idx = np.clip(idx, 0, ny-1)
                frame = np.where(Mnp[:,idx,:], Tnp[:,idx,:], np.nan).T
                title = f"t={t:.3f} s | XZ @ Y={idx}"
            else:
                idx = np.clip(idx, 0, nx-1)
                frame = np.where(Mnp[idx,:,:], Tnp[idx,:,:], np.nan).T
                title = f"t={t:.3f} s | YZ @ X={idx}"
            if np.all(np.isnan(frame)):
                frame = np.zeros_like(frame); title += " (пустой срез)"
            im.set_data(frame)
            vmin = float(np.nanmin(frame)) if not np.all(np.isnan(frame)) else 0.0
            vmax = float(np.nanmax(frame)) if not np.all(np.isnan(frame)) else 1.0
            if vmax == vmin: vmax = vmin + 1.0
            im.set_clim(vmin=vmin, vmax=vmax)
            ax.set_title(title)
            fig.canvas.draw_idle()

        s_time.on_changed(redraw)
        s_idx.on_changed(redraw)
        radio.on_clicked(redraw)
        redraw()
        plt.show()

if __name__ == "__main__":
    main()
