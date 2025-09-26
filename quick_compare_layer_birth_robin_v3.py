
# -*- coding: utf-8 -*-
"""
quick_compare_layer_birth_robin_v3.py

Adds *lateral Robin area correction* for Cartesian voxel cylinder:
the digital (axis-aligned) perimeter overestimates the true circumference
by ~4/π ≈ 1.273, which caused ~20–25% extra cooling. We compute the exact
digital perimeter on the current grid and scale h_side by
gamma = (true_perimeter) / (digital_perimeter).

Everything else is the same as v2 (analytics, ADI, GIF).

Enable/disable with --fix_side_area (1/0). Default: 1 (enabled).
"""
import math, argparse, numpy as np, sys, os

def main():
    p = argparse.ArgumentParser(description="Layer-by-layer birth vs analytics (side Robin, end Robin) + r–z heatmaps, with lateral area correction.")
    # --- Material / geometry ---
    p.add_argument("--k",      type=float, default=54.0, help="W/m/K")
    p.add_argument("--rho",    type=float, default=7800.0)
    p.add_argument("--cp",     type=float, default=490.0)
    p.add_argument("--R",      type=float, default=0.02, help="Cylinder radius, m")
    p.add_argument("--h_side", type=float, default=500.0, help="Side Robin h, W/m^2/K (physical)")
    p.add_argument("--h_end",  type=float, default=500.0, help="End-face Robin h, W/m^2/K")
    p.add_argument("--T_inf",  type=float, default=20.0, help="Ambient, °C (sides & far end)")
    p.add_argument("--Ts",     type=float, default=1000.0, help="Layer birth temperature, °C")
    # --- Growth schedule ---
    p.add_argument("--d",        type=float, default=0.02, help="Layer thickness, m")
    p.add_argument("--t_step",   type=float, default=20.0, help="Time between births, s")
    p.add_argument("--N_total",  type=int,   default=6,    help="Total number of layers")
    p.add_argument("--t_tail",   type=float, default=200.0,help="Relaxation time after last birth, s")
    # --- Domain extents ---
    p.add_argument("--z_back",  type=float, default=0.20, help="Initial pre-existing rod length in -z (m)")
    p.add_argument("--z_front", type=float, default=0.22, help="Reserved space in +z for future layers (m)")
    # --- Numerics/grids ---
    p.add_argument("--nxr",     type=int,   default=64, help="#cells across radius (R/dx)")
    p.add_argument("--theta",   type=float, default=0.5, help="ADI theta")
    p.add_argument("--cfl",     type=float, default=3000.0, help="dt_cap ~ cfl*dx^2/alpha")
    p.add_argument("--backend", choices=["cpu","gpu"], default="cpu")
    # --- Analytics sampling / plotting ---
    p.add_argument("--modes",   type=int,   default=20, help="# radial modes in analytic sum")
    p.add_argument("--nframes", type=int,   default=14, help="# frames in the time slider / plots")
    p.add_argument("--z_probe", type=float, default=0.05, help="Probe position z (absolute, m)")
    # --- Options ---
    p.add_argument("--fix_side_area", type=int, default=1, help="1: scale h_side by true/digital perimeter; 0: raw h_side")
    # --- I/O ---
    p.add_argument("--save_vtk",type=int,   default=0, help="1 to save VTK snapshots at output times")
    args = p.parse_args()

    # --- Imports that may be slow ---
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from scipy.special import j0, j1, jvp, erfc
    import scipy.optimize as opt

    # numeric backends (CPU or GPU)
    if args.backend == "cpu":
        import adi3d_numba_coeff as adi
    else:
        import adi3d_gpu_coeff as adi
        import cupy as cp  # used only if gpu

    alpha = args.k / (args.rho * args.cp)
    dx = args.R / float(args.nxr)
    # allocate length: back + front reserve
    L_back = args.z_back
    L_front = max(args.z_front, args.N_total*args.d + 3*dx)
    total_len = L_back + L_front
    nz = int(round(total_len / dx))
    nx = ny = int(round((2.0*args.R)/dx))
    zmax = nz * dx
    print(f"[grid] nx=ny={nx}, nz={nz}, dx={dx:.6e} m; zmax={zmax:.3f} m; back={L_back:.3f} m, front={L_front:.3f} m")

    # --- Build cylinder mask ---
    def build_cylinder_mask(nx, ny, nz, dx, R):
        cx = (nx/2.0); cy = (ny/2.0)
        xs = (np.arange(nx)+0.5-cx)*dx
        ys = (np.arange(ny)+0.5-cy)*dx
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        r_xy = np.sqrt(X**2 + Y**2)
        mask2d = (r_xy <= R + 1e-12)
        return np.repeat(mask2d[:, :, None], nz, axis=2), mask2d, xs, ys

    # grid origin so that z=0 corresponds to interface between "old rod" and first born layer
    k0 = int(round(L_back/dx - 0.5))
    z_coords = (np.arange(nz)+0.5)*dx - (k0+0.5)*dx  # absolute z where 0 is initial interface

    mask3d, mask2d, xs, ys = build_cylinder_mask(nx, ny, nz, dx, args.R)
    # Initially: fill only the back portion [0..k0], the rest is void (future layers)
    initial_k_end = max(0, min(nz-1, k0))
    mask3d[:, :, initial_k_end+1:] = False  # everything beyond end is empty

    # --- Lateral perimeter correction factor ---
    # Count exposed faces on the 2D section (per z-layer)
    def count_exposed_faces(mask2d):
        nx, ny = mask2d.shape
        cnt = 0
        for i in range(nx):
            for j in range(ny):
                if not mask2d[i,j]: continue
                if i-1<0 or not mask2d[i-1,j]: cnt += 1
                if i+1>=nx or not mask2d[i+1,j]: cnt += 1
                if j-1<0 or not mask2d[i,j-1]: cnt += 1
                if j+1>=ny or not mask2d[i,j+1]: cnt += 1
        return cnt
    faces = count_exposed_faces(mask2d)
    perim_digital = faces * dx
    perim_true    = 2.0 * math.pi * args.R
    gamma = (perim_true / perim_digital) if args.fix_side_area else 1.0
    h_side_eff = args.h_side * gamma
    Bi_side_eff = h_side_eff * args.R / args.k
    beta_end = args.h_end / args.k
    print(f"[phys] alpha={alpha:.3e} m^2/s, Bi_side={args.h_side*args.R/args.k:.3f} (raw), "
          f"Bi_side_eff={Bi_side_eff:.3f} (gamma={gamma:.6f}), beta_end={beta_end:.3f} 1/m")
    if args.fix_side_area:
        print(f"[note] Lateral Robin correction: digital perim = {perim_digital:.6g} m, true = {perim_true:.6g} m, scale h_side by γ=π/4≈{math.pi/4:.6f} (here {gamma:.6f}).")

    # --- Helper: rebuild packs when mask changes ---
    def build_packs_for_mask(mask3d_local):
        # Dirichlet only at far top to approximate infinity (z large positive):
        dir_mask = np.zeros((nx,ny,nz), dtype=bool)
        dir_mask[:,:,nz-1] = mask3d_local[:,:,nz-1]
        dir_val  = np.full((nx,ny,nz), args.T_inf, dtype=float)
        packs = adi.precompute_coeff_packs_unified(
            grid, mat,
            dir_mask=dir_mask, dir_value=dir_val,
            neumann=None,
            robin_h={'x-': h_side_eff, 'x+': h_side_eff,
                     'y-': h_side_eff, 'y+': h_side_eff,
                     'z+': args.h_end},        # moving END is at +z face of current mask
            robin_Tinf=args.T_inf
        )
        return packs

    # --- Create backend-specific grid / material ---
    grid = adi.Grid3D(nx, ny, nz, dx, mask3d if args.backend=="cpu" else (cp.asarray(mask3d)))
    mat  = adi.Material(args.rho, args.cp, args.k)

    packs = build_packs_for_mask(mask3d)
    params = adi.Params(dt=1e-3, theta=args.theta)
    dt_cap = args.cfl * dx*dx / alpha

    # --- Initial temperature field ---
    if args.backend == "cpu":
        T = np.full((nx,ny,nz), args.T_inf, dtype=float)
    else:
        T = cp.full((nx,ny,nz), args.T_inf, dtype=cp.float64)

    # --- Growth schedule times (analytics-style: more resolution in growth, then relax) ---
    def build_times():
        times_growth = []
        for j in range(args.N_total):
            t0 = j*args.t_step
            t1 = t0 + min(args.t_step, 0.8*args.t_step)
            seg = t0 + np.geomspace(1e-4, max(1e-3, t1-t0), max(3, args.nframes//args.N_total))
            seg = seg[seg <= t0 + args.t_step - 1e-6]
            times_growth.extend(seg.tolist())
        growth_end = (args.N_total-1)*args.t_step
        times_relax = np.linspace(growth_end, growth_end + args.t_tail, max(6, args.nframes//2))
        times = np.unique(np.clip(np.array(times_growth + times_relax.tolist()), 0.0, None))
        if len(times) > args.nframes:
            idx = np.linspace(0, len(times)-1, args.nframes).round().astype(int)
            times = times[idx]
        return times
    times = build_times()
    print(f"[time] output frames: {len(times)} from t={times[0]:.4g}s to {times[-1]:.4g}s")

    # --- Analytics (centerline) ---
    Delta = args.Ts - args.T_inf
    def robin_mu_roots(Bi, n_roots=20, mu_max=200.0, grid_pts=30000):
        def f(mu): return mu*jvp(0, mu, 1) + Bi*j0(mu)
        mus = np.linspace(1e-8, mu_max, grid_pts)
        vals = f(mus)
        roots = []
        for i in range(len(mus)-1):
            a, b = mus[i], mus[i+1]
            fa, fb = vals[i], vals[i+1]
            if np.isnan(fa) or np.isnan(fb): continue
            if fa*fb < 0.0:
                try:
                    r = opt.brentq(f, a, b, maxiter=200)
                    if len(roots) == 0 or abs(r - roots[-1]) > 1e-9:
                        roots.append(r)
                        if len(roots) >= n_roots: break
                except ValueError:
                    pass
        return np.array(roots, dtype=float)
    # IMPORTANT: analytics uses the *physical* Bi for the radial eigenproblem (not the corrected one),
    # because analytics is continuum, not voxel. So use args.h_side, not h_side_eff, here.
    Bi_side_physical = args.h_side * args.R / args.k
    mu = robin_mu_roots(Bi_side_physical, n_roots=args.modes)
    lam = mu / args.R
    J0_mu = j0(mu); J1_mu = j1(mu)
    C_n = 2.0*J1_mu/(mu*(J0_mu**2+J1_mu**2))  # projection over disk (m=0)
    def G_R(s, xi, u, beta, alpha):
        denom = np.sqrt(4.0*np.pi*alpha*u)
        spx = s + xi
        smx = s - xi
        ga = np.exp(-(smx**2)/(4.0*alpha*u)) / denom
        gb = np.exp(-(spx**2)/(4.0*alpha*u)) / denom
        corr = beta * np.exp(beta*spx + alpha*(beta**2)*u) * erfc(spx/(2.0*np.sqrt(alpha*u)) + beta*np.sqrt(alpha*u))
        return ga + gb - corr

    def analytic_centerline_profile_at_time(t, z_grid):
        N_now = int(np.floor(t/args.t_step)) + 1
        N_now = max(0, min(N_now, args.N_total))
        L = N_now * args.d
        s = L - z_grid
        exist = s >= 0.0
        theta = np.zeros_like(z_grid)
        if N_now == 0:
            return args.T_inf + theta
        Nxi = 64
        xi = np.linspace(0.0, args.d, Nxi)
        w_xi = np.gradient(xi)
        for j in range(N_now):
            t_j = j*args.t_step
            u = t - t_j
            if u <= 0.0: continue
            offset = (N_now - 1 - j) * args.d
            xi_shift = xi[None, :] + offset
            s_exist = s[exist][:, None]  # (Nz_exist,1)
            Gmat = G_R(s_exist, xi_shift, u, args.h_end/args.k, alpha)
            base_profile = (Gmat * w_xi).sum(axis=1)  # (Nz_exist,)
            decay = np.exp(-alpha*(lam**2)*u)         # (modes,)
            radial_factor = (Delta * C_n * decay).sum()  # centerline: J0(0)=1
            theta[exist] += radial_factor * base_profile
        return args.T_inf + theta

    # --- Numeric run with element birth ---
    def adi_step(T, params, packs, Tinf):
        if args.backend == "cpu":
            try:
                return adi.adi_step_numba_coeff(T, grid, mat, params, packs, Tinf=Tinf)
            except TypeError:
                return adi.adi_step_numba_coeff(T, grid, mat, params, packs)
        else:
            return adi.adi_step_gpu_coeff(T, grid, mat, params, packs, Tinf=Tinf)

    n_per_layer = max(1, int(round(args.d/dx)))
    current_end_k = initial_k_end
    births = [j*args.t_step for j in range(args.N_total)]

    params.dt = min(dt_cap, max(1e-6, 1e-3))
    T = adi_step(T, params, packs, args.T_inf)

    profiles_ana=[]; profiles_num=[]
    trace_ana=[]; trace_num=[]
    t_cur = 0.0
    probe_z = args.z_probe
    k_probe = int(np.clip(int(round((probe_z - z_coords[0])/dx)), 0, nz-1))
    i0 = nx//2; j0 = ny//2

    print(f"[schedule] births at: {births}")
    print(f"[layers] n_per_layer={n_per_layer}, initial_end_k={initial_k_end} (z≈{z_coords[initial_k_end]:.4f} m)")

    # helper to apply a birth at current_end_k -> current_end_k + n_per_layer
    def apply_birth(T, current_end_k):
        k_start = current_end_k + 1
        k_end   = min(nz-1, current_end_k + n_per_layer)
        if k_end < k_start: 
            return T, current_end_k
        if args.backend == "cpu":
            born = np.zeros_like(mask3d, dtype=bool)
            born[:,:,k_start:k_end+1] = mask2d[:,:,None]
            T[born] = args.Ts
            mask3d[born] = True
            grid.mask = mask3d
        else:
            born = cp.zeros_like(grid.mask, dtype=cp.bool_)
            cross = cp.asarray(mask2d, dtype=cp.bool_)
            for kk in range(k_start, k_end+1):
                born[:,:,kk] = cross
            T[born] = args.Ts
            grid.mask[born] = True
        new_end_k = k_end
        return T, new_end_k

    # storage for heatmap frames (mid-plane, right half: r>=0)
    half_nx = nx - i0
    heatmaps = []
    heatmap_times = []
    current_end_positions = []
    r_axis = (np.arange(half_nx)+0.5)*dx

    next_birth_idx = 0
    for tt in times:
        while next_birth_idx < len(births) and births[next_birth_idx] <= tt + 1e-15:
            t_b = births[next_birth_idx]
            if t_b > t_cur + 1e-15:
                seg = t_b - t_cur
                nsub = max(1, int(math.ceil(seg / dt_cap)))
                params.dt = max(seg / nsub, 1e-15)
                for _ in range(nsub):
                    T = adi_step(T, params, packs, args.T_inf)
                t_cur = t_b
            T, current_end_k = apply_birth(T, current_end_k)
            packs = build_packs_for_mask(grid.mask.get() if args.backend=="gpu" else grid.mask)
            next_birth_idx += 1

        seg = tt - t_cur
        if seg > 1e-15:
            nsub = max(1, int(math.ceil(seg / dt_cap)))
            params.dt = max(seg / nsub, 1e-15)
            for _ in range(nsub):
                T = adi_step(T, params, packs, args.T_inf)
            t_cur = tt

        if args.backend == "cpu":
            prof_num = T[i0,j0,:].copy()
        else:
            prof_num = cp.asnumpy(T[i0,j0,:])
        profiles_num.append(prof_num)

        prof_ana = analytic_centerline_profile_at_time(tt, z_coords)
        profiles_ana.append(prof_ana)

        trace_num.append(float(prof_num[k_probe]))
        trace_ana.append(float(prof_ana[k_probe]))

        if args.backend == "cpu":
            slice_xz = T[i0:, j0, :].copy()
        else:
            slice_xz = cp.asnumpy(T[i0:, j0, :])
        mask_half = mask3d[i0:, j0, :]
        H = np.where(mask_half, slice_xz, np.nan)
        heatmaps.append(H); heatmap_times.append(tt)
        L = z_coords[current_end_k]
        current_end_positions.append(L)

        print(f"[frame] t={t_cur:.4g}s | end_k={current_end_k} (z≈{z_coords[current_end_k]:.4f} m) | T_probe num/ana = {trace_num[-1]:.2f}/{trace_ana[-1]:.2f} °C")

    # --- Plots ---
    import matplotlib.pyplot as plt
    # 1) Profiles overlay
    plt.figure(figsize=(8.0,5.0))
    for ti, (tt, Ta, Tn) in enumerate(zip(times, profiles_ana, profiles_num)):
        ls = "-" if ti%2==0 else "--"
        plt.plot(z_coords, Ta, ls=ls, lw=1.8, label=f"ana t={tt:.2g}s")
        plt.plot(z_coords, Tn, ls=":", lw=1.4, label=f"num t={tt:.2g}s")
    plt.axvline(x=0.0, linewidth=0.8, alpha=0.6)
    for j in range(args.N_total+1):
        plt.axvline(x=j*args.d, linewidth=0.3, alpha=0.25)
    plt.xlabel("z (m)  [z=0 at first born layer start]")
    plt.ylabel("T (°C) at r=0")
    plt.title("Layer-by-layer accretion: analytics vs ADI numeric (with lateral-area-corrected Robin)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("z_profiles_layers_overlay.png", dpi=150)
    print("Saved: z_profiles_layers_overlay.png")

    # 2) Time trace at probe
    plt.figure(figsize=(7.2,4.4))
    plt.plot(times, trace_ana, linewidth=2.0, label="analytic")
    plt.plot(times, trace_num, linewidth=1.6, linestyle="--", label="numeric")
    plt.xlabel("t (s)")
    plt.ylabel(f"T(z={args.z_probe:g} m, r=0) (°C)")
    plt.title("Probe time trace (analytics vs numeric)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("time_trace_layers_overlay.png", dpi=150)
    print("Saved: time_trace_layers_overlay.png")

    # 3) r–z heatmap GIF
    finite_vals = np.concatenate([np.nan_to_num(H, nan=np.nan) for H in heatmaps], axis=None)
    finite_vals = finite_vals[~np.isnan(finite_vals)]
    vmin = float(np.min(finite_vals)) if finite_vals.size else args.T_inf
    vmax = float(np.max(finite_vals)) if finite_vals.size else args.T_inf
    extent = [z_coords[0], z_coords[-1], 0.0, args.R]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    im = ax.imshow(heatmaps[0], origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Temperature (°C)")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("r (m)")
    vtline = ax.axvline(current_end_positions[0], linewidth=1.5)
    title = ax.set_title(f"Layer birth, t={heatmap_times[0]:.2f} s")

    def update(frame):
        im.set_data(heatmaps[frame])
        vtline.set_xdata([current_end_positions[frame], current_end_positions[frame]])
        title.set_text(f"Layer birth, t={heatmap_times[frame]:.2f} s")
        return [im, vtline, title]

    anim = animation.FuncAnimation(fig, update, frames=len(heatmaps), interval=200, blit=False)
    try:
        anim.save("rz_heatmap_layers.gif", writer=animation.PillowWriter(fps=4))
        print("Saved: rz_heatmap_layers.gif")
    except Exception as e:
        print("GIF save failed:", e)
        fig.savefig("rz_heatmap_layers_frame0.png", dpi=150)
        print("Saved fallback: rz_heatmap_layers_frame0.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
