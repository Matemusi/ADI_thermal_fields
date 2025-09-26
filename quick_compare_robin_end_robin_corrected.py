# quick_compare_robin_end_robin_corrected.py
# -*- coding: utf-8 -*-
"""Comparison test using Robin coefficient correction from STL geometry."""

import math
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser(
        description=(
            "Robin heating at z=0 with lateral Robin; coefficients corrected "
            "from STL geometry"
        )
    )
    # material & geometry
    p.add_argument("--k", type=float, default=54.0)
    p.add_argument("--rho", type=float, default=7800.0)
    p.add_argument("--cp", type=float, default=490.0)
    p.add_argument("--R", type=float, default=0.02)
    # BCs
    p.add_argument("--h_side", type=float, default=500.0, help="Side Robin coefficient, W/m^2/K")
    p.add_argument("--h_end", type=float, default=800.0, help="End-face Robin coefficient, W/m^2/K")
    p.add_argument("--T_inf", type=float, default=20.0, help="Ambient for sides/top (°C)")
    p.add_argument("--Delta_end", type=float, default=200.0, help="Ambient STEP at end face, K")
    # numerics/grids
    p.add_argument("--nxr", type=int, default=64)
    p.add_argument("--nz", type=int, default=160)
    p.add_argument("--zmax", type=float, default=0.12)
    p.add_argument("--tmin", type=float, default=0.01)
    p.add_argument("--tmax", type=float, default=5.0)
    p.add_argument("--nframes", type=int, default=6)
    p.add_argument("--z_probe", type=float, default=0.04)
    p.add_argument("--modes", type=int, default=25)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--cfl", type=float, default=2.0)
    # correction params
    p.add_argument("--mesh_sections", type=int, default=96, help="Sections for synthetic STL cylinder")
    p.add_argument("--max_subdiv", type=int, default=6, help="Max triangle subdivision per voxel span")
    args = p.parse_args()

    import matplotlib.pyplot as plt
    from scipy.special import j0, j1, jvp, erfc
    import scipy.optimize as opt
    import trimesh

    import adi3d_numba_coeff as adi
    from voxel_bc_correction import build_corrected_robin_fields

    # --- material / geometry ---
    alpha = args.k / (args.rho * args.cp)
    dx = args.R / float(args.nxr)
    nx = ny = int(round((2.0 * args.R) / dx))
    nz = args.nz if args.nz > 0 else int(round(args.zmax / dx))
    zmax = nz * dx
    print(f"[grid] nx=ny={nx}, nz={nz}, dx={dx:.6e} m; zmax={zmax:.3f} m")
    Bi_side = args.h_side * args.R / args.k
    print(f"[phys] alpha={alpha:.3e} m^2/s, Bi_side={Bi_side:.3f}, beta_end={args.h_end/args.k:.3f}")

    times = np.linspace(args.tmin, args.tmax, args.nframes)

    # --- radial eigenvalues for side Robin ---
    def robin_mu_roots(Bi, n_roots=20, mu_max=220.0, grid_pts=40000):
        def f(mu):
            return mu * jvp(0, mu, 1) + Bi * j0(mu)

        mus = np.linspace(1e-8, mu_max, grid_pts)
        vals = f(mus)
        roots = []
        for i in range(len(mus) - 1):
            a, b = mus[i], mus[i + 1]
            fa, fb = vals[i], vals[i + 1]
            if np.isnan(fa) or np.isnan(fb):
                continue
            if fa * fb < 0:
                try:
                    r = opt.brentq(f, a, b, maxiter=200)
                    if len(roots) == 0 or abs(r - roots[-1]) > 1e-10:
                        roots.append(r)
                        if len(roots) >= n_roots:
                            break
                except ValueError:
                    pass
        return np.array(roots, dtype=float)

    mu = robin_mu_roots(Bi_side, n_roots=args.modes)
    lam = mu / args.R
    J0_mu = j0(mu)
    J1_mu = j1(mu)
    # projection of "1" over disk
    C_n = (2.0 * J1_mu) / (mu * (J0_mu ** 2 + J1_mu ** 2))

    # --- analytic kernel: Robin step (heating) at end face ---
    beta = args.h_end / args.k
    Delta = args.Delta_end

    def H_R_step(z, t, lam_val, beta_val, alpha_val, eps=1e-12):
        t = max(float(t), 1e-16)
        A = z / (2.0 * np.sqrt(alpha_val * t))
        B = lam_val * np.sqrt(alpha_val * t)

        den_p = (beta_val + lam_val)
        den_m = (beta_val - lam_val)
        den_b = (beta_val ** 2 - lam_val ** 2)
        if abs(den_p) < eps:
            den_p = np.sign(den_p) if den_p != 0 else 1.0
            den_p *= eps
        if abs(den_m) < eps:
            den_m = np.sign(den_m) if den_m != 0 else 1.0
            den_m *= eps
        if abs(den_b) < eps:
            den_b = np.sign(den_b) if den_b != 0 else 1.0
            den_b *= eps

        term1 = np.exp(-lam_val * z) * erfc(A - B) / (2.0 * den_p)
        term2 = np.exp(+lam_val * z) * erfc(A + B) / (2.0 * den_m)
        term3 = (
            (beta_val / den_b)
            * np.exp(beta_val * z + alpha_val * (beta_val ** 2) * t)
            * erfc(A + beta_val * np.sqrt(alpha_val * t))
            * np.exp(-alpha_val * (lam_val ** 2) * t)
        )

        return beta_val * (term1 + term2 - term3)

    # z grid for profiles (cell centers along axis r=0):
    z_centers = (np.arange(nz) + 0.5) * dx

    # --- analytic profiles and probe trace (r=0) ---
    profiles_ana = []
    trace_ana = []
    k_probe = int(np.clip(int(round(args.z_probe / dx - 0.5)), 0, nz - 1))
    z_probe = (k_probe + 0.5) * dx
    for tt in times:
        H_mat = np.array([H_R_step(z_centers, tt, l, beta, alpha) for l in lam])
        Tz = args.T_inf + Delta * (C_n[:, None] * H_mat).sum(axis=0)
        profiles_ana.append(Tz)
        Hp = np.array([H_R_step(z_probe, tt, l, beta, alpha) for l in lam])
        Tp = args.T_inf + Delta * (C_n * Hp).sum()
        trace_ana.append(float(Tp))

    # --- build cylinder mask ---
    def build_cyl(nx_val, ny_val, nz_val, dx_val, R_val):
        cx = nx_val / 2.0
        cy = ny_val / 2.0
        xs = (np.arange(nx_val) + 0.5 - cx) * dx_val
        ys = (np.arange(ny_val) + 0.5 - cy) * dx_val
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        mask2d = np.sqrt(X ** 2 + Y ** 2) <= R_val + 1e-12
        return np.repeat(mask2d[:, :, None], nz_val, axis=2), mask2d

    mask3d, _ = build_cyl(nx, ny, nz, dx, args.R)

    grid = adi.Grid3D(nx, ny, nz, dx, mask3d)
    mat = adi.Material(args.rho, args.cp, args.k)

    # --- BCs for ADI ---
    dir_mask = np.zeros((nx, ny, nz), dtype=bool)
    dir_mask[:, :, nz - 1] = mask3d[:, :, nz - 1]
    dir_val = np.full((nx, ny, nz), args.T_inf, dtype=float)

    # --- Corrected Robin coefficients from synthetic STL ---
    cyl_mesh = trimesh.creation.cylinder(radius=args.R, height=zmax, sections=max(3, args.mesh_sections))
    cyl_mesh.apply_translation([0.0, 0.0, zmax / 2.0])
    origin = np.array([-args.R, -args.R, 0.0], dtype=float)
    base_h = {
        "x-": args.h_side,
        "x+": args.h_side,
        "y-": args.h_side,
        "y+": args.h_side,
        "z-": args.h_end,
    }
    robin_fields, scale_fields = build_corrected_robin_fields(
        cyl_mesh,
        mask3d,
        origin=origin,
        dx=dx,
        base_h=base_h,
        fallback_to_base=True,
        max_subdiv=max(1, args.max_subdiv),
    )

    for face in ("x-", "x+", "y-", "y+", "z-"):
        exp = adi.exposed_mask(mask3d, face)
        scales = scale_fields.get(face)
        if scales is None:
            continue
        vals = scales[exp]
        if vals.size == 0:
            continue
        print(
            f"[corr] face {face}: scale min={vals.min():.3f}, max={vals.max():.3f}, "
            f"mean={vals.mean():.3f}"
        )

    # End-face z- Robin: same strategy as baseline test
    q_add = args.h_end * Delta
    packs = adi.precompute_coeff_packs_unified(
        grid,
        mat,
        dir_mask=dir_mask,
        dir_value=dir_val,
        neumann={"z-": q_add},
        robin_h=robin_fields,
        robin_Tinf=args.T_inf,
    )
    print("[bc] Using corrected Robin coefficients from STL projections.")

    qz = packs[2].qflux
    bot = qz[:, :, 0][mask3d[:, :, 0]]
    print(
        f"[sanity] bottom q_add A/Ccell: mean={np.mean(bot):.3e}, "
        f"min={np.min(bot):.3e}, max={np.max(bot):.3e}"
    )

    # --- time stepping ---
    kappa = alpha
    dt_cap = args.cfl * dx * dx / kappa
    params = adi.Params(dt=1e-3, theta=args.theta)

    Tcur = np.full((nx, ny, nz), args.T_inf, dtype=float)
    params.dt = min(dt_cap, max(1e-6, times[0] / 4.0))
    Tcur = adi.adi_step_numba_coeff(Tcur, grid, mat, params, packs, Tinf=args.T_inf)

    i0 = nx // 2
    j0 = ny // 2
    profiles_num = []
    trace_num = []
    t_cur = 0.0
    for ti, tt in enumerate(times):
        remain = float(tt - t_cur)
        nsub = max(1, int(math.ceil(remain / dt_cap))) if remain > 0 else 0
        dt = remain / nsub if nsub > 0 else 0.0
        params.dt = max(dt, 1e-15)
        for _ in range(nsub):
            Tcur = adi.adi_step_numba_coeff(Tcur, grid, mat, params, packs, Tinf=args.T_inf)
        t_cur = tt
        profiles_num.append(Tcur[i0, j0, :].copy())
        trace_num.append(Tcur[i0, j0, k_probe].item())
        print(
            f"[numeric] t={t_cur:.4g}s, sub={nsub}, dt={params.dt:.3g}s, "
            f"T∈[{Tcur.min():.2f},{Tcur.max():.2f}]"
        )
        if ti == 0:
            print(f"[sanity] z- center T(k=0) = {Tcur[i0, j0, 0]:.2f} °C")

    # error metrics
    errs = []
    for Ta, Tn in zip(profiles_ana, profiles_num):
        errs.append(np.max(np.abs(Ta - Tn)))
    print(f"[error] max |ana-num| per frame: {[f'{e:.3f}' for e in errs]}")

    plt.figure(figsize=(7.8, 4.6))
    for tt, Ta, Tn in zip(times, profiles_ana, profiles_num):
        plt.plot(z_centers, Ta, lw=1.8, label=f"ana t={tt:.2g}s")
        plt.plot(z_centers, Tn, lw=1.2, ls="--", label=f"num t={tt:.2g}s")
    plt.xlabel("z, m")
    plt.ylabel("T, °C")
    plt.title("Robin heating @ z=0 with STL-corrected coefficients")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("z_profiles_robin_end_overlay_corrected.png", dpi=150)
    print("Saved: z_profiles_robin_end_overlay_corrected.png")

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(times, trace_ana, lw=1.8, marker="o", label="ana")
    plt.plot(times, trace_num, lw=1.6, marker="s", label="num")
    plt.xlabel("t, s")
    plt.ylabel("T(r=0, z_probe), °C")
    plt.title("Probe trace @ r=0 with STL correction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("trace_robin_end_corrected.png", dpi=150)
    print("Saved: trace_robin_end_corrected.png")


if __name__ == "__main__":
    main()

