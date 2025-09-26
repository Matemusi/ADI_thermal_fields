# quick_compare_dirichlet_robin.py
# -*- coding: utf-8 -*-
"""
Сравнение аналитики для ступеньки температуры на торце (Дирихле: T=Ts на z=0 для t>0)
при боковой конвекции (Робин) с численным ADI.
Сверху z=zmax задаём Дирихле T_inf (аппроксимация полупространства).

Аналитика (осьсимметрия, m=0):
  θ(r,z,t) = Σ_n A_n J0(λ_n r) F(z,t; λ_n),
  A_n = (2 Δ / μ_n) * J1(μ_n) / (J0(μ_n)^2+J1(μ_n)^2),  Δ=Ts - T_inf,
  μ_n: корни μ J0'(μ)+Bi J0(μ)=0,  λ_n = μ_n / R,
  F(z,t;λ) = 0.5 * [ e^{-λ z} erfc(z/(2√(αt)) - λ√(αt)) + e^{λ z} erfc(z/(2√(αt)) + λ√(αt)) ]

Выход:
  - z_profiles_dirichlet_overlay.png
  - time_trace_dirichlet_overlay.png
"""

import math
import argparse
import numpy as np

def main():
    p = argparse.ArgumentParser(description="Dirichlet @ z=0 (Ts), side Robin; analytic vs ADI")
    # Материал/геометрия
    p.add_argument("--k",      type=float, default=54.0)
    p.add_argument("--rho",    type=float, default=7800.0)
    p.add_argument("--cp",     type=float, default=490.0)
    p.add_argument("--R",      type=float, default=0.02)
    p.add_argument("--h_side", type=float, default=500.0)
    p.add_argument("--T_inf",  type=float, default=20.0)
    p.add_argument("--Ts",     type=float, default=1000.0)
    # Сетка/время
    p.add_argument("--nxr",    type=int,   default=64)
    p.add_argument("--nz",     type=int,   default=160)
    p.add_argument("--zmax",   type=float, default=0.12)
    p.add_argument("--tmin",   type=float, default=0.01)
    p.add_argument("--tmax",   type=float, default=5.0)
    p.add_argument("--nframes",type=int,   default=6)
    p.add_argument("--z_probe",type=float, default=0.04)
    p.add_argument("--modes",  type=int,   default=18)
    # Численный шаг
    p.add_argument("--theta",  type=float, default=0.5)
    p.add_argument("--cfl",    type=float, default=2.0)
    args = p.parse_args()

    # --- Imports ---
    import matplotlib.pyplot as plt
    from scipy.special import j0, j1, jvp, erfc
    import scipy.optimize as opt
    try:
        import adi3d_numba_coeff as adi
    except Exception as e:
        raise ImportError("Не найден adi3d_numba_coeff.py рядом со скриптом.") from e

    # --- Физика/сетка ---
    alpha = args.k/(args.rho*args.cp)
    dx = args.R / float(args.nxr)
    nx = ny = int(round((2.0*args.R)/dx))
    nz = args.nz if args.nz > 0 else int(round(args.zmax/dx))
    zmax = nz * dx
    print(f"[grid] nx=ny={nx}, nz={nz}, dx={dx:.6e} m; zmax={zmax:.3f} m")
    print(f"[phys] alpha={alpha:.3e} m^2/s, Bi_side={args.h_side*args.R/args.k:.3f}")

    times = np.linspace(args.tmin, args.tmax, args.nframes)

    # --- Аналитика: корни и коэффициенты ---
    def robin_mu_roots(Bi, n_roots=20, mu_max=200.0, grid_pts=15000):
        def f(mu): return mu*jvp(0, mu, 1) + Bi*j0(mu)
        mus = np.linspace(1e-8, mu_max, grid_pts)
        vals = f(mus)
        roots = []
        for i in range(len(mus)-1):
            a, b = mus[i], mus[i+1]
            fa, fb = vals[i], vals[i+1]
            if np.isnan(fa) or np.isnan(fb):
                continue
            if fa*fb < 0:
                try:
                    r = opt.brentq(f, a, b, maxiter=200)
                    if len(roots) == 0 or abs(r - roots[-1]) > 1e-8:
                        roots.append(r)
                        if len(roots) >= n_roots:
                            break
                except ValueError:
                    pass
        return np.array(roots, dtype=float)

    Bi_side = args.h_side*args.R/args.k
    mu = robin_mu_roots(Bi_side, n_roots=args.modes)
    lam = mu/args.R
    J0_mu = j0(mu); J1_mu = j1(mu)
    Delta = args.Ts - args.T_inf
    A_n = (2.0 * Delta / mu) * (J1_mu / (J0_mu**2 + J1_mu**2))

    def F_kernel(z, t, lam_):
        t = max(float(t), 1e-15)
        A = z/(2.0*np.sqrt(alpha*t))
        B = lam_ * np.sqrt(alpha*t)
        return 0.5 * ( np.exp(-lam_*z)*erfc(A - B) + np.exp(lam_*z)*erfc(A + B) )

    z_centers = (np.arange(nz)+0.5)*dx

    profiles_ana, trace_ana = [], []
    for tt in times:
        F = np.array([F_kernel(z_centers, tt, l) for l in lam])  # (modes, nz)
        Tz = args.T_inf + (A_n[:,None] * F).sum(axis=0)  # r=0 ⇒ J0(0)=1
        profiles_ana.append(Tz)
        k_probe = int(np.clip(int(round(args.z_probe/dx - 0.5)), 0, nz-1))
        z_probe = (k_probe+0.5)*dx
        Fp = np.array([F_kernel(z_probe, tt, l) for l in lam])
        Tp = args.T_inf + (A_n * Fp).sum()
        trace_ana.append(float(Tp))

    # --- Численно: цилиндрическая маска и ГУ (Dirichlet z-: Ts; Robin sides; Dirichlet top: T_inf) ---
    def build_cylinder_mask(nx, ny, nz, dx, R):
        cx = (nx/2.0); cy = (ny/2.0)
        xs = (np.arange(nx)+0.5-cx)*dx
        ys = (np.arange(ny)+0.5-cy)*dx
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        r_xy = np.sqrt(X**2 + Y**2)
        mask2d = (r_xy <= R + 1e-12)
        return np.repeat(mask2d[:, :, None], nz, axis=2), mask2d

    mask3d, mask2d = build_cylinder_mask(nx, ny, nz, dx, args.R)
    grid = adi.Grid3D(nx, ny, nz, dx, mask3d)
    mat  = adi.Material(args.rho, args.cp, args.k)

    dir_mask = np.zeros((nx,ny,nz), dtype=bool)
    dir_val  = np.full((nx,ny,nz), args.T_inf, dtype=float)
    # нижний слой = Ts, верхний = T_inf
    dir_mask[:,:,0]    = mask3d[:,:,0]
    dir_val[:,:,0]     = args.Ts
    dir_mask[:,:,nz-1] = mask3d[:,:,nz-1]
    dir_val[:,:,nz-1]  = args.T_inf

    packs = adi.precompute_coeff_packs_unified(
        grid, mat,
        dir_mask=dir_mask, dir_value=dir_val,
        neumann=None,
        robin_h={'x-': args.h_side, 'x+': args.h_side,
                 'y-': args.h_side, 'y+': args.h_side},
        robin_Tinf=args.T_inf
    )
    print("[bc] Dirichlet @ z- (Ts), Robin on sides (h_side), Dirichlet @ top (T_inf)")

    # Шаги по времени
    kappa = alpha
    dt_cap = args.cfl * dx*dx / kappa
    params = adi.Params(dt=1e-3, theta=args.theta)

    # Начальное поле
    Tcur = np.full((nx,ny,nz), args.T_inf, dtype=float)

    # прогрев numba
    try:
        params.dt = min(dt_cap, max(1e-6, times[0]/4))
        Tcur = adi.adi_step_numba_coeff(Tcur, grid, mat, params, packs, Tinf=args.T_inf)
    except TypeError:
        Tcur = adi.adi_step_numba_coeff(Tcur, grid, mat, params, packs)

    # Основной проход
    i0 = nx//2; j0 = ny//2
    k_probe = int(np.clip(int(round(args.z_probe/dx - 0.5)), 0, nz-1))
    z_probe = (k_probe+0.5)*dx

    profiles_num, trace_num = [], []
    t_cur = 0.0
    for tt in times:
        remain = float(tt - t_cur)
        nsub = max(1, int(math.ceil(remain / dt_cap))) if remain>0 else 0
        dt = remain / nsub if nsub>0 else 0.0
        params.dt = max(dt, 1e-15)
        for _ in range(nsub):
            try:
                Tcur = adi.adi_step_numba_coeff(Tcur, grid, mat, params, packs, Tinf=args.T_inf)
            except TypeError:
                Tcur = adi.adi_step_numba_coeff(Tcur, grid, mat, params, packs)
        t_cur = tt
        profiles_num.append(Tcur[i0, j0, :].copy())
        trace_num.append(Tcur[i0, j0, k_probe].item())
        print(f"[numeric] t={t_cur:.4g}s, sub={nsub}, dt={params.dt:.3g}s, T∈[{Tcur.min():.1f},{Tcur.max():.1f}]")

    # --- Графики ---
    # 1) Профили по z при r=0
    plt.figure(figsize=(7.8,4.6))
    for tt, Tz_a, Tz_n in zip(times, profiles_ana, profiles_num):
        plt.plot(z_centers, Tz_a, lw=1.8, label=f"ana t={tt:.2g}s")
        plt.plot(z_centers, Tz_n, lw=1.2, ls="--", label=f"num t={tt:.2g}s")
    plt.xlabel("z, m"); plt.ylabel("T, °C")
    plt.title("Dirichlet Ts @ z=0, side Robin h; T(z) at r=0 (analytic vs ADI)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("z_profiles_dirichlet_overlay.png", dpi=150)
    print("Saved: z_profiles_dirichlet_overlay.png")

    # 2) Трасса по времени
    plt.figure(figsize=(7.2,4.2))
    plt.plot(times, trace_ana, lw=1.8, marker="o", label="ana")
    plt.plot(times, trace_num, lw=1.2, marker="s", label="num", alpha=0.9)
    plt.xlabel("t, s"); plt.ylabel("T, °C")
    plt.title(f"Dirichlet Ts @ z=0, r=0, z≈{z_probe:.4f} m")
    plt.legend()
    plt.tight_layout()
    plt.savefig("time_trace_dirichlet_overlay.png", dpi=150)
    print("Saved: time_trace_dirichlet_overlay.png")

if __name__ == "__main__":
    main()
