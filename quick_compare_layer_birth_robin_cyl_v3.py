#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, math, time
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, jvp, erfc

# corrected core
from adi3d_cyl_phi_v3 import GridCyl, Material, Params, RobinR, ZBC, adi_step

# ---------- analytics (as in your v3) ----------
def robin_mu_roots(Bi: float, n_roots: int = 20, mu_max: float = 200.0, grid_pts: int = 30000) -> np.ndarray:
    xs = np.linspace(1e-12, mu_max, grid_pts)
    def f(mu): return mu * jvp(0, mu, 1) + Bi * j0(mu)
    vals = f(xs)
    roots = []
    for i in range(len(xs)-1):
        a,b = xs[i], xs[i+1]
        fa,fb = vals[i], vals[i+1]
        if np.isnan(fa) or np.isnan(fb): continue
        if fa == 0.0:
            roots.append(a)
        elif fa*fb < 0.0:
            try:
                r = brentq(f, a, b, maxiter=200)
                if (not roots) or abs(r-roots[-1])>1e-9:
                    roots.append(r)
                    if len(roots)>=n_roots: break
            except Exception: pass
    return np.array(roots, float)

def precompute_radial_modes(Bi: float, R: float, n_roots: int = 20):
    mu = robin_mu_roots(Bi, n_roots=n_roots); lam = mu/R
    J0, J1 = j0(mu), j1(mu)
    Cn = 2.0 * J1 / (mu*(J0**2 + J1**2))
    return mu, lam, Cn

def G_R(s: np.ndarray, xi: np.ndarray, u: float, beta: float, alpha: float) -> np.ndarray:
    denom = np.sqrt(4.0*np.pi*alpha*u)
    spx = s + xi
    smx = s - xi
    ga = np.exp(-(smx*smx)/(4.0*alpha*u)) / denom
    gb = np.exp(-(spx*spx)/(4.0*alpha*u)) / denom
    corr = beta*np.exp(beta*spx + alpha*(beta**2)*u) * erfc(spx/(2.0*np.sqrt(alpha*u)) + beta*np.sqrt(alpha*u))
    return ga + gb - corr

def analytic_axis_profile(z_grid: np.ndarray, t: float, d: float, t_step: float, N_total: int,
                          R: float, k: float, rho: float, c: float,
                          h_side: float, h_end: float, T_inf: float, Ts: float,
                          modes=None):
    alpha = k/(rho*c)
    if modes is None:
        mu, lam, Cn = precompute_radial_modes((h_side*R)/k, R, n_roots=24)
    else:
        mu, lam, Cn = modes
    Delta = Ts - T_inf
    N_now = max(0, min(N_total, int(math.floor(t/t_step))+1))
    theta = np.zeros_like(z_grid)
    if N_now == 0: return T_inf + theta, (z_grid*0).astype(bool)
    L = N_now*d
    s = L - z_grid
    exist = s >= 0.0
    if not np.any(exist): return T_inf + theta, exist
    Nxi = 64
    xi = np.linspace(0.0, d, Nxi)
    w_xi = np.gradient(xi)
    for j in range(N_now):
        u = t - j*t_step
        if u <= 0.0: continue
        offset = (N_now - 1 - j) * d
        xi_shift = xi[None, :] + offset
        s_exist = s[exist][:, None]
        Gmat = G_R(s_exist, xi_shift, u, beta=h_end/k, alpha=alpha)
        base_profile = (Gmat * w_xi).sum(axis=1)
        radial_factor = (Delta * (Cn * np.exp(-alpha*(lam**2)*u))).sum()
        theta[exist] += radial_factor * base_profile
    return T_inf + theta, exist

def build_time_grid(nframes:int, t_growth_end:float, t_tail:float, frame_dt:float|None):
    if frame_dt and frame_dt>0.0:
        tmax = t_growth_end + max(0.0, t_tail)
        n = int(np.floor(tmax/frame_dt + 1e-12)) + 1
        times = np.linspace(0.0, n*frame_dt, n, endpoint=False)
        return times[times <= tmax + 1e-12]
    n_g = max(2, nframes//2); n_r = max(2, nframes-n_g)
    tg = np.linspace(0.0, t_growth_end, n_g, endpoint=True)
    tr = np.linspace(t_growth_end, t_growth_end+t_tail, n_r, endpoint=True)
    return np.unique(np.clip(np.concatenate([tg,tr]), 0.0, None))

def main():
    p = argparse.ArgumentParser(description="Quick compare ADI (cyl, φ) vs analytic on axis")
    # физика/геометрия
    p.add_argument("--R", type=float, required=True)
    p.add_argument("--z_back", type=float, required=True)
    p.add_argument("--d", type=float, required=True)
    p.add_argument("--t_step", type=float, required=True)
    p.add_argument("--N_total", type=int, required=True)
    p.add_argument("--t_tail", type=float, required=True)
    # сетка
    p.add_argument("--nr", type=int, required=True)
    p.add_argument("--nphi", type=int, required=True)
    # ГУ
    p.add_argument("--h_side", type=float, required=True)
    p.add_argument("--h_end", type=float, required=True)
    p.add_argument("--T_inf", type=float, required=True)
    p.add_argument("--Ts", type=float, required=True)
    # материал
    p.add_argument("--rho", type=float, default=7800.0)
    p.add_argument("--cp", type=float, default=490.0)
    p.add_argument("--k", type=float, default=54.0)
    # временные параметры
    p.add_argument("--nframes", type=int, default=15)
    p.add_argument("--z_probe", type=float, default=0.0)
    p.add_argument("--cfl", type=float, default=1.0)
    p.add_argument("--dt_fixed", type=float, default=None)
    p.add_argument("--scheme", type=str, default="be", choices=["be","douglas"])
    p.add_argument("--frame_dt", type=float, default=None)
    p.add_argument("--ana_stride", type=int, default=1)
    args = p.parse_args()

    # шаг сетки
    dr = args.R/args.nr
    dz = dr
    dphi = (2.0*np.pi)/max(args.nphi,1)
    alpha = args.k/(args.rho*args.cp)
    dt_cap = args.cfl * min(dr*dr, dz*dz, (args.R*dphi)**2 if args.nphi>1 else 1e9) / max(alpha,1e-16)
    dt0 = args.dt_fixed if args.dt_fixed is not None else dt_cap

    # стартовая сетка (первый слой уже присутствует)
    L = args.d
    nz0 = int(round((args.z_back + L)/dz))
    grid = GridCyl(args.nr, args.nphi, nz0, dr, dphi, dz, args.R)
    mat  = Material(args.rho, args.cp, args.k)
    prm  = Params(dt0, 1.0 if args.scheme=='be' else 0.5, scheme=args.scheme)
    rob  = RobinR(args.h_side, args.T_inf)
    zbc  = ZBC('neumann0', 'robin', h_top=args.h_end, T_inf_top=args.T_inf)

    # поле T: фон T_inf + горячий хвост толщиной d
    T = np.full((grid.nr, grid.nphi, grid.nz), args.T_inf, dtype=float)
    nz_extra = int(round(args.d/dz))
    T[:,:, -nz_extra:] = args.Ts

    # глобальная z-сетка для профилей/аналитики
    L_final = args.N_total*args.d
    nz_final = int(round((args.z_back + L_final)/dz))
    zgrid_out = np.linspace(-args.z_back, L_final, nz_final, endpoint=False)

    # таймлайн кадров
    growth_end = max(0.0, (args.N_total-1)*args.t_step)
    times = build_time_grid(args.nframes, growth_end, args.t_tail, args.frame_dt)

    # кэш мод для аналитики
    modes = precompute_radial_modes((args.h_side*args.R)/args.k, args.R, n_roots=24)
    ana_stride = max(1, int(args.ana_stride))

    # индекс пробы
    iz_probe_global = int(round((args.z_probe + args.z_back)/dz))
    iz_probe_global = max(0, min(iz_probe_global, nz_final-1))

    def dump_frame(t, grid, T):
        y_num = np.full(nz_final, np.nan, float); y_num[:grid.nz] = T[0,0,:]
        if ((len(times_out)) % ana_stride)==0:
            y_ana,_ = analytic_axis_profile(zgrid_out, t, args.d, args.t_step, args.N_total,
                                            args.R, args.k, args.rho, args.cp,
                                            args.h_side, args.h_end, args.T_inf, args.Ts, modes)
        else:
            y_ana = np.full_like(y_num, np.nan)
        num_at_probe = float(y_num[iz_probe_global]) if 0<=iz_probe_global<len(y_num) else float('nan')
        ana_at_probe = float(y_ana[iz_probe_global]) if np.isfinite(y_ana[iz_probe_global]) else float('nan')
        print(f"[frame] t={t:0.3f}s | nz={grid.nz}, probe num/ana: {num_at_probe:0.2f}/{ana_at_probe:0.2f} °C")
        return y_num, y_ana
    # цикл: событийно согласованный маршинг
    times_out=[]; num_vals=[]; ana_vals=[]
    t = 0.0
    next_birth = args.t_step if args.N_total>1 else float('inf')
    # первый кадр
    y_num, y_ana = dump_frame(t, grid, T)
    times_out.append(t)
    num_vals.append(y_num[iz_probe_global])
    ana_vals.append(y_ana[iz_probe_global] if np.isfinite(y_ana[iz_probe_global]) else np.nan)

    # локальный базовый шаг (не мутируем prm.dt при подшагах)
    dt_base = dt0
    eps = 1e-12

    for t_target in times[1:]:
        while t < t_target - eps:
            remaining = t_target - t
            to_birth  = next_birth - t
            dt_step = min(dt_base, remaining, max(eps, to_birth))
            prm_step = Params(dt_step, prm.theta, prm.scheme)
            T = adi_step(T, grid, mat, prm_step, rob, zbc, S=None)
            t += dt_step
            if abs(t - next_birth) <= eps:
                # рождаем слой ровно в момент события
                if (grid.nz + nz_extra) <= nz_final:
                    old = T
                    nz_new = grid.nz + nz_extra
                    T = np.full((grid.nr, grid.nphi, nz_new), args.T_inf, float)
                    T[:, :, :old.shape[2]] = old
                    T[:, :, -nz_extra:] = args.Ts
                    grid = GridCyl(args.nr, args.nphi, nz_new, dr, dphi, dz, args.R)
                next_birth = next_birth + args.t_step if np.isfinite(next_birth) else float('inf')
        # кадр в точном времени
        t = t_target
        y_num, y_ana = dump_frame(t, grid, T)
        times_out.append(t)
        num_vals.append(y_num[iz_probe_global])
        ana_vals.append(y_ana[iz_probe_global] if np.isfinite(y_ana[iz_probe_global]) else np.nan)

    print(f"done in {len(times_out)} frames")
    return np.array(times_out), np.array(num_vals), np.array(ana_vals)


if __name__ == "__main__":
    main()
