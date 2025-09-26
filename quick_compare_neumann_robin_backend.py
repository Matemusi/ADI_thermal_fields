# -*- coding: utf-8 -*-
"""
Neumann (q0>0 HEATING) @ z=0; бок — Robin(h_side, T_inf); верх — Dirichlet(T_inf).
Аналитика + численно (CPU или GPU) с теми же ГУ, как в твоём исходнике. (:contentReference[oaicite:5]{index=5})

Пример запуска (ровно твой кейс):
%run quick_compare_neumann_robin_backend.py \
  --R 0.02 --h_side 500 --k 54 --rho 7800 --cp 490 --T_inf 20 \
  --q0 2.0e6 \
  --nxr 64 --nz 1600 --zmax 1.2 --modes 18 \
  --tmin 0.01 --tmax 300 --nframes 6 --z_probe 0.005 \
  --theta 0.5 --cfl 3000 --backend both
"""
import math, argparse, numpy as np
import time

def main():
    p = argparse.ArgumentParser()
    # физика/геометрия
    p.add_argument("--k",      type=float, default=54.0)
    p.add_argument("--rho",    type=float, default=7800.0)
    p.add_argument("--cp",     type=float, default=490.0)
    p.add_argument("--R",      type=float, default=0.02)
    # ГУ
    p.add_argument("--h_side", type=float, default=500.0)
    p.add_argument("--T_inf",  type=float, default=20.0)
    p.add_argument("--q0",     type=float, default=2.0e6)
    # сетка/время
    p.add_argument("--nxr",    type=int,   default=64)
    p.add_argument("--nz",     type=int,   default=1600)
    p.add_argument("--zmax",   type=float, default=1.2)
    p.add_argument("--tmin",   type=float, default=0.01)
    p.add_argument("--tmax",   type=float, default=300.0)
    p.add_argument("--nframes",type=int,   default=6)
    p.add_argument("--z_probe",type=float, default=0.005)
    p.add_argument("--modes",  type=int,   default=18)
    # численно
    p.add_argument("--theta",  type=float, default=0.5)
    p.add_argument("--cfl",    type=float, default=3000.0)
    p.add_argument("--backend",choices=["cpu","gpu","both"], default="both")
    args = p.parse_args()

    import matplotlib.pyplot as plt
    from scipy.special import j0, j1, jvp, erfc, erfcx
    import scipy.optimize as opt

    alpha = args.k/(args.rho*args.cp)
    dx = args.R/float(args.nxr)
    nx = ny = int(round((2.0*args.R)/dx))
    nz = args.nz if args.nz>0 else int(round(args.zmax/dx))
    zmax = nz*dx
    print(f"[grid] nx=ny={nx}, nz={nz}, dx={dx:.6e} m; zmax={zmax:.3f} m")
    print(f"[phys] alpha={alpha:.3e} m^2/s, Bi_side={args.h_side*args.R/args.k:.3f}")

    # времена
    times = np.linspace(args.tmin, args.tmax, args.nframes)

    # --- аналитика (как в твоем скрипте) ---
    def robin_mu_roots(Bi, n_roots=20, mu_max=200.0, grid_pts=15000):
        def f(mu): return mu*jvp(0, mu, 1) + Bi*j0(mu)
        mus = np.linspace(1e-8, mu_max, grid_pts); vals=f(mus)
        roots=[]
        for i in range(len(mus)-1):
            a,b=mus[i],mus[i+1]; fa,fb=vals[i],vals[i+1]
            if np.isnan(fa) or np.isnan(fb): continue
            if fa*fb<0:
                try:
                    r=opt.brentq(f,a,b,maxiter=200)
                    if len(roots)==0 or abs(r-roots[-1])>1e-8:
                        roots.append(r)
                        if len(roots)>=n_roots: break
                except ValueError: pass
        return np.array(roots,dtype=float)
    Bi_side = args.h_side*args.R/args.k
    mu = robin_mu_roots(Bi_side, n_roots=args.modes)
    lam = mu/args.R
    J0_mu=j0(mu); J1_mu=j1(mu)
    C_n = (2.0*J1_mu)/(mu*(J0_mu**2+J1_mu**2))

    def K_neu(z,t,lam_):
        t=max(float(t),1e-15); z=np.asarray(z,dtype=float)
        A = z/(2.0*np.sqrt(alpha*t)); B=lam_*np.sqrt(alpha*t)
        term1 = np.exp(-lam_*z)*erfc(A-B)
        term2 = np.exp(lam_*z - (A+B)**2)*erfcx(A+B)
        return 0.5/lam_*(term1 - term2)

    z_centers = (np.arange(nz)+0.5)*dx
    profiles_ana=[]; trace_ana=[]
    k_probe = int(np.clip(int(round(args.z_probe/dx - 0.5)), 0, nz-1))
    z_probe = (k_probe+0.5)*dx
    for tt in times:
        K_mat = np.array([K_neu(z_centers, tt, l) for l in lam])
        Tz = args.T_inf + (args.q0/args.k)*(C_n[:,None]*K_mat).sum(axis=0)
        profiles_ana.append(Tz)
        Kp = np.array([K_neu(z_probe, tt, l) for l in lam])
        Tp = args.T_inf + (args.q0/args.k)*(C_n*Kp).sum()
        trace_ana.append(float(Tp))

    # --- общая подготовка маски цилиндра ---
    def build_cyl(nx,ny,nz,dx,R):
        cx=(nx/2.0); cy=(ny/2.0)
        xs=(np.arange(nx)+0.5-cx)*dx; ys=(np.arange(ny)+0.5-cy)*dx
        X,Y=np.meshgrid(xs,ys,indexing='ij')
        mask2d = (np.sqrt(X**2+Y**2) <= R+1e-12)
        return np.repeat(mask2d[:,:,None], nz, axis=2), mask2d
    mask3d,_ = build_cyl(nx,ny,nz,dx,args.R)

    # --- функция-обёртка: один прогон численно на выбранном бэкенде ---
    def run_numeric(backend):
        if backend=="cpu":
            import adi3d_numba_coeff as adi  # твой модуль (:contentReference[oaicite:6]{index=6})
            grid = adi.Grid3D(nx,ny,nz,dx,mask3d)
            mat  = adi.Material(args.rho,args.cp,args.k)
        else:
            import adi3d_gpu_coeff as adi
            grid = adi.Grid3D(nx,ny,nz,dx,mask3d)
            mat  = adi.Material(args.rho,args.cp,args.k)

        # ГУ: верх — Dirichlet T_inf, бок — Robin(h_side), низ — Neumann q0 (нагрев внутрь)
        dir_mask = np.zeros((nx,ny,nz), dtype=bool); dir_mask[:,:,nz-1]=mask3d[:,:,nz-1]
        dir_val  = np.full((nx,ny,nz), args.T_inf, dtype=float)
        packs = adi.precompute_coeff_packs_unified(
            grid, mat,
            dir_mask=dir_mask, dir_value=dir_val,
            neumann={'z-': args.q0},
            robin_h={'x-': args.h_side, 'x+': args.h_side,
                     'y-': args.h_side, 'y+': args.h_side},
            robin_Tinf=args.T_inf
        )
        # численные параметры
        kappa = alpha; dt_cap = args.cfl*dx*dx/kappa
        params = adi.Params(dt=1e-3, theta=args.theta)

        # поле
        if backend=="cpu":
            T = np.full((nx,ny,nz), args.T_inf, dtype=float)
            params.dt = min(dt_cap, max(1e-6, times[0]/4.0))
            T = adi.adi_step_numba_coeff(T, grid, mat, params, packs, Tinf=args.T_inf)
        else:
            import cupy as cp
            T = cp.full((nx,ny,nz), args.T_inf, dtype=cp.float64)
            params.dt = min(dt_cap, max(1e-6, times[0]/4.0))
            T = adi.adi_step_gpu_coeff(T, grid, mat, params, packs, Tinf=args.T_inf)

        i0=nx//2; j0=ny//2
        profiles=[]; trace=[]; t_cur=0.0
        for tt in times:
            remain = float(tt-t_cur)
            nsub = max(1,int(math.ceil(remain/dt_cap))) if remain>0 else 0
            dt = remain/nsub if nsub>0 else 0.0
            params.dt = max(dt,1e-15)
            for _ in range(nsub):
                if backend=="cpu":
                    T = adi.adi_step_numba_coeff(T, grid, mat, params, packs, Tinf=args.T_inf)
                else:
                    T = adi.adi_step_gpu_coeff(T, grid, mat, params, packs, Tinf=args.T_inf)
            t_cur = tt
            if backend=="cpu":
                profiles.append(T[i0,j0,:].copy())
                trace.append(float(T[i0,j0,k_probe]))
            else:
                import cupy as cp
                profiles.append(cp.asnumpy(T[i0,j0,:]))
                trace.append(float(cp.asnumpy(T[i0,j0,k_probe])))
        return profiles, trace

    want = args.backend
    profiles_cpu=trace_cpu=None
    profiles_gpu=trace_gpu=None

    if want in ("cpu","both"):
        print("[run] CPU backend (numba)")
        t0 = time.perf_counter()
        profiles_cpu, trace_cpu = run_numeric("cpu")
        t1 = time.perf_counter()
        print(f"[time] CPU run took {t1 - t0:.3f} s")

    if want in ("gpu","both"):
        print("[run] GPU backend (cupy)")
        t0 = time.perf_counter()
        profiles_gpu, trace_gpu = run_numeric("gpu")
        # Синхронизация перед измерением конца (важно для GPU)
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        print(f"[time] GPU run took {t1 - t0:.3f} s")

    # --- рисунки: аналитика vs выбранный(е) бэкенд(ы) ---
    zc = z_centers
    if want in ("cpu","both"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.8,4.6))
        for tt,Ta,Tn in zip(times, profiles_ana, profiles_cpu):
            plt.plot(zc, Ta, lw=1.8, label=f"ana t={tt:.2g}s")
            plt.plot(zc, Tn, lw=1.2, ls="--", label=f"cpu t={tt:.2g}s")
        plt.xlabel("z, m"); plt.ylabel("T, °C")
        plt.title("Neumann q0>0; analytic vs CPU")
        plt.legend(ncol=2, fontsize=8); plt.tight_layout()
        plt.savefig("z_profiles_neumann_overlay_cpu.png", dpi=150)

        plt.figure(figsize=(7.2,4.2))
        plt.plot(times, trace_ana, lw=1.8, marker="o", label="ana")
        plt.plot(times, trace_cpu, lw=1.2, marker="s", label="cpu")
        plt.xlabel("t, s"); plt.ylabel("T, °C"); plt.legend()
        plt.title(f"r=0, z≈{z_probe:.4f} m — analytic vs CPU")
        plt.tight_layout(); plt.savefig("time_trace_neumann_overlay_cpu.png", dpi=150)

    if want in ("gpu","both"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.8,4.6))
        for tt,Ta,Tn in zip(times, profiles_ana, profiles_gpu):
            plt.plot(zc, Ta, lw=1.8, label=f"ana t={tt:.2g}s")
            plt.plot(zc, Tn, lw=1.2, ls="--", label=f"gpu t={tt:.2g}s")
        plt.xlabel("z, m"); plt.ylabel("T, °C")
        plt.title("Neumann q0>0; analytic vs GPU")
        plt.legend(ncol=2, fontsize=8); plt.tight_layout()
        plt.savefig("z_profiles_neumann_overlay_gpu.png", dpi=150)

        plt.figure(figsize=(7.2,4.2))
        plt.plot(times, trace_ana, lw=1.8, marker="o", label="ana")
        plt.plot(times, trace_gpu, lw=1.2, marker="^", label="gpu")
        plt.xlabel("t, s"); plt.ylabel("T, °C"); plt.legend()
        plt.title(f"r=0, z≈{z_probe:.4f} m — analytic vs GPU")
        plt.tight_layout(); plt.savefig("time_trace_neumann_overlay_gpu.png", dpi=150)

    if want == "both":
        import matplotlib.pyplot as plt
        diff = np.array(trace_gpu) - np.array(trace_cpu)
        rms  = float(np.sqrt(np.mean(diff**2)))
        mx   = float(np.max(np.abs(diff)))
        print(f"[cmp] CPU vs GPU @ probe: RMS={rms:.4g} °C; max|diff|={mx:.4g} °C")
        plt.figure(figsize=(7.2,4.2))
        plt.plot(times, trace_cpu, lw=1.5, marker="s", label="cpu")
        plt.plot(times, trace_gpu, lw=1.5, marker="^", label="gpu")
        plt.xlabel("t, s"); plt.ylabel("T, °C"); plt.legend()
        plt.title(f"CPU vs GPU (r=0, z≈{z_probe:.4f} m)")
        plt.tight_layout(); plt.savefig("time_trace_neumann_cpu_vs_gpu.png", dpi=150)

if __name__ == "__main__":
    main()
