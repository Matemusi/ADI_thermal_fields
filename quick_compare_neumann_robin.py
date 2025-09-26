# quick_compare_neumann_robin.py
# -*- coding: utf-8 -*-
import math, argparse, numpy as np

def main():
    p = argparse.ArgumentParser(description="Neumann heating at z=0: q0>0 heats into body; side Robin; top Dirichlet")
    p.add_argument("--k",      type=float, default=54.0)
    p.add_argument("--rho",    type=float, default=7800.0)
    p.add_argument("--cp",     type=float, default=490.0)
    p.add_argument("--R",      type=float, default=0.02)
    p.add_argument("--h_side", type=float, default=500.0)
    p.add_argument("--T_inf",  type=float, default=20.0)
    p.add_argument("--q0",     type=float, default=2.0e5)
    p.add_argument("--nxr",    type=int,   default=64)
    p.add_argument("--nz",     type=int,   default=160)
    p.add_argument("--zmax",   type=float, default=0.12)
    p.add_argument("--tmin",   type=float, default=0.01)
    p.add_argument("--tmax",   type=float, default=5.0)
    p.add_argument("--nframes",type=int,   default=6)
    p.add_argument("--z_probe",type=float, default=0.04)
    p.add_argument("--modes",  type=int,   default=18)
    p.add_argument("--theta",  type=float, default=0.5)
    p.add_argument("--cfl",    type=float, default=2.0)
    args = p.parse_args()

    import matplotlib.pyplot as plt
    from scipy.special import j0, j1, jvp, erfc, erfcx
    import scipy.optimize as opt
    import adi3d_numba_coeff as adi

    alpha = args.k/(args.rho*args.cp)
    dx = args.R/float(args.nxr)
    nx = ny = int(round((2.0*args.R)/dx))
    nz = args.nz if args.nz>0 else int(round(args.zmax/dx))
    zmax = nz*dx
    print(f"[grid] nx=ny={nx}, nz={nz}, dx={dx:.6e} m; zmax={zmax:.3f} m")
    print(f"[phys] alpha={alpha:.3e} m^2/s, Bi_side={args.h_side*args.R/args.k:.3f}")

    times = np.linspace(args.tmin, args.tmax, args.nframes)

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
        return 0.5/lam_*(term1 - term2)  # minus => heating for q0>0

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

    def build_cyl(nx,ny,nz,dx,R):
        cx=(nx/2.0); cy=(ny/2.0)
        xs=(np.arange(nx)+0.5-cx)*dx; ys=(np.arange(ny)+0.5-cy)*dx
        X,Y=np.meshgrid(xs,ys,indexing='ij')
        mask2d = (np.sqrt(X**2+Y**2) <= R+1e-12)
        return np.repeat(mask2d[:,:,None], nz, axis=2), mask2d
    mask3d,_ = build_cyl(nx,ny,nz,dx,args.R)
    grid = adi.Grid3D(nx,ny,nz,dx,mask3d)
    mat  = adi.Material(args.rho,args.cp,args.k)

    dir_mask = np.zeros((nx,ny,nz), dtype=bool)
    dir_mask[:,:,nz-1] = mask3d[:,:,nz-1]
    dir_val = np.full((nx,ny,nz), args.T_inf, dtype=float)

    packs = adi.precompute_coeff_packs_unified(
        grid, mat,
        dir_mask=dir_mask, dir_value=dir_val,
        neumann={'z-': args.q0},
        robin_h={'x-': args.h_side, 'x+': args.h_side,
                 'y-': args.h_side, 'y+': args.h_side},
        robin_Tinf=args.T_inf
    )
    print("[bc] Neumann heating: using q0>0 as HEATING into the body; Robin on sides; Dirichlet @ top")
    qz = packs[2].qflux
    bot = qz[:,:,0][mask3d[:,:,0]]
    print(f"[sanity] bottom qflux A/Ccell: mean={np.mean(bot):.3e}, min={np.min(bot):.3e}, max={np.max(bot):.3e}")

    kappa = alpha; dt_cap = args.cfl*dx*dx/kappa
    params = adi.Params(dt=1e-3, theta=args.theta)
    dT_est = min((times[1]-times[0]) if len(times)>1 else times[0], dt_cap)*args.q0/(args.rho*args.cp*dx)
    print(f"[sanity] est ΔT first step ≈ {dT_est:.3f} K (>0 expected)")

    Tcur = np.full((nx,ny,nz), args.T_inf, dtype=float)
    # warmup small step
    params.dt = min(dt_cap, max(1e-6, times[0]/4.0))
    Tcur = adi.adi_step_numba_coeff(Tcur, grid, mat, params, packs, Tinf=args.T_inf)

    i0=nx//2; j0=ny//2
    profiles_num=[]; trace_num=[]
    t_cur=0.0
    for ti, tt in enumerate(times):
        remain = float(tt-t_cur)
        nsub = max(1,int(math.ceil(remain/dt_cap))) if remain>0 else 0
        dt = remain/nsub if nsub>0 else 0.0
        params.dt = max(dt,1e-15)
        for _ in range(nsub):
            Tcur = adi.adi_step_numba_coeff(Tcur, grid, mat, params, packs, Tinf=args.T_inf)
        t_cur = tt
        profiles_num.append(Tcur[i0,j0,:].copy())
        trace_num.append(Tcur[i0,j0,k_probe].item())
        print(f"[numeric] t={t_cur:.4g}s, sub={nsub}, dt={params.dt:.3g}s, T∈[{Tcur.min():.2f},{Tcur.max():.2f}]")
        if ti==0:
            print(f"[sanity] bottom-center T(k=0) = {Tcur[i0,j0,0]:.2f} °C (should be > {args.T_inf} °C)")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.8,4.6))
    for tt,Ta,Tn in zip(times, profiles_ana, profiles_num):
        plt.plot(z_centers, Ta, lw=1.8, label=f"ana t={tt:.2g}s")
        plt.plot(z_centers, Tn, lw=1.2, ls="--", label=f"num t={tt:.2g}s")
    plt.xlabel("z, m"); plt.ylabel("T, °C")
    plt.title("Neumann q0>0 @ z=0 (heating), side Robin; T(z) at r=0: analytic vs ADI")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.savefig("z_profiles_neumann_overlay.png", dpi=150)
    print("Saved: z_profiles_neumann_overlay.png")

    plt.figure(figsize=(7.2,4.2))
    plt.plot(times, trace_ana, lw=1.8, marker="o", label="ana")
    plt.plot(times, trace_num, lw=1.2, marker="s", label="num")
    plt.xlabel("t, s"); plt.ylabel("T, °C"); plt.title(f"Neumann q0>0, r=0, z≈{z_probe:.4f} m")
    plt.legend(); plt.tight_layout(); plt.savefig("time_trace_neumann_overlay.png", dpi=150)
    print("Saved: time_trace_neumann_overlay.png")

if __name__ == "__main__":
    main()
