
# quick_compare_robin_end_robin.py
# -*- coding: utf-8 -*-
"""
Robin HEATING at end face z=0 (step in ambient): T_amb_end = T_inf + Delta_end
Robin on lateral surface (h_side, ambient T_inf). Top z=zmax: Dirichlet T_inf.
Axisymmetric cylinder of radius R, semi-infinite in z approximated by a long domain.
Compares analytic solution (modal sum) vs ADI numeric solution.

Usage example (matches your Neumann test scales):
%run quick_compare_robin_end_robin.py \
  --R 0.02 --h_side 500 --h_end 800 --k 54 --rho 7800 --cp 490 --T_inf 20 \
  --Delta_end 200 \
  --nxr 64 --nz 1600 --zmax 1.2 --modes 25 \
  --tmin 0.01 --tmax 300 --nframes 6 --z_probe 0.005 \
  --theta 0.5 --cfl 3000
"""
import math, argparse, numpy as np

def main():
    p = argparse.ArgumentParser(description="Robin heating at z=0 (step in ambient); side Robin; top Dirichlet")
    # material & geometry
    p.add_argument("--k",      type=float, default=54.0)
    p.add_argument("--rho",    type=float, default=7800.0)
    p.add_argument("--cp",     type=float, default=490.0)
    p.add_argument("--R",      type=float, default=0.02)
    # BCs
    p.add_argument("--h_side", type=float, default=500.0, help="Side Robin coefficient, W/m^2/K")
    p.add_argument("--h_end",  type=float, default=800.0, help="End-face Robin coefficient, W/m^2/K (heating face)")
    p.add_argument("--T_inf",  type=float, default=20.0, help="Ambient for sides/top (°C)")
    p.add_argument("--Delta_end", type=float, default=200.0, help="Ambient STEP at end face, K; T_amb_end=T_inf+Delta_end")
    # numerics/grids
    p.add_argument("--nxr",    type=int,   default=64)
    p.add_argument("--nz",     type=int,   default=160)
    p.add_argument("--zmax",   type=float, default=0.12)
    p.add_argument("--tmin",   type=float, default=0.01)
    p.add_argument("--tmax",   type=float, default=5.0)
    p.add_argument("--nframes",type=int,   default=6)
    p.add_argument("--z_probe",type=float, default=0.04)
    p.add_argument("--modes",  type=int,   default=25)
    p.add_argument("--theta",  type=float, default=0.5)
    p.add_argument("--cfl",    type=float, default=2.0)
    args = p.parse_args()

    import matplotlib.pyplot as plt
    from scipy.special import j0, j1, jvp, erfc
    import scipy.optimize as opt
    import adi3d_numba_coeff as adi

    # --- material / geometry ---
    alpha = args.k/(args.rho*args.cp)
    dx = args.R/float(args.nxr)
    nx = ny = int(round((2.0*args.R)/dx))
    nz = args.nz if args.nz>0 else int(round(args.zmax/dx))
    zmax = nz*dx
    print(f"[grid] nx=ny={nx}, nz={nz}, dx={dx:.6e} m; zmax={zmax:.3f} m")
    Bi_side = args.h_side*args.R/args.k
    print(f"[phys] alpha={alpha:.3e} m^2/s, Bi_side={Bi_side:.3f}, beta_end={args.h_end/args.k:.3f}")

    times = np.linspace(args.tmin, args.tmax, args.nframes)

    # --- radial eigenvalues for side Robin ---
    def robin_mu_roots(Bi, n_roots=20, mu_max=220.0, grid_pts=40000):
        def f(mu): return mu*jvp(0, mu, 1) + Bi*j0(mu)
        mus = np.linspace(1e-8, mu_max, grid_pts); vals=f(mus)
        roots=[]
        for i in range(len(mus)-1):
            a,b=mus[i],mus[i+1]; fa,fb=vals[i],vals[i+1]
            if np.isnan(fa) or np.isnan(fb): continue
            if fa*fb<0:
                try:
                    r=opt.brentq(f,a,b,maxiter=200)
                    if len(roots)==0 or abs(r-roots[-1])>1e-10:
                        roots.append(r)
                        if len(roots)>=n_roots: break
                except ValueError:
                    pass
        return np.array(roots,dtype=float)

    mu = robin_mu_roots(Bi_side, n_roots=args.modes)
    lam = mu/args.R
    J0_mu=j0(mu); J1_mu=j1(mu)
    # projection of "1" over disk
    C_n = (2.0*J1_mu)/(mu*(J0_mu**2+J1_mu**2))

    # --- analytic kernel: Robin step (heating) at end face ---
    beta = args.h_end/args.k
    Delta = args.Delta_end  # K

    def H_R_step(z, t, lam_val, beta, alpha, eps=1e-12):
        """
        Kernel for 'end-face Robin step' (ambient T_inf+Delta at z=0 for t>0).
        """
        t = max(float(t), 1e-16)
        A = z/(2.0*np.sqrt(alpha*t))
        B = lam_val*np.sqrt(alpha*t)

        den_p = (beta + lam_val); den_m = (beta - lam_val); den_b = (beta**2 - lam_val**2)
        if abs(den_p) < eps: den_p = np.sign(den_p) if den_p!=0 else 1.0; den_p *= eps
        if abs(den_m) < eps: den_m = np.sign(den_m) if den_m!=0 else 1.0; den_m *= eps
        if abs(den_b) < eps: den_b = np.sign(den_b) if den_b!=0 else 1.0; den_b *= eps

        term1 = np.exp(-lam_val*z) * erfc(A - B) / (2.0*den_p)
        term2 = np.exp(+lam_val*z) * erfc(A + B) / (2.0*den_m)
        term3 = (beta/den_b) * np.exp(beta*z + alpha*(beta**2)*t) * erfc(A + beta*np.sqrt(alpha*t)) \
                * np.exp(-alpha*(lam_val**2)*t)

        return beta * (term1 + term2 - term3)

    # z grid for profiles (cell centers along axis r=0):
    z_centers = (np.arange(nz)+0.5)*dx

    # --- analytic profiles and probe trace (r=0) ---
    profiles_ana=[]; trace_ana=[]
    k_probe = int(np.clip(int(round(args.z_probe/dx - 0.5)), 0, nz-1))
    z_probe = (k_probe+0.5)*dx
    for tt in times:
        H_mat = np.array([H_R_step(z_centers, tt, l, beta, alpha) for l in lam])
        Tz = args.T_inf + Delta * (C_n[:,None] * H_mat).sum(axis=0)
        profiles_ana.append(Tz)
        Hp = np.array([H_R_step(z_probe, tt, l, beta, alpha) for l in lam])
        Tp = args.T_inf + Delta * (C_n * Hp).sum()
        trace_ana.append(float(Tp))

    # --- build cylinder mask ---
    def build_cyl(nx,ny,nz,dx,R):
        cx=(nx/2.0); cy=(ny/2.0)
        xs=(np.arange(nx)+0.5-cx)*dx; ys=(np.arange(ny)+0.5-cy)*dx
        X,Y=np.meshgrid(xs,ys,indexing='ij')
        mask2d = (np.sqrt(X**2+Y**2) <= R+1e-12)
        return np.repeat(mask2d[:,:,None], nz, axis=2), mask2d
    mask3d,_ = build_cyl(nx,ny,nz,dx,args.R)

    grid = adi.Grid3D(nx,ny,nz,dx,mask3d)
    mat  = adi.Material(args.rho,args.cp,args.k)

    # --- BCs for ADI ---
    # Top Dirichlet (z=zmax): T=T_inf
    dir_mask = np.zeros((nx,ny,nz), dtype=bool)
    dir_mask[:,:,nz-1] = mask3d[:,:,nz-1]
    dir_val = np.full((nx,ny,nz), args.T_inf, dtype=float)

    # Side Robin: h_side, ambient T_inf (global Tinf passed into ADI step)
    # End-face z- Robin: h_end with ambient T_inf, PLUS an effective extra Neumann flux
    # q_add = h_end*(T_amb_end - T_inf) = h_end * Delta, which converts global-ambient Robin to desired Tamb_end
    q_add = args.h_end * Delta  # W/m^2
    packs = adi.precompute_coeff_packs_unified(
        grid, mat,
        dir_mask=dir_mask, dir_value=dir_val,
        neumann={'z-': q_add},
        robin_h={'x-': args.h_side, 'x+': args.h_side,
                 'y-': args.h_side, 'y+': args.h_side,
                 'z-': args.h_end},
        robin_Tinf=args.T_inf
    )
    print("[bc] Robin heating at end: implemented as Robin(z-)=h_end with ambient T_inf + extra Neumann q_add=h_end*Delta; side Robin; top Dirichlet @ T_inf")
    # sanity print
    qz = packs[2].qflux
    bot = qz[:,:,0][mask3d[:,:,0]]
    print(f"[sanity] bottom q_add A/Ccell: mean={np.mean(bot):.3e}, min={np.min(bot):.3e}, max={np.max(bot):.3e} (K/s contribution)")

    # --- time stepping ---
    kappa = alpha; dt_cap = args.cfl*dx*dx/kappa
    params = adi.Params(dt=1e-3, theta=args.theta)

    Tcur = np.full((nx,ny,nz), args.T_inf, dtype=float)
    # warmup small step for numba JIT & stability
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
            print(f"[sanity] z- center T(k=0) = {Tcur[i0,j0,0]:.2f} °C (should be > {args.T_inf:.1f} °C)")

    # --- plots ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7.8,4.6))
    for tt,Ta,Tn in zip(times, profiles_ana, profiles_num):
        plt.plot(z_centers, Ta, lw=1.8, label=f"ana t={tt:.2g}s")
        plt.plot(z_centers, Tn, lw=1.2, ls="--", label=f"num t={tt:.2g}s")
    plt.xlabel("z, m"); plt.ylabel("T, °C")
    plt.title("Robin heating @ z=0, side Robin; T(z) at r=0: analytic vs ADI")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.savefig("z_profiles_robin_end_overlay.png", dpi=150)
    print("Saved: z_profiles_robin_end_overlay.png")

    plt.figure(figsize=(7.2,4.2))
    plt.plot(times, trace_ana, lw=1.8, marker="o", label="ana")
    plt.plot(times, trace_num, lw=1.2, marker="s", label="num")
    plt.xlabel("t, s"); plt.ylabel("T, °C"); plt.title(f"Robin end heating, r=0, z≈{z_probe:.4f} m")
    plt.legend(); plt.tight_layout(); plt.savefig("time_trace_robin_end_overlay.png", dpi=150)
    print("Saved: time_trace_robin_end_overlay.png")

if __name__ == "__main__":
    main()
