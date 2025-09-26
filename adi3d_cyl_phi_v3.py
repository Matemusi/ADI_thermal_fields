#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cylindrical heat ADI core (r, phi, z) with periodic phi and proper Robin BCs
at r = R and at the top/bottom z-faces (Neumann/Dirichlet/Robin supported).

This version ("v3") fixes causes of time-lag vs analytics:
1) Robin at r = R via ghost-cell elimination consistent with the finite-volume
   radial operator (no planar approximation).
2) Robin at z-top (and optional bottom) via ghost-cell elimination in the axial
   second-derivative stencil.

Default scheme is backward-Euler (scheme="be") for robustness. Optional
Douglas/Peaceman–Rachford (scheme="douglas", theta~0.5) is available.
"""
from __future__ import annotations
import numpy as np

# -------- optional numba ----------
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def deco(f): return f
        return deco

def _ensure_c(a: np.ndarray, dtype=np.float64) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=dtype)

# ---------- grid / material ----------
class GridCyl:
    def __init__(self, nr:int, nphi:int, nz:int, dr:float, dphi:float, dz:float, R:float):
        self.nr = int(nr); self.nphi = int(nphi); self.nz = int(nz)
        self.dr = float(dr); self.dphi = float(dphi); self.dz = float(dz)
        self.R = float(R)
        # cell centers and faces
        self.r = (np.arange(self.nr, dtype=np.float64) + 0.5)*self.dr
        self.r_imh = self.r - 0.5*self.dr
        self.r_iph = self.r + 0.5*self.dr
        # outer physical boundary lies at the outer face of the last cell
        self.r_outer_face = self.r_iph[-1]

class Material:
    def __init__(self, rho:float, cp:float, k:float):
        self.rho = float(rho); self.cp=float(cp); self.k=float(k)
    @property
    def alpha(self)->float:
        return self.k/(self.rho*self.cp)

class Params:
    def __init__(self, dt:float, theta:float=0.5, scheme:str="be"):
        self.dt=float(dt); self.theta=float(theta); self.scheme=str(scheme).lower()

class RobinR:
    def __init__(self, h:float, T_inf:float):
        self.h=float(h); self.T_inf=float(T_inf)

class ZBC:
    def __init__(self, kind_bot:str='neumann0', kind_top:str='robin',
                 h_bot:float=0.0, h_top:float=0.0,
                 T_inf_bot:float=20.0, T_inf_top:float=20.0,
                 T_bot:float=20.0, T_top:float=20.0):
        self.kind_bot=kind_bot; self.kind_top=kind_top
        self.h_bot=float(h_bot); self.h_top=float(h_top)
        self.T_inf_bot=float(T_inf_bot); self.T_inf_top=float(T_inf_top)
        self.T_bot=float(T_bot); self.T_top=float(T_top)

# ------------------ batched tridiagonal solvers ------------------
def _thomas_batch_np(a,b,c,d):
    m, n = d.shape
    x  = np.empty_like(d)
    cp = np.empty_like(c)
    dp = np.empty_like(d)
    # forward
    cp[:,0] = c[:,0]/b[:,0]
    dp[:,0] = d[:,0]/b[:,0]
    for i in range(1,n):
        denom = b[:,i] - a[:,i]*cp[:,i-1]
        cp[:,i] = np.where(i<n-1, c[:,i]/denom, 0.0)
        dp[:,i] = (d[:,i] - a[:,i]*dp[:,i-1])/denom
    # back
    x[:,n-1] = dp[:,n-1]
    for i in range(n-2,-1,-1):
        x[:,i] = dp[:,i] - cp[:,i]*x[:,i+1]
    return x

thomas_batch = _thomas_batch_np  # numpy path is fine and robust


def _cyclic_thomas_batch_np(a,b,c,d,alpha,beta):
    """
    Solve a batch of cyclic tridiagonal systems.
      For each row k:
        b[k,i] x_i + a[k,i] x_{i-1} + c[k,i] x_{i+1] = d[k,i],
      with wrap-around: + beta[k]*x_{n-1} in the first eq and + alpha[k]*x_0 in the last.
    alpha, beta are 1D arrays of length M (rows).
    """
    M, n = d.shape
    aw = a.copy(); bw = b.copy(); cw = c.copy()
    # incorporate wrap couplings into the tri-diagonal diagonals (Sherman–Morrison)
    bw[:,0]  -= beta.reshape(M)
    bw[:,-1] -= alpha.reshape(M)
    def solve_tri(aa,bb,cc,rr):
        aa=aa.copy(); bb=bb.copy(); cc=cc.copy(); rr=rr.copy()
        for i in range(1,n):
            mlt = aa[:,i]/bb[:,i-1]
            bb[:,i]  -= mlt*cc[:,i-1]
            rr[:,i]  -= mlt*rr[:,i-1]
        x = np.empty_like(rr)
        x[:,-1] = rr[:,-1]/bb[:,-1]
        for i in range(n-2,-1,-1):
            x[:,i] = (rr[:,i] - cc[:,i]*x[:,i+1])/bb[:,i]
        return x
    x_hat = solve_tri(aw,bw,cw,d)
    e = np.zeros_like(d); e[:,0] = 1.0; e[:,-1] = 1.0
    z = solve_tri(aw,bw,cw,e)
    denom = 1.0 + z[:,0] + beta*z[:,-1]
    mu = (x_hat[:,0] + beta*x_hat[:,-1]) / denom
    return x_hat - mu[:,None]*z

cyclic_thomas_batch = _cyclic_thomas_batch_np

# --------------- DISCRETE OPERATORS (used only in Douglas explicit) ---------------
def apply_Lr(T: np.ndarray, grid: GridCyl) -> np.ndarray:
    nr,nphi,nz = T.shape
    dr = grid.dr
    r = grid.r; r_imh = np.maximum(grid.r_imh,1e-15); r_iph = grid.r_iph
    Trp = np.empty_like(T); Trm=np.empty_like(T)
    Trp[:-1,:,:]=T[1:,:,:]; Trp[-1,:,:]=T[-1,:,:]
    Trm[1:,:,:]=T[:-1,:,:]; Trm[0,:,:]=T[0,:,:]
    flux_p = r_iph[:,None,None]*(Trp - T)/dr
    flux_m = r_imh[:,None,None]*(T - Trm)/dr
    out = (flux_p - flux_m)/(np.maximum(r,1e-15)[:,None,None]*dr)
    return out

def apply_Lphi(T: np.ndarray, grid: GridCyl) -> np.ndarray:
    dphi = grid.dphi
    if grid.nphi==1: return np.zeros_like(T)
    r = grid.r
    Tph = np.roll(T, -1, axis=1); Tmh = np.roll(T, +1, axis=1)
    out = (Tph - 2*T + Tmh)/(np.maximum(r,1e-12)[:,None,None]**2 * dphi*dphi)
    out[0,:,:] = 0.0  # regularity at axis
    return out

def apply_Lz_neumann0(T: np.ndarray, grid: GridCyl) -> np.ndarray:
    dz = grid.dz
    Tzp = np.empty_like(T); Tzm=np.empty_like(T)
    Tzp[:,:,1:] = T[:,:,1:]; Tzp[:,:,-1] = T[:,:,-1]
    Tzm[:,:,:-1] = T[:,:,:-1]; Tzm[:,:,0] = T[:,:,0]
    return (Tzp - 2*T + Tzm)/(dz*dz)

# ---------------- COEFFICIENT BUILDERS (implicit sweeps) ----------------
def build_coeff_r(grid: GridCyl, mat: Material, prm: Params, robin_r: RobinR,
                  theta: float, rhs: np.ndarray, Tn: np.ndarray):
    """
    Build tridiagonal matrices A_r for the r‑sweep of (I - θ dt α L_r) x = rhs
    with Robin at r = R imposed by ghost‑cell elimination that matches the
    finite‑volume operator.
    """
    nr, nphi, nz = grid.nr, grid.nphi, grid.nz
    alpha, dt, dr = mat.alpha, prm.dt, grid.dr
    r_i = np.maximum(grid.r, 1e-15)
    r_imh = np.maximum(grid.r_imh, 1e-15)
    r_iph = grid.r_iph

    fac = theta * alpha * dt
    M = nphi * nz
    a = np.zeros((M, nr), dtype=np.float64)
    b = np.zeros((M, nr), dtype=np.float64)
    c = np.zeros((M, nr), dtype=np.float64)

    # interior i = 1..nr-2
    ai = -fac * (r_imh[1:-1] / (r_i[1:-1] * dr*dr))
    ci = -fac * (r_iph[1:-1] / (r_i[1:-1] * dr*dr))
    bi = 1.0 - (ai + ci)
    a[:,1:-1] = ai
    b[:,1:-1] = bi
    c[:,1:-1] = ci

    # axis (Neumann0): flux through r=0 is zero
    a[:,0] = 0.0
    c0 = -fac * (r_iph[0] / (r_i[0]*dr*dr))
    b[:,0] = 1.0 - c0
    c[:,0] = c0

    # outer face: Robin  -k ∂T/∂r = h (T - T_inf) at r=R
    h = float(robin_r.h)
    aN = -fac * (r_imh[-1] / (r_i[-1] * dr*dr))            # multiplies T_{N-1}
    bN = 1.0 + fac * (r_imh[-1] / (r_i[-1] * dr*dr))       # base contribution
    if h != 0.0:
        bN += fac * (r_iph[-1] * (h/mat.k)) / (r_i[-1] * dr)    # Robin extra
    a[:, -1] = aN
    b[:, -1] = bN
    c[:, -1] = 0.0

    # RHS assembly
    rhs_r = np.moveaxis(rhs, 0, -1).reshape(M, nr).astype(np.float64, copy=True)
    if h != 0.0:
        rhs_r[:, -1] += fac * (r_iph[-1] * (h/mat.k)) / (r_i[-1] * dr) * robin_r.T_inf
    return a, b, c, rhs_r




def build_coeff_phi(grid: GridCyl, mat: Material, prm: Params, theta: float, rhs: np.ndarray):
    """
    Periodic phi-sweep builder (robust for nphi==1 and nphi>1).
    Returns a,b,c,d with shapes (M, nphi), and wrap vectors alpha,beta of shape (M,).
    M = nr * nz (rows are (ir, iz) pairs).
    """
    nr, nphi, nz = grid.nr, grid.nphi, grid.nz
    alpha, dt, dphi = mat.alpha, prm.dt, grid.dphi
    M = nr * nz

    # RHS rows: (nr, nphi, nz) -> (M, nphi)
    # Expect rhs is shaped (nr, nphi, nz). If not, transpose safely.
    if rhs.shape == (nr, nphi, nz):
        d = rhs.transpose(0,2,1).reshape(M, nphi).astype(np.float64, copy=True)
    elif rhs.shape == (nr, nz, nphi):
        d = rhs.reshape(nr*nz, nphi).astype(np.float64, copy=True)
    else:
        # last resort: move axis 1 (phi) to last
        d = np.moveaxis(rhs, 1, -1).reshape(M, nphi).astype(np.float64, copy=True)

    a = np.zeros((M, nphi), dtype=np.float64)
    b = np.zeros((M, nphi), dtype=np.float64)
    c = np.zeros((M, nphi), dtype=np.float64)
    alpha_wrap = np.zeros((M,), dtype=np.float64)
    beta_wrap  = np.zeros((M,), dtype=np.float64)

    if nphi == 1:
        b[:,0] = 1.0
        return a,b,c,d, alpha_wrap, beta_wrap

    # per-row fac depends on radius (r=0 row uses 0 for regularity)
    idx = 0
    for ir in range(nr):
        r = grid.r[ir]
        fac = 0.0 if ir == 0 else theta * alpha * dt / (r*r * dphi*dphi)
        ai = -fac; ci = -fac; bi = 1.0 - (ai + ci)
        for _ in range(nz):
            a[idx, :] = ai
            b[idx, :] = bi
            c[idx, :] = ci
            alpha_wrap[idx] = ai  # last->first coupling
            beta_wrap[idx]  = ci  # first->last coupling
            idx += 1

    # remove direct band couplings at the wrap positions; wrap handled by alpha/beta
    a[:, 0] = 0.0
    c[:, -1] = 0.0
    return a,b,c,d, alpha_wrap, beta_wrap
def build_coeff_z(grid: GridCyl, mat: Material, prm: Params, zbc: ZBC,
                  theta: float, rhs: np.ndarray, Tn: np.ndarray):
    nr, nphi, nz = grid.nr, grid.nphi, grid.nz
    alpha, dt, dz = mat.alpha, prm.dt, grid.dz
    M = nr * nphi
    a = np.zeros((M, nz), dtype=np.float64)
    b = np.zeros((M, nz), dtype=np.float64)
    c = np.zeros((M, nz), dtype=np.float64)
    # interior
    fac = theta * alpha * dt / (dz*dz)
    a[:,1:-1] = -fac
    b[:,1:-1] = 1.0 + 2.0*fac
    c[:,1:-1] = -fac
    d = rhs.reshape(M, nz).astype(np.float64, copy=True)

    # bottom
    if zbc.kind_bot == 'neumann0':
        a[:,0] = 0.0; b[:,0] = 1.0 + fac; c[:,0] = -fac
    elif zbc.kind_bot == 'dirichlet':
        a[:,0] = 0.0; b[:,0] = 1.0; c[:,0] = 0.0
        d[:,0] = zbc.T_bot
    elif zbc.kind_bot == 'robin':
        beta = zbc.h_bot/mat.k  # 1/m
        a[:,0] = 0.0
        b[:,0] = 1.0 + fac*(1.0 + beta*dz)
        c[:,0] = -fac
        d[:,0] += (theta*alpha*dt)*(beta/dz)*zbc.T_inf_bot
    else:
        raise ValueError("unknown zbc.kind_bot")

    # top
    if zbc.kind_top == 'neumann0':
        a[:,-1] = -fac; b[:,-1] = 1.0 + fac; c[:,-1] = 0.0
    elif zbc.kind_top == 'dirichlet':
        a[:,-1] = 0.0; b[:,-1] = 1.0; c[:,-1] = 0.0
        d[:,-1] = zbc.T_top
    elif zbc.kind_top == 'robin':
        beta = zbc.h_top/mat.k  # 1/m
        a[:,-1] = -fac; b[:,-1] = 1.0 + fac*(1.0 + beta*dz); c[:,-1] = 0.0
        d[:,-1] += (theta*alpha*dt)*(beta/dz)*zbc.T_inf_top
    else:
        raise ValueError("unknown zbc.kind_top")

    return a,b,c,d


# ---------------- SPECTRAL φ-SWEEP (robust for nphi>1) ----------------
def phi_solve_spectral(Tin: np.ndarray, grid: GridCyl, mat: Material, theta: float, dt: float) -> np.ndarray:
    """
    Solve (I - theta*dt*alpha*L_phi) X = Tin along periodic φ using FFT.
    For each radius r_i, eigenvalues are λ_k = 1 + 2*fac_i*(1 - cos(2πk/nphi)),
    where fac_i = theta*alpha*dt / (r_i^2 * dphi^2). For r=0 (axis), fac=0 => identity.
    """
    nr, nphi, nz = Tin.shape
    if nphi == 1:
        return Tin.copy()
    r = grid.r.copy()
    dphi = grid.dphi
    alpha = mat.alpha
    fac = np.zeros(nr, dtype=np.float64)
    # fac[0]=0 for axis; others per formula
    for ir in range(1, nr):
        fac[ir] = theta * alpha * dt / (r[ir]*r[ir] * dphi * dphi)
    # eigenvalues for rfft freqs k = 0..nphi//2
    k = np.arange(nphi//2 + 1, dtype=np.float64)
    cosk = np.cos(2.0*np.pi*k/float(nphi))  # shape (K,)
    # broadcast lambdas: (nr, K) = 1 + 2*fac[:,None]*(1 - cosk[None,:])
    lam = 1.0 + 2.0 * fac[:, None] * (1.0 - cosk[None, :])
    # FFT along phi, solve per (ir, k, iz), iFFT back
    # reshape to (nr, nphi, nz) -> FFT -> (nr, K, nz)
    F = np.fft.rfft(Tin, axis=1)
    # divide with broadcasting: (nr, K, nz) / (nr, K, 1)
    F /= lam[:, :, None]
    X = np.fft.irfft(F, n=nphi, axis=1)
    return X

# ---------------- TIME STEP ----------------------
def adi_step(Tn: np.ndarray, grid: GridCyl, mat: Material, prm: Params,
             robin_r: RobinR, zbc: ZBC, S: np.ndarray | None = None,
             theta: float | None = None) -> np.ndarray:
    alpha = mat.alpha; dt = prm.dt
    scheme = prm.scheme if prm.scheme in ("be","douglas") else "be"

    if scheme == "be":
        R0 = Tn + (dt*(S/(mat.rho*mat.cp)) if S is not None else 0.0)
        # r-implicit
        a,b,c,d = build_coeff_r(grid, mat, Params(dt,1.0,"be"), robin_r, 1.0, R0.copy(), Tn)
        nr, nphi, nz = grid.nr, grid.nphi, grid.nz
        X = thomas_batch(a,b,c,d)  # (M, nr), M=nphi*nz
        TR = np.moveaxis(X.reshape(nphi, nz, nr), -1, 0)  # -> (nr,nphi,nz)
        # phi-implicit
        Tphi = phi_solve_spectral(TR, grid, mat, 1.0, dt)
        # z-implicit
        a,b,c,d = build_coeff_z(grid, mat, Params(dt,1.0,"be"), zbc, 1.0, Tphi.copy(), Tn)
        Tnp1 = thomas_batch(a,b,c,d).reshape(nr,nphi,nz)
        return Tnp1

    # Douglas / Peaceman–Rachford (optional)
    th = prm.theta if (theta is None) else float(theta)
    if th <= 0.0 or th > 1.0: th = 0.5
    # Operators on T^n (no BC here)
    Lr_Tn   = apply_Lr(Tn, grid)
    Lphi_Tn = apply_Lphi(Tn, grid)
    Lz_Tn   = apply_Lz_neumann0(Tn, grid)  # BC in matrices
    Y0 = Tn + dt*(Lr_Tn + Lphi_Tn + Lz_Tn)
    # r
    a,b,c,d = build_coeff_r(grid, mat, Params(dt,th,"douglas"), robin_r, th, Y0.copy(), Tn)
    nr,nphi,nz = grid.nr, grid.nphi, grid.nz
    Xr = thomas_batch(a,b,c,d)  # (M, nr)
    Xr = np.moveaxis(Xr.reshape(nphi, nz, nr), -1, 0)  # (nr,nphi,nz)
    # phi
    Y1 = Xr + (1-th)*dt*apply_Lr(Tn, grid)
    Xphi = phi_solve_spectral(Y1, grid, mat, th, dt)
    # z
    Y2 = Xphi + (1-th)*dt*apply_Lphi(Tn, grid)
    a,b,c,d = build_coeff_z(grid, mat, Params(dt,th,"douglas"), zbc, th, Y2.copy(), Tn)
    Xz = thomas_batch(a,b,c,d).reshape(nr, nphi, nz)
    Tnp1 = Xz + (1-th)*dt*apply_Lz_neumann0(Tn, grid)
    return Tnp1
