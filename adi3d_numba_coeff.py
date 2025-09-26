# adi3d_numba_coeff.py
# -*- coding: utf-8 -*-
import numpy as np

try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco

class Grid3D:
    def __init__(self, nx, ny, nz, dx, mask):
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.dx = float(dx)
        self.mask = mask.astype(np.bool_, copy=True, order='C')
        assert self.mask.shape == (self.nx, self.ny, self.nz)

class Material:
    def __init__(self, rho, cp, k):
        self.rho = float(rho); self.cp=float(cp); self.k=float(k)

class Params:
    def __init__(self, dt, theta=0.5):
        self.dt=float(dt); self.theta=float(theta)

class AxisCoeffPack:
    def __init__(self, coeff, dir_mask, dir_val, qflux=None):
        self.coeff    = coeff.astype(np.float64, copy=True, order='C')
        self.dir_mask = dir_mask.astype(np.bool_,   copy=True, order='C')
        self.dir_val  = dir_val.astype(np.float64,  copy=True, order='C')
        if qflux is None:
            qflux = np.zeros_like(coeff, dtype=np.float64)
        self.qflux    = qflux.astype(np.float64,    copy=True, order='C')

def exposed_mask(mask, face):
    nx, ny, nz = mask.shape
    exp = np.zeros_like(mask, dtype=np.bool_)
    if face == 'x-':
        exp[1:,:,:] = mask[1:,:,:] & (~mask[:-1,:,:]); exp[0,:,:]  = mask[0,:,:]
    elif face == 'x+':
        exp[:-1,:,:] = mask[:-1,:,:] & (~mask[1:,:,:]); exp[-1,:,:] = mask[-1,:,:]
    elif face == 'y-':
        exp[:,1:,:] = mask[:,1:,:] & (~mask[:,:-1,:]); exp[:,0,:]  = mask[:,0,:]
    elif face == 'y+':
        exp[:,:-1,:] = mask[:,:-1,:] & (~mask[:,1:,:]); exp[:,-1,:] = mask[:,-1,:]
    elif face == 'z-':
        exp[:,:,1:] = mask[:,:,1:] & (~mask[:,:,:-1]); exp[:,:,0]  = mask[:,:,0]
    elif face == 'z+':
        exp[:,:,:-1] = mask[:,:,:-1] & (~mask[:,:,1:]); exp[:,:,-1] = mask[:,:,-1]
    else:
        raise ValueError("bad face")
    return exp

def precompute_coeff_packs_unified(grid, mat,
                                   dir_mask=None, dir_value=None,
                                   neumann=None,
                                   robin_h=None, robin_Tinf=None):
    """
    Build packs for ADI with unified BCs.
      - Dirichlet: dir_mask (bool 3D), dir_value (scalar or 3D)
      - Neumann: dict face-> q'' (W/m^2) applied on exposed boundary cells of that face;
                 SIGN: q''>0 means HEAT INTO THE SOLID (heating).
      - Robin: robin_h scalar/3D or dict face->(scalar/3D); ambient = robin_Tinf (used at solve)
               Implemented as volumetric sink term h*A/Ccell acting on exposed cells of the given face.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx = grid.dx; A = dx*dx; V = dx**3
    Ccell = mat.rho * mat.cp * V

    if dir_mask is None:
        dir_mask = np.zeros((nx,ny,nz), dtype=np.bool_)
    if dir_value is None:
        dir_value = np.zeros((nx,ny,nz), dtype=np.float64)
    elif np.isscalar(dir_value):
        dir_value = np.full((nx,ny,nz), float(dir_value), dtype=np.float64)

    def h_field_for_face(f):
        if robin_h is None: return None
        if isinstance(robin_h, dict):
            hv = robin_h.get(f, 0.0)
            if np.isscalar(hv): return np.full((nx,ny,nz), float(hv), dtype=np.float64)
            return hv.astype(np.float64, copy=False)
        else:
            if np.isscalar(robin_h): return np.full((nx,ny,nz), float(robin_h), dtype=np.float64)
            return robin_h.astype(np.float64, copy=False)

    coeff_x = np.zeros((nx,ny,nz), dtype=np.float64)
    coeff_y = np.zeros((nx,ny,nz), dtype=np.float64)
    coeff_z = np.zeros((nx,ny,nz), dtype=np.float64)
    for f, coeff_axis in (('x-', coeff_x), ('x+', coeff_x),
                          ('y-', coeff_y), ('y+', coeff_y),
                          ('z-', coeff_z), ('z+', coeff_z)):
        exp = exposed_mask(grid.mask, f)
        hf = h_field_for_face(f)
        if hf is not None:
            coeff_axis[exp] += (hf[exp] * A / Ccell)

    qx = np.zeros((nx,ny,nz), dtype=np.float64)
    qy = np.zeros((nx,ny,nz), dtype=np.float64)
    qz = np.zeros((nx,ny,nz), dtype=np.float64)
    if neumann is not None:
        for f, qv in neumann.items():
            if qv is None: continue
            exp = exposed_mask(grid.mask, f)
            if np.isscalar(qv): qarr = np.full((nx,ny,nz), float(qv), dtype=np.float64)
            else: qarr = qv.astype(np.float64, copy=False)
            S = np.zeros((nx,ny,nz), dtype=np.float64)
            S[exp] = qarr[exp] * A / Ccell   # ΔT contribution per 1 s: q''/(ρ c dx)
            if f[0]=='x': qx += S
            elif f[0]=='y': qy += S
            elif f[0]=='z': qz += S

    return (AxisCoeffPack(coeff_x, dir_mask, dir_value, qx),
            AxisCoeffPack(coeff_y, dir_mask, dir_value, qy),
            AxisCoeffPack(coeff_z, dir_mask, dir_value, qz))

@njit(cache=True)
def thomas_solve(a, b, c, d, n):
    for i in range(1, n):
        m = a[i] / b[i-1]
        b[i] = b[i] - m * c[i-1]
        d[i] = d[i] - m * d[i-1]
    x = np.empty(n, dtype=d.dtype)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x

@njit(cache=True)
def sweep_axis0(Tn, R0, mask, coeff_rob, dir_mask, dir_val, qflux, theta, gam, dt, kappa, Tinf):
    nx, ny, nz = Tn.shape
    out = R0.copy()
    for j in range(ny):
        for k in range(nz):
            cnt = 0
            for i in range(nx):
                if mask[i,j,k]: cnt += 1
            if cnt == 0: 
                continue
            a = np.zeros(cnt); b = np.zeros(cnt); c = np.zeros(cnt); d = np.zeros(cnt)
            idx = np.zeros(cnt, dtype=np.int64)
            s = 0
            for i in range(nx):
                if not mask[i,j,k]: continue
                idx[s] = i
                nnb = 0
                left = 0.0; right = 0.0
                if i-1 >= 0 and mask[i-1,j,k]:
                    left = -theta*gam; nnb += 1
                if i+1 < nx and mask[i+1,j,k]:
                    right = -theta*gam; nnb += 1
                diag = 1.0 + theta*gam*nnb + dt*coeff_rob[i,j,k]

                if dir_mask[i,j,k]:
                    a[s]=0.0; c[s]=0.0; b[s]=1.0; d[s]=dir_val[i,j,k]
                else:
                    a[s] = left; b[s] = diag; c[s] = right
                    # явный робин уже в R0; здесь только поток Неймана этой оси
                    d[s]  = out[i,j,k] + dt*qflux[i,j,k] + dt*coeff_rob[i,j,k]*Tinf
                s += 1
            x = thomas_solve(a,b,c,d,cnt)
            for m in range(cnt):
                out[idx[m], j, k] = x[m]
    return out

@njit(cache=True)
def sweep_axis1(Tn, R0, mask, coeff_rob, dir_mask, dir_val, qflux, theta, gam, dt, kappa, Tinf):
    nx, ny, nz = Tn.shape
    out = R0.copy()
    for i in range(nx):
        for k in range(nz):
            cnt = 0
            for j in range(ny):
                if mask[i,j,k]: cnt += 1
            if cnt == 0: continue
            a = np.zeros(cnt); b = np.zeros(cnt); c = np.zeros(cnt); d = np.zeros(cnt)
            idx = np.zeros(cnt, dtype=np.int64)
            s = 0
            for j in range(ny):
                if not mask[i,j,k]: continue
                idx[s] = j
                nnb = 0
                down = 0.0; up = 0.0
                if j-1 >= 0 and mask[i,j-1,k]:
                    down = -theta*gam; nnb += 1
                if j+1 < ny and mask[i,j+1,k]:
                    up = -theta*gam; nnb += 1
                diag = 1.0 + theta*gam*nnb + dt*coeff_rob[i,j,k]

                if dir_mask[i,j,k]:
                    a[s]=0.0; c[s]=0.0; b[s]=1.0; d[s]=dir_val[i,j,k]
                else:
                    a[s]=down; b[s]=diag; c[s]=up
                    d[s]  = out[i,j,k] + dt*qflux[i,j,k] + dt*coeff_rob[i,j,k]*Tinf
                s+=1
            x = thomas_solve(a,b,c,d,cnt)
            for m in range(cnt):
                out[i, idx[m], k] = x[m]
    return out

@njit(cache=True)
def sweep_axis2(Tn, R0, mask, coeff_rob, dir_mask, dir_val, qflux, theta, gam, dt, kappa, Tinf):
    nx, ny, nz = Tn.shape
    out = R0.copy()
    for i in range(nx):
        for j in range(ny):
            cnt = 0
            for k in range(nz):
                if mask[i,j,k]: cnt += 1
            if cnt == 0: continue
            a = np.zeros(cnt); b = np.zeros(cnt); c = np.zeros(cnt); d = np.zeros(cnt)
            idx = np.zeros(cnt, dtype=np.int64)
            s = 0
            for k in range(nz):
                if not mask[i,j,k]: continue
                idx[s] = k
                nnb = 0
                back = 0.0; front = 0.0
                if k-1 >= 0 and mask[i,j,k-1]:
                    back = -theta*gam; nnb += 1
                if k+1 < nz and mask[i,j,k+1]:
                    front = -theta*gam; nnb += 1
                diag = 1.0 + theta*gam*nnb + dt*coeff_rob[i,j,k]

                if dir_mask[i,j,k]:
                    a[s]=0.0; c[s]=0.0; b[s]=1.0; d[s]=dir_val[i,j,k]
                else:
                    a[s]=back; b[s]=diag; c[s]=front
                    d[s]  = out[i,j,k] + dt*qflux[i,j,k] + dt*coeff_rob[i,j,k]*Tinf
                s+=1
            x = thomas_solve(a,b,c,d,cnt)
            for m in range(cnt):
                out[i, j, idx[m]] = x[m]
    return out

@njit(cache=True)
def lap1D_x(T, mask, dx):
    nx, ny, nz = T.shape
    out = np.zeros_like(T)
    invdx2 = 1.0/(dx*dx)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not mask[i,j,k]: continue
                s = 0.0; c = 0.0
                if i-1>=0 and mask[i-1,j,k]:
                    s += T[i-1,j,k]; c += 1.0
                if i+1<nx and mask[i+1,j,k]:
                    s += T[i+1,j,k]; c += 1.0
                out[i,j,k] = (s - c*T[i,j,k]) * invdx2
    return out

@njit(cache=True)
def lap1D_y(T, mask, dx):
    nx, ny, nz = T.shape
    out = np.zeros_like(T)
    invdx2 = 1.0/(dx*dx)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not mask[i,j,k]: continue
                s = 0.0; c = 0.0
                if j-1>=0 and mask[i,j-1,k]:
                    s += T[i,j-1,k]; c += 1.0
                if j+1<ny and mask[i,j+1,k]:
                    s += T[i,j+1,k]; c += 1.0
                out[i,j,k] = (s - c*T[i,j,k]) * invdx2
    return out

@njit(cache=True)
def lap1D_z(T, mask, dx):
    nx, ny, nz = T.shape
    out = np.zeros_like(T)
    invdx2 = 1.0/(dx*dx)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not mask[i,j,k]: continue
                s = 0.0; c = 0.0
                if k-1>=0 and mask[i,j,k-1]:
                    s += T[i,j,k-1]; c += 1.0
                if k+1<nz and mask[i,j,k+1]:
                    s += T[i,j,k+1]; c += 1.0
                out[i,j,k] = (s - c*T[i,j,k]) * invdx2
    return out

def adi_step_numba_coeff(Tn, grid, mat, params, packs, Tinf=0.0):
    theta = params.theta; dt=params.dt; dx=grid.dx
    kappa = mat.k/(mat.rho*mat.cp); gam = kappa*dt/(dx*dx)
    packx, packy, packz = packs
    Lx = lap1D_x(Tn, grid.mask, dx)
    Ly = lap1D_y(Tn, grid.mask, dx)
    Lz = lap1D_z(Tn, grid.mask, dx)
    # явная часть Робина — один раз, суммарно по всем граням:
    R0 = Tn + dt*kappa*(1.0-theta)*(Lx+Ly+Lz)
    U = sweep_axis0(Tn, R0, grid.mask, packx.coeff, packx.dir_mask, packx.dir_val, packx.qflux, theta, gam, dt, kappa, Tinf)
    V = sweep_axis1(Tn, U,  grid.mask, packy.coeff, packy.dir_mask, packy.dir_val, packy.qflux, theta, gam, dt, kappa, Tinf)
    W = sweep_axis2(Tn, V,  grid.mask, packz.coeff, packz.dir_mask, packz.dir_val, packz.qflux, theta, gam, dt, kappa, Tinf)
    return W

def apply_surface_impulse_Q(T, grid, mat, Q, face='z-'):
    dx = grid.dx; dT = Q/(mat.rho*mat.cp*dx)
    exp = exposed_mask(grid.mask, face)
    if face == 'z-':
        T[:,:,0][exp[:,:,0]] += dT
    elif face == 'z+':
        T[:,:,-1][exp[:,:,-1]] += dT
    elif face == 'x-':
        T[0,:,:][exp[0,:,:]] += dT
    elif face == 'x+':
        T[-1,:,:][exp[-1,:,:]] += dT
    elif face == 'y-':
        T[:,0,:][exp[:,0,:]] += dT
    elif face == 'y+':
        T[:,-1,:][exp[:,-1,:]] += dT
    else:
        raise ValueError("bad face")
