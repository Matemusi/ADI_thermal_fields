# -*- coding: utf-8 -*-
# ADI 3D (θ-схема) на GPU: CuPy + пакетный метод Томаса вдоль осей.
import cupy as cp
import numpy as np  # только для удобной проверки скаляров

class Grid3D:
    def __init__(self, nx, ny, nz, dx, mask):
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.dx = float(dx)
        # маска сразу на GPU
        self.mask = cp.asarray(mask, dtype=cp.bool_, order='C')
        assert self.mask.shape == (self.nx, self.ny, self.nz)

class Material:
    def __init__(self, rho, cp_, k):
        self.rho = float(rho); self.cp=float(cp_); self.k=float(k)

class Params:
    def __init__(self, dt, theta=0.5):
        self.dt=float(dt); self.theta=float(theta)

class AxisCoeffPack:
    def __init__(self, coeff, dir_mask, dir_val, qflux=None):
        self.coeff    = cp.asarray(coeff,   dtype=cp.float64, order='C')
        self.dir_mask = cp.asarray(dir_mask,dtype=cp.bool_,   order='C')
        self.dir_val  = cp.asarray(dir_val, dtype=cp.float64, order='C')
        if qflux is None:
            qflux = cp.zeros_like(self.coeff, dtype=cp.float64)
        self.qflux    = cp.asarray(qflux,   dtype=cp.float64, order='C')

def _exposed_mask(mask, face):
    nx, ny, nz = mask.shape
    exp = cp.zeros_like(mask, dtype=cp.bool_)
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
    Полный аналог CPU-версии (:contentReference[oaicite:3]{index=3}), только на GPU.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx = grid.dx
    A = dx*dx; V = dx**3
    Ccell = mat.rho * mat.cp * V

    if dir_mask is None:
        dir_mask = cp.zeros((nx,ny,nz), dtype=cp.bool_)
    else:
        dir_mask = cp.asarray(dir_mask, dtype=cp.bool_)
    if dir_value is None:
        dir_value = cp.zeros((nx,ny,nz), dtype=cp.float64)
    elif np.isscalar(dir_value):
        dir_value = cp.full((nx,ny,nz), float(dir_value), dtype=cp.float64)
    else:
        dir_value = cp.asarray(dir_value, dtype=cp.float64)

    def _h_face(f):
        if robin_h is None: return None
        if isinstance(robin_h, dict):
            hv = robin_h.get(f, 0.0)
            if np.isscalar(hv): return cp.full((nx,ny,nz), float(hv), dtype=cp.float64)
            return cp.asarray(hv, dtype=cp.float64)
        else:
            if np.isscalar(robin_h): return cp.full((nx,ny,nz), float(robin_h), dtype=cp.float64)
            return cp.asarray(robin_h, dtype=cp.float64)

    coeff_x = cp.zeros((nx,ny,nz), dtype=cp.float64)
    coeff_y = cp.zeros((nx,ny,nz), dtype=cp.float64)
    coeff_z = cp.zeros((nx,ny,nz), dtype=cp.float64)
    for f, coeff_axis in (('x-', coeff_x), ('x+', coeff_x),
                          ('y-', coeff_y), ('y+', coeff_y),
                          ('z-', coeff_z), ('z+', coeff_z)):
        exp = _exposed_mask(grid.mask, f)
        hf = _h_face(f)
        if hf is not None:
            coeff_axis[exp] += (hf[exp] * A / Ccell)

    qx = cp.zeros((nx,ny,nz), dtype=cp.float64)
    qy = cp.zeros((nx,ny,nz), dtype=cp.float64)
    qz = cp.zeros((nx,ny,nz), dtype=cp.float64)
    if neumann is not None:
        for f, qv in neumann.items():
            if qv is None: continue
            exp = _exposed_mask(grid.mask, f)
            qarr = cp.full((nx,ny,nz), float(qv), dtype=cp.float64) if np.isscalar(qv) else cp.asarray(qv, dtype=cp.float64)
            S = cp.zeros((nx,ny,nz), dtype=cp.float64)
            S[exp] = qarr[exp] * A / Ccell
            if f[0]=='x': qx += S
            elif f[0]=='y': qy += S
            elif f[0]=='z': qz += S

    return (AxisCoeffPack(coeff_x, dir_mask, dir_value, qx),
            AxisCoeffPack(coeff_y, dir_mask, dir_value, qy),
            AxisCoeffPack(coeff_z, dir_mask, dir_value, qz))

# --- лапласианы (вклад только вдоль соответствующей оси), как в CPU-версии (:contentReference[oaicite:4]{index=4})
def _lap1D_x(T, mask, dx):
    invdx2 = 1.0/(dx*dx)
    s = cp.zeros_like(T); c = cp.zeros_like(T, dtype=cp.float64)
    m = (mask[1:,:,:] & mask[:-1,:,:])
    s[1:,:,:] += T[:-1,:,:] * m; c[1:,:,:] += m
    s[:-1,:,:] += T[1:,:,:] * m; c[:-1,:,:] += m
    return (s - c*T) * invdx2

def _lap1D_y(T, mask, dx):
    invdx2 = 1.0/(dx*dx)
    s = cp.zeros_like(T); c = cp.zeros_like(T, dtype=cp.float64)
    m = (mask[:,1:,:] & mask[:,:-1,:])
    s[:,1:,:] += T[:,:-1,:] * m; c[:,1:,:] += m
    s[:,:-1,:] += T[:,1:,:] * m; c[:,:-1,:] += m
    return (s - c*T) * invdx2

def _lap1D_z(T, mask, dx):
    invdx2 = 1.0/(dx*dx)
    s = cp.zeros_like(T); c = cp.zeros_like(T, dtype=cp.float64)
    m = (mask[:,:,1:] & mask[:,:,:-1])
    # сосед сверху
    s[:,:,1:]  += T[:,:,:-1] * m; c[:,:,1:]  += m
    # сосед снизу
    s[:,:,:-1] += T[:,:,1:]  * m; c[:,:,:-1] += m
    return (s - c*T) * invdx2

# --- пакетный метод Томаса вдоль первой оси массива (axis=0) ---
def _thomas_batch_axis0(a,b,c,d):
    n = a.shape[0]
    c_ = cp.zeros_like(c); d_ = cp.zeros_like(d)
    c_[0] = c[0]/b[0]; d_[0] = d[0]/b[0]
    for i in range(1, n):
        denom = b[i] - a[i]*c_[i-1]
        c_[i] = c[i]/denom
        d_[i] = (d[i] - a[i]*d_[i-1]) / denom
    x = cp.zeros_like(d)
    x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i]*x[i+1]
    return x

def _build_tridiag_for_axis(Tn, R0, mask, coeff_rob, dir_mask, dir_val, qflux, theta, gam, dt, Tinf, axis):
    # приводим массивы так, чтобы "ось решения" была axis=0
    if axis == 0:
        T = Tn; R = R0; M = mask; C = coeff_rob; Dm = dir_mask; Dv = dir_val; Q = qflux
    elif axis == 1:
        T = cp.transpose(Tn, (1,0,2)); R = cp.transpose(R0,(1,0,2))
        M = cp.transpose(mask,(1,0,2)); C = cp.transpose(coeff_rob,(1,0,2))
        Dm = cp.transpose(dir_mask,(1,0,2)); Dv = cp.transpose(dir_val,(1,0,2))
        Q = cp.transpose(qflux,(1,0,2))
    else:
        T = cp.transpose(Tn, (2,1,0)); R = cp.transpose(R0,(2,1,0))
        M = cp.transpose(mask,(2,1,0)); C = cp.transpose(coeff_rob,(2,1,0))
        Dm = cp.transpose(dir_mask,(2,1,0)); Dv = cp.transpose(dir_val,(2,1,0))
        Q = cp.transpose(qflux,(2,1,0))

    n0 = T.shape[0]
    a = cp.zeros_like(T); b = cp.zeros_like(T); c = cp.zeros_like(T); d = cp.zeros_like(T)

    # сосед слева/справа существует, если обе ячейки внутри маски
    m = (M[1:,:,:] & M[:-1,:,:])  # shape (n0-1, ...)

    left  = cp.zeros_like(T);  left[1:,:,:]  = (-theta*gam) * m
    right = cp.zeros_like(T);  right[:-1,:,:]= (-theta*gam) * m
    nnb   = cp.zeros_like(T, dtype=cp.float64)
    nnb[1:,:,:]  += m
    nnb[:-1,:,:] += m

    a  = left
    b  = 1.0 + theta*gam*nnb + dt*C
    c  = right
    d  = R + dt*Q + dt*C*Tinf

    # Дирихле
    a[Dm] = 0.0; c[Dm] = 0.0; b[Dm] = 1.0; d[Dm] = Dv[Dm]

    if axis == 0:   return a,b,c,d
    if axis == 1:   return tuple(cp.transpose(X,(1,0,2)) for X in (a,b,c,d))
    else:           return tuple(cp.transpose(X,(2,1,0)) for X in (a,b,c,d))

def _sweep_axis(Tn, R0, pack, mask, theta, gam, dt, dx, kappa, Tinf, axis):
    a,b,c,d = _build_tridiag_for_axis(Tn, R0, mask, pack.coeff, pack.dir_mask, pack.dir_val, pack.qflux,
                                      theta, gam, dt, Tinf, axis)
    # переупорядочим под решатель, где ось решения — 0
    if axis == 0:
        x = _thomas_batch_axis0(a,b,c,d)
        return x
    elif axis == 1:
        x = _thomas_batch_axis0(cp.transpose(a,(1,0,2)),
                                cp.transpose(b,(1,0,2)),
                                cp.transpose(c,(1,0,2)),
                                cp.transpose(d,(1,0,2)))
        return cp.transpose(x,(1,0,2))
    else:
        x = _thomas_batch_axis0(cp.transpose(a,(2,1,0)),
                                cp.transpose(b,(2,1,0)),
                                cp.transpose(c,(2,1,0)),
                                cp.transpose(d,(2,1,0)))
        return cp.transpose(x,(2,1,0))

def adi_step_gpu_coeff(Tn, grid, mat, params, packs, Tinf=0.0):
    theta = params.theta; dt = params.dt; dx = grid.dx
    kappa = mat.k/(mat.rho*mat.cp); gam = kappa*dt/(dx*dx)
    packx, packy, packz = packs

    # явная часть
    Lx = _lap1D_x(Tn, grid.mask, dx)
    Ly = _lap1D_y(Tn, grid.mask, dx)
    Lz = _lap1D_z(Tn, grid.mask, dx)
    R0 = Tn + dt*kappa*(1.0-theta)*(Lx+Ly+Lz)

    U = _sweep_axis(Tn, R0, packx, grid.mask, theta, gam, dt, dx, kappa, Tinf, axis=0)
    V = _sweep_axis(Tn, U,  packy, grid.mask, theta, gam, dt, dx, kappa, Tinf, axis=1)
    W = _sweep_axis(Tn, V,  packz, grid.mask, theta, gam, dt, dx, kappa, Tinf, axis=2)

    # вне маски возвращаем как было (на всякий случай)
    W = cp.where(grid.mask, W, Tn)
    return W
