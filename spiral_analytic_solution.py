"""Analytic reference solution for spiral layer deposition in an annular wall.

This module adapts the standalone script shared by the user so that tests can
call it programmatically.  The solution reconstructs the temperature field on a
(φ, z) grid located at a chosen radial probe (typically the mid-wall radius).

The formulation assumes:
  * Layers are deposited sequentially with a fixed time per full loop
    (``tau_dep``),
  * Each loop is discretised into ``n_phi_depo`` arc events of equal angular
    span, and
  * Robin boundary conditions act on the inner/outer radii and on the active
    end face of the growing wall.

The implementation keeps the number of modes configurable so tests can trade
accuracy vs. runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import optimize
from scipy.special import erfc, jv, jvp, yv, yvp


@dataclass(frozen=True)
class SpiralAnalyticConfig:
    """Collection of physical and numerical parameters for the analytic model."""

    k: float
    rho: float
    cp: float
    T_inf: float
    T_deposit: float
    inner_radius: float
    wall_thickness: float
    h_inner: float
    h_outer: float
    h_end: float
    base_height: float  # extent below z=0 (substrate already present)
    layer_height: float
    n_layers: int
    tau_dep: float
    n_phi_depo: int
    z_back: float
    z_front: float
    Nz: int
    Nphi: int
    M_ang: int
    Nr_modes: int
    probe_radius: float | None = None

    def annulus_bounds(self) -> Tuple[float, float]:
        a = self.inner_radius
        b = self.inner_radius + self.wall_thickness
        return a, b

    @property
    def alpha(self) -> float:
        return self.k / (self.rho * self.cp)

    @property
    def delta_T(self) -> float:
        return self.T_deposit - self.T_inf

    @property
    def total_height(self) -> float:
        return self.layer_height * self.n_layers

    @property
    def probe_r(self) -> float:
        if self.probe_radius is not None:
            return self.probe_radius
        a, b = self.annulus_bounds()
        return 0.5 * (a + b)


@dataclass
class SpiralAnalyticCache:
    kappas: Dict[int, np.ndarray]
    proj_radial: Dict[int, np.ndarray]
    radial_at_probe: Dict[int, np.ndarray]
    slice_events: List[Tuple[float, int, float]]
    phi_grid: np.ndarray
    z_grid: np.ndarray


def _det_robin_annulus(m: int, kap: float, cfg: SpiralAnalyticConfig) -> float:
    a, b = cfg.annulus_bounds()
    gamma_i = cfg.h_inner / cfg.k
    gamma_o = cfg.h_outer / cfg.k
    Ja, Ya = jv(m, kap * a), yv(m, kap * a)
    Jb, Yb = jv(m, kap * b), yv(m, kap * b)
    dJa, dYa = jvp(m, kap * a, 1), yvp(m, kap * a, 1)
    dJb, dYb = jvp(m, kap * b, 1), yvp(m, kap * b, 1)
    Ra1 = -kap * dJa - gamma_i * Ja
    Ra2 = -kap * dYa - gamma_i * Ya
    Rb1 = -kap * dJb - gamma_o * Jb
    Rb2 = -kap * dYb - gamma_o * Yb
    return Ra1 * Rb2 - Ra2 * Rb1


def _find_kappas_for_m(m: int, cfg: SpiralAnalyticConfig) -> np.ndarray:
    grid_pts = max(20000, 5 * cfg.Nr_modes)
    kap_max = 400.0
    xs = np.linspace(1e-6, kap_max, grid_pts)
    vals = _det_robin_annulus(m, xs, cfg)
    roots: List[float] = []
    sgn = np.sign(vals)
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        if np.isnan(vals[i]) or np.isnan(vals[i + 1]):
            continue
        if sgn[i] * sgn[i + 1] < 0:
            try:
                r = optimize.brentq(lambda u: _det_robin_annulus(m, u, cfg), a, b, maxiter=200)
            except ValueError:
                continue
            if not roots or abs(r - roots[-1]) > 1e-6:
                roots.append(r)
                if len(roots) >= cfg.Nr_modes:
                    break
    return np.array(roots, dtype=float)


def _build_radial_mode(m: int, kap: float, cfg: SpiralAnalyticConfig) -> Tuple[float, float]:
    a, b = cfg.annulus_bounds()
    gamma_i = cfg.h_inner / cfg.k
    Ja, Ya = jv(m, kap * a), yv(m, kap * a)
    dJa, dYa = jvp(m, kap * a, 1), yvp(m, kap * a, 1)
    Ra1 = -kap * dJa - gamma_i * Ja
    Ra2 = -kap * dYa - gamma_i * Ya
    B = 0.0 if abs(Ra2) < 1e-14 else -Ra1 / Ra2

    def R_raw(r: np.ndarray | float) -> np.ndarray:
        return jv(m, kap * r) + B * yv(m, kap * r)

    rs = np.linspace(a, b, 1024)
    w = np.gradient(rs)
    Rv = R_raw(rs)
    norm2 = np.sum((Rv * Rv) * rs * w)
    scale = 1.0 / np.sqrt(max(norm2, 1e-30))

    def R_norm(r: np.ndarray | float) -> np.ndarray:
        return scale * (jv(m, kap * r) + B * yv(m, kap * r))

    P = np.sum(Rv * rs * w) * scale
    R_probe = float(R_norm(cfg.probe_r))
    return P, R_probe


def _build_slice_events(cfg: SpiralAnalyticConfig) -> List[Tuple[float, int, float]]:
    events: List[Tuple[float, int, float]] = []
    dt_slice = cfg.tau_dep / cfg.n_phi_depo
    for layer in range(cfg.n_layers):
        t_layer = layer * cfg.tau_dep
        for p in range(cfg.n_phi_depo):
            t_evt = t_layer + (p + 0.5) * dt_slice
            phi0 = 2.0 * np.pi * (p + 0.5) / cfg.n_phi_depo
            events.append((t_evt, layer, phi0))
    events.sort(key=lambda x: x[0])
    return events


def build_cache(cfg: SpiralAnalyticConfig) -> SpiralAnalyticCache:
    kappas: Dict[int, np.ndarray] = {}
    proj_radial: Dict[int, np.ndarray] = {}
    radial_at_probe: Dict[int, np.ndarray] = {}
    for m in range(cfg.M_ang + 1):
        kap_list = _find_kappas_for_m(m, cfg)
        kappas[m] = kap_list
        if kap_list.size == 0:
            proj_radial[m] = np.zeros(0)
            radial_at_probe[m] = np.zeros(0)
            continue
        proj_list: List[float] = []
        probe_list: List[float] = []
        for kap in kap_list:
            P, R_probe = _build_radial_mode(m, kap, cfg)
            proj_list.append(P)
            probe_list.append(R_probe)
        proj_radial[m] = np.array(proj_list, dtype=float)
        radial_at_probe[m] = np.array(probe_list, dtype=float)

    phi_grid = np.linspace(0.0, 2.0 * np.pi, cfg.Nphi, endpoint=False)
    z_grid = np.linspace(-cfg.z_back, cfg.z_front, cfg.Nz)
    events = _build_slice_events(cfg)
    return SpiralAnalyticCache(kappas, proj_radial, radial_at_probe, events, phi_grid, z_grid)


def _G_R(s: np.ndarray, xi: np.ndarray, u: float, cfg: SpiralAnalyticConfig) -> np.ndarray:
    beta = cfg.h_end / cfg.k
    alpha = cfg.alpha
    denom = np.sqrt(4.0 * np.pi * alpha * u)
    spx = s + xi
    smx = s - xi
    ga = np.exp(-(smx * smx) / (4.0 * alpha * u)) / denom
    gb = np.exp(-(spx * spx) / (4.0 * alpha * u)) / denom
    corr = beta * np.exp(beta * spx + alpha * (beta ** 2) * u) * erfc(spx / (2.0 * np.sqrt(alpha * u)) + beta * np.sqrt(alpha * u))
    return ga + gb - corr


def temperature_phi_z_at_time(
    cfg: SpiralAnalyticConfig,
    cache: SpiralAnalyticCache,
    t: float,
) -> Tuple[np.ndarray, float, float]:
    """Return the analytic temperature map at radius ``cfg.probe_r``.

    Parameters
    ----------
    cfg, cache : configuration data produced by :func:`build_cache`.
    t : float
        Time instant at which to evaluate the field.

    Returns
    -------
    T_map : ndarray, shape (Nz, Nphi)
        Temperature at the requested radius.  NaNs mark void regions where the
        wall is not yet present.
    L_full : float
        Height of the fully deposited section at time ``t``.
    phi_progress : float
        Angular progress of the currently growing layer (0..2π).
    """

    Nz, Nphi = cfg.Nz, cfg.Nphi
    theta = np.full((Nz, Nphi), np.nan, dtype=float)

    full_layers = min(cfg.n_layers, int(np.floor(t / cfg.tau_dep)))
    L_full = full_layers * cfg.layer_height
    frac = 0.0
    if full_layers < cfg.n_layers:
        frac = max(0.0, min(1.0, (t - full_layers * cfg.tau_dep) / cfg.tau_dep))
    phi_progress = 2.0 * np.pi * frac

    z_grid = cache.z_grid
    phi_grid = cache.phi_grid
    exist = np.zeros((Nz, Nphi), dtype=bool)
    exist_base = (z_grid <= L_full) & (z_grid >= -cfg.base_height)
    exist[exist_base, :] = True
    if (full_layers < cfg.n_layers) and (frac > 0.0):
        z_low = L_full
        z_high = L_full + cfg.layer_height
        in_band = (z_grid >= z_low) & (z_grid <= z_high)
        phi_mask = phi_grid[None, :] < phi_progress
        exist[in_band, :] |= phi_mask

    rows_idx = np.where(exist.any(axis=1))[0]
    if rows_idx.size == 0:
        return cfg.T_inf + theta, L_full, phi_progress

    z_sel = z_grid[rows_idx]
    s_sel = L_full - z_sel
    Nz_exist = z_sel.size
    xi = np.linspace(0.0, cfg.layer_height, 64)
    w_xi = np.gradient(xi)

    A_cos = [np.zeros((len(cache.kappas[m]), Nz_exist)) for m in range(cfg.M_ang + 1)]
    A_sin = [np.zeros((len(cache.kappas[m]), Nz_exist)) for m in range(cfg.M_ang + 1)]

    for (t_evt, layer_idx, phi0) in cache.slice_events:
        if t_evt > t:
            break
        u = t - t_evt
        if u <= 0.0:
            continue
        if layer_idx < full_layers:
            offset = (full_layers - 1 - layer_idx) * cfg.layer_height
        elif layer_idx == full_layers:
            offset = 0.0
            if phi0 > phi_progress:
                continue
        else:
            continue

        s_mat = s_sel[:, None]
        xi_shift = xi[None, :] + offset
        Gmat = _G_R(s_mat, xi_shift, u, cfg)
        base_profile = (Gmat * w_xi).sum(axis=1)

        dphi = 2.0 * np.pi / cfg.n_phi_depo
        for m in range(cfg.M_ang + 1):
            kap_list = cache.kappas[m]
            if kap_list.size == 0:
                continue
            decay = np.exp(-cfg.alpha * (kap_list ** 2) * u)
            amp_rad = cfg.delta_T * cache.proj_radial[m] * cache.radial_at_probe[m] * decay
            weight_cos = (dphi / (2.0 * np.pi)) * np.cos(m * phi0)
            weight_sin = (dphi / (2.0 * np.pi)) * np.sin(m * phi0)
            if weight_cos != 0.0:
                A_cos[m] += (weight_cos * amp_rad[:, None]) * base_profile[None, :]
            if m > 0 and weight_sin != 0.0:
                A_sin[m] += (weight_sin * amp_rad[:, None]) * base_profile[None, :]

    cos_cache = {m: np.cos(m * phi_grid) for m in range(cfg.M_ang + 1)}
    sin_cache = {m: np.sin(m * phi_grid) for m in range(1, cfg.M_ang + 1)}

    theta_rows = np.zeros((Nz_exist, Nphi))
    for m in range(cfg.M_ang + 1):
        if cache.kappas[m].size == 0:
            continue
        Ac = A_cos[m].sum(axis=0)
        theta_rows += Ac[:, None] * cos_cache[m][None, :]
        if m > 0:
            As = A_sin[m].sum(axis=0)
            theta_rows += As[:, None] * sin_cache[m][None, :]

    theta[rows_idx, :] = np.where(exist[rows_idx, :], theta_rows, np.nan)
    return cfg.T_inf + theta, L_full, phi_progress


__all__ = [
    "SpiralAnalyticConfig",
    "SpiralAnalyticCache",
    "build_cache",
    "temperature_phi_z_at_time",
]
