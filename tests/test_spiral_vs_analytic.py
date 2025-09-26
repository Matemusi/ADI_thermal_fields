import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from adi3d_cyl_phi_v3 import Material, Params, RobinR, ZBC
from quick_spiral_deposition_gif_v5 import adi_step_masked, build_grid_annular
from spiral_analytic_solution import (
    SpiralAnalyticConfig,
    build_cache,
    temperature_phi_z_at_time,
)


def _run_numeric_simulation(times, cfg_numeric):
    grid, R_in, R_out, dz = build_grid_annular(
        cfg_numeric["R_out"],
        cfg_numeric["wall_thickness"],
        cfg_numeric["height"],
        cfg_numeric["z_back"],
        cfg_numeric["nr"],
        cfg_numeric["nphi"],
        dz_override=cfg_numeric["dz_override"],
    )
    mat = Material(cfg_numeric["rho"], cfg_numeric["cp"], cfg_numeric["k"])
    rob_outer = RobinR(cfg_numeric["h_side"], cfg_numeric["T_inf"])
    rob_inner = RobinR(cfg_numeric["h_side"], cfg_numeric["T_inf"])
    h_void = cfg_numeric.get("h_void", cfg_numeric["h_side"])
    rob_void = RobinR(h_void, cfg_numeric["T_inf"])
    zbc = ZBC(
        kind_bot="neumann0",
        kind_top="robin",
        h_top=cfg_numeric["h_end"],
        T_inf_top=cfg_numeric["T_inf"],
    )

    layer_cells = cfg_numeric["layer_cells"]
    layer_height = layer_cells * grid.dz
    iz_base = int(round(cfg_numeric["z_back"] / grid.dz))
    n_layers = cfg_numeric["n_layers"]
    loops = cfg_numeric["loops_per_layer"]

    T = np.full((grid.nr, grid.nphi, grid.nz), cfg_numeric["T_inf"], dtype=float)
    active = np.zeros((grid.nr, grid.nphi, grid.nz), dtype=bool)
    active[:, :, :iz_base] = True

    current_layer = 0
    current_loop = 0
    current_angle = 0.0
    current_iz = iz_base
    iz_max = grid.nz - 1

    dt = cfg_numeric["dt"]
    omega = cfg_numeric["omega"]
    prm = Params(dt, 1.0, "be")

    def mark_arc_on_layer(iz, ang0, ang1):
        if iz < 0 or iz > iz_max:
            return 0
        if ang1 <= ang0:
            return 0
        dphi = grid.dphi
        i0 = int(math.floor(ang0 / dphi))
        i1 = int(math.floor((ang1 - 1e-12) / dphi))
        if i1 < i0:
            i1 = i0
        added = 0
        for i in range(i0, i1 + 1):
            iphi = i % grid.nphi
            if not active[0, iphi, iz]:
                active[:, iphi, iz] = True
                T[:, iphi, iz] = cfg_numeric["T_deposit"]
                added += 1
        return added

    snapshots = []
    active_snaps = []
    t = 0.0
    eps = 1e-12
    for t_target in times:
        while t < t_target - eps:
            t_next = min(t + dt, t_target)
            angle_left = omega * (t_next - t)
            while angle_left > 0.0 and current_layer < n_layers:
                rem = (2.0 * math.pi) - current_angle
                seg = min(angle_left, rem)
                if seg > 0.0:
                    a0 = current_angle
                    a1 = current_angle + seg
                    mark_arc_on_layer(current_iz, a0, a1)
                    current_angle += seg
                    angle_left -= seg
                if current_angle >= (2.0 * math.pi - 1e-15):
                    current_angle = 0.0
                    current_loop += 1
                    if current_loop >= loops:
                        current_loop = 0
                        current_layer += 1
                        current_iz = iz_base + current_layer * layer_cells
                        if current_iz > iz_max:
                            current_layer = n_layers
                            break
            prm.dt = (t_next - t)
            T[:] = adi_step_masked(
                T,
                grid,
                mat,
                prm,
                rob_outer,
                zbc,
                active,
                robin_inner=rob_inner,
                robin_void=rob_void,
            )
            t = t_next
        snapshots.append(T.copy())
        active_snaps.append(active.copy())
    return grid, snapshots, active_snaps


def test_spiral_numeric_matches_analytic():
    k = 54.0
    rho = 7800.0
    cp = 490.0
    T_inf = 20.0
    T_deposit = 900.0
    R_in = 0.03
    wall_thickness = 0.002
    h_side = 400.0
    h_end = 500.0
    z_back = 0.02
    layer_height = 0.004
    n_layers = 2
    nphi = 36
    tau_dep = 2.0
    dt = tau_dep / nphi
    times = np.linspace(0.0, tau_dep * n_layers, 5)

    cfg_numeric = {
        "R_out": R_in + wall_thickness,
        "wall_thickness": wall_thickness,
        "height": layer_height * n_layers,
        "z_back": z_back,
        "nr": 6,
        "nphi": nphi,
        "dz_override": layer_height,
        "rho": rho,
        "cp": cp,
        "k": k,
        "h_side": h_side,
        "h_end": h_end,
        "T_inf": T_inf,
        "T_deposit": T_deposit,
        "h_void": h_side,
        "layer_cells": 1,
        "n_layers": n_layers,
        "loops_per_layer": 1,
        "dt": dt,
        "omega": 2.0 * math.pi / tau_dep,
    }

    grid, snapshots, active_snaps = _run_numeric_simulation(times, cfg_numeric)

    cfg_analytic = SpiralAnalyticConfig(
        k=k,
        rho=rho,
        cp=cp,
        T_inf=T_inf,
        T_deposit=T_deposit,
        inner_radius=R_in,
        wall_thickness=wall_thickness,
        h_inner=h_side,
        h_outer=h_side,
        h_end=h_end,
        base_height=z_back,
        layer_height=layer_height,
        n_layers=n_layers,
        tau_dep=tau_dep,
        n_phi_depo=nphi,
        z_back=z_back,
        z_front=layer_height * n_layers,
        Nz=grid.nz,
        Nphi=grid.nphi,
        M_ang=12,
        Nr_modes=6,
    )
    cache = build_cache(cfg_analytic)

    probe_r = cfg_analytic.probe_r
    ir_probe = int(np.abs(grid.r - probe_r).argmin())

    mean_tol = 60.0
    max_tol = 120.0

    for t, T_snap, active_snap in zip(times, snapshots, active_snaps):
        T_map_ana, _, _ = temperature_phi_z_at_time(cfg_analytic, cache, float(t))
        num_map = T_snap[ir_probe, :, :].T
        active_map = active_snap[ir_probe, :, :].T

        valid = np.isfinite(T_map_ana) & active_map
        if not np.any(valid):
            continue
        diff = np.abs(num_map - T_map_ana)
        mean_err = float(np.nanmean(diff[valid]))
        max_err = float(np.nanmax(diff[valid]))
        assert mean_err < mean_tol, f"mean error {mean_err:.2f}°C exceeds tolerance at t={t:.2f}s"
        assert max_err < max_tol, f"max error {max_err:.2f}°C exceeds tolerance at t={t:.2f}s"
