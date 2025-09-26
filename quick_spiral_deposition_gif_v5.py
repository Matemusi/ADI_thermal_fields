
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WAAM tube deposition (layer-by-layer rings), fixed-dt masked ADI (r,phi,z).

- One or more full circumferential loops are deposited at a *fixed* z-layer,
  then the nozzle steps up by `layer_cells_z` cells and repeats.
- Inside each fixed time step dt, we activate all (phi, z) columns covered by the
  nozzle arc during this dt on the current layer (and possibly finish the ring
  and start the next layer if the arc crosses 2π).
- Heat solves use adi_step_masked (no Dirichlet in void), Robin at material/void.

CLI highlights:
  --layer_cells_z  number of z-cells per layer (default 1 ⇒ layer_height=dz)
  --loops_per_layer number of loops at each layer (default 1)
  --dt_fixed       fixed dt (s), used strictly (no internal sub-stepping)
  --auto_speed     pick ω so that all planned layers&loops fit exactly in t_tot
  --scale {linear,log}, --vmin/--vmax, --fix_scale: for stable colormap
"""

import argparse, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm

from adi3d_cyl_phi_v3 import GridCyl, Material, Params, RobinR, ZBC, adi_step


def adi_step_masked(Tn, grid, mat, prm, robin_outer, zbc, active,
                    robin_inner=None, robin_void=None):
    """Wrapper around :func:`adi_step` that keeps void cells clamped.

    Parameters
    ----------
    Tn : np.ndarray
        Temperature field at the start of the step ``(nr, nphi, nz)``.
    active : np.ndarray
        Boolean mask selecting cells that belong to solid material.
    robin_inner / robin_void : RobinR, optional
        Convection data to use at material/void interfaces.  They are
        currently used to determine the ambient temperature that void
        cells should be clamped to.  When ``None`` they default to the
        outer Robin data.
    """

    robin_inner = robin_inner or robin_outer
    robin_void = robin_void or robin_outer

    T_work = np.array(Tn, copy=True)
    ambient_inner = float(robin_inner.T_inf)
    ambient_void = float(robin_void.T_inf)

    void_mask = ~active
    if np.any(void_mask):
        T_work[void_mask] = ambient_void

    Tnp1 = adi_step(T_work, grid, mat, prm, robin_outer, zbc)

    if np.any(void_mask):
        Tnp1[void_mask] = ambient_void

    # For completeness keep innermost cells (axis) tied to the provided
    # inner Robin reference temperature when they are inactive.
    axis_mask = (~active[0])
    if np.any(axis_mask):
        Tnp1[0, axis_mask] = ambient_inner

    return Tnp1

# ---------------------------- utils / grid ----------------------------------

def build_grid_annular(R_out, wall_thickness, height, z_back, nr, nphi, dz_override=None):
    R_in = max(0.0, R_out - wall_thickness)
    dr = (R_out - R_in)/float(nr)
    dz = dr if (dz_override is None or dz_override <= 0.0) else float(dz_override)
    nz = int(round((z_back + height)/dz))
    dphi = (2.0*math.pi) / max(1, nphi)
    return GridCyl(nr, nphi, nz, dr, dphi, dz, R_out, R_in=R_in), R_in, R_out, dz

# ------------------------------- main ---------------------------------------

def main():
    p = argparse.ArgumentParser(description="WAAM tube: layer-by-layer (rings), fixed-dt masked ADI")
    # geom/grid
    p.add_argument('--R_out', type=float, required=True)
    p.add_argument('--wall_thickness', type=float, required=True)
    p.add_argument('--height', type=float, required=True)
    p.add_argument('--z_back', type=float, required=True)
    p.add_argument('--nr', type=int, default=24)
    p.add_argument('--nphi', type=int, default=36)
    p.add_argument('--dz', type=float, default=None, help='override dz (if None => dz=dr)')
    # material
    p.add_argument('--rho', type=float, default=7800.0)
    p.add_argument('--cp', type=float, default=490.0)
    p.add_argument('--k', type=float, default=54.0)
    # BCs
    p.add_argument('--h_side', type=float, default=300.0)
    p.add_argument('--h_end', type=float, default=150.0)
    p.add_argument('--T_inf', type=float, default=20.0)
    p.add_argument('--Ts', type=float, default=1000.0)
    p.add_argument('--h_void', type=float, default=None, help='Robin h at material/void (default=h_side)')
    # time
    p.add_argument('--t_tot', type=float, default=30.0)
    p.add_argument('--dt_fixed', type=float, default=0.05, help='fixed dt (s).')
    p.add_argument('--nframes', type=int, default=60)
    # deposition kinematics: ring per layer
    p.add_argument('--pitch', type=float, required=True, help='vertical distance per full turn (m); if loops_per_layer==1, pitch==layer_height')
    p.add_argument('--speed', type=float, default=None, help='tangential speed (m/s) along ring')
    p.add_argument('--auto_speed', action='store_true', help='choose speed so total planned work fits in t_tot')
    p.add_argument('--loops_per_layer', type=int, default=1, help='circumferential loops per layer (>=1)')
    p.add_argument('--layer_cells_z', type=int, default=1, help='layer thickness in z-cells (default 1 ⇒ layer_height=dz)')
    p.add_argument('--z0', type=float, default=0.0)
    p.add_argument('--deposit_mode', type=str, default='set', choices=['set'])
    # viz
    p.add_argument('--view', type=str, default='surface', choices=['surface','slice'])
    p.add_argument('--iphi_slice', type=int, default=0)
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--fix_scale', action='store_true')
    p.add_argument('--vmin', type=float, default=None)
    p.add_argument('--vmax', type=float, default=None)
    p.add_argument('--scale', type=str, default='linear', choices=['linear','log'])
    p.add_argument('--fig_w', type=float, default=8.0)
    p.add_argument('--fig_h', type=float, default=4.5)
    p.add_argument('--gif_path', type=str, required=True)
    # verbosity
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--log_stride_cols', type=int, default=0)
    args = p.parse_args()

    # grid/material/BC
    grid, R_in, R_out, dz = build_grid_annular(args.R_out, args.wall_thickness, args.height, args.z_back, args.nr, args.nphi, args.dz)
    mat = Material(args.rho, args.cp, args.k)
    rob_outer = RobinR(args.h_side, args.T_inf)
    rob_inner = RobinR(args.h_side, args.T_inf)
    zbc = ZBC(kind_bot='neumann0', kind_top='robin', h_top=args.h_end, T_inf_top=args.T_inf)

    if not args.quiet:
        print(f"[geom] tube R_in={R_in:0.4f} m, R_out={R_out:0.4f} m, height={args.height:0.4f} m")
        print(f"[grid] nr={grid.nr}, nphi={grid.nphi}, nz={grid.nz}, dr≈{grid.dr:0.4e} m, dz≈{grid.dz:0.4e} m")
        print(f"[bc]   Robin(inner & outer): h={args.h_side} W/m^2/K; top Robin h_end={args.h_end}")

    # layers
    iz_base = int(round(args.z_back / grid.dz))         # z≈0 index
    layer_cells = max(1, int(args.layer_cells_z))
    layer_height = layer_cells * grid.dz
    n_layers = int(math.ceil(args.height / layer_height))
    if not args.quiet:
        print(f"[layers] layer_height≈{layer_height:.4e} m ({layer_cells} cells), planned layers≈{n_layers}")

    # speed / omega
    r_eff = 0.5*(R_in + R_out)
    loops = max(1, int(args.loops_per_layer))
    loop_length = 2.0 * math.pi * r_eff

    if args.auto_speed or (args.speed is None):
        total_loops = n_layers * loops
        omega = (total_loops * 2.0*math.pi) / max(args.t_tot, 1e-15)
        speed = omega * r_eff
        if not args.quiet:
            print(f"[goal] auto_speed: total_loops={total_loops}, omega≈{omega:.4f} rad/s, speed≈{speed:.4f} m/s")
    else:
        speed = float(args.speed)
        omega = speed / max(r_eff, 1e-15)
        if not args.quiet:
            time_per_loop = loop_length / max(speed, 1e-15)
            print(f"[path] speed={speed} m/s ⇒ time_per_loop≈{time_per_loop:.3f} s (r_eff={r_eff:.4f} m)")

    # time
    dt = float(args.dt_fixed)
    prm = Params(dt, 1.0, "be")
    times = np.linspace(0.0, args.t_tot, args.nframes + 1)

    # state & mask
    T = np.full((grid.nr, grid.nphi, grid.nz), args.T_inf, float)
    active = np.zeros((grid.nr, grid.nphi, grid.nz), dtype=bool)
    active[:, :, :iz_base] = True  # substrate present below z=0

    # deposition state
    current_layer = 0
    current_loop_idx = 0         # within layer [0..loops-1]
    current_angle = 0.0          # angle along current loop [0..2π)
    current_iz = iz_base + current_layer * layer_cells
    iz_max = grid.nz - 1

    # --- helpers ---
    def normalize_angle(a):
        a = a % (2.0*math.pi)
        return a

    def mark_arc_on_layer(iz, ang0, ang1):
        """Activate all phi-columns at fixed z-index iz along arc (ang0 -> ang1).
           ang0,ang1 in [0, 2π), and ang1>=ang0 (no wrap)."""
        if iz < 0 or iz > iz_max:
            return 0
        if ang1 <= ang0:
            return 0
        dphi = grid.dphi
        # which indices are crossed? take integer cells whose [i*dphi,(i+1)*dphi) intersect (ang0,ang1]
        i0 = int(math.floor(ang0 / dphi))
        i1 = int(math.floor(ang1 / dphi))
        added = 0
        for i in range(i0+1, i1+1):
            iphi = i % grid.nphi
            if not active[0, iphi, iz]:
                active[:, iphi, iz] = True
                T[:, iphi, iz] = args.Ts
                added += 1
        return added

    # --- viz ---
    fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h), dpi=120)
    if args.view == 'surface':
        data = T[-1, :, :].T
        extent = [0, 360, -args.z_back, args.height]
        ax.set_xlabel('phi, deg'); ax.set_ylabel('z, m')
    else:
        data = T[:, int(args.iphi_slice % grid.nphi), :].T
        extent = None
        ax.set_xlabel('r-index'); ax.set_ylabel('z-index')

    vmin = args.vmin if args.vmin is not None else (args.T_inf if args.fix_scale else None)
    vmax = args.vmax if args.vmax is not None else (args.Ts    if args.fix_scale else None)
    cmap = plt.get_cmap('jet')
    if args.scale == 'log':
        if vmin is None or vmin <= 0:
            vmin = max(1e-6, args.T_inf + 1e-6)
        vmax_eff = vmax if vmax is not None else max(args.Ts, float(np.nanmax(data))+1e-6)
        im = ax.imshow(data, origin='lower', extent=extent, aspect='auto',
                       cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax_eff))
    else:
        im = ax.imshow(data, origin='lower', extent=extent, aspect='auto',
                       cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('t=0.00 s')

    if not args.quiet:
        print(f"[sim] start; frames={len(times)-1}, fixed-dt={dt:g} s")

    deposited_total = 0
    last_reported = 0

    def update(frame_idx):
        nonlocal current_layer, current_loop_idx, current_angle, current_iz, deposited_total, last_reported
        t0 = times[frame_idx]
        t1 = times[frame_idx+1]
        t = t0
        while t < t1 - 1e-12:
            t_next = min(t + dt, t1)
            dtheta = omega * (t_next - t)
            # advance along ring(s) possibly across multiple loops/layers
            angle_left = dtheta
            while angle_left > 0.0 and current_layer < n_layers:
                # how much remains in current loop until 2π?
                rem = (2.0*math.pi) - current_angle
                seg = min(angle_left, rem)
                if seg > 0.0:
                    # deposit on current layer
                    a0 = current_angle
                    a1 = current_angle + seg
                    added = mark_arc_on_layer(current_iz, a0, a1)
                    deposited_total += added
                    current_angle += seg
                    angle_left -= seg
                if current_angle >= (2.0*math.pi - 1e-15):
                    # finished loop
                    current_angle = 0.0
                    current_loop_idx += 1
                    if current_loop_idx >= loops:
                        # finished all loops for this layer -> move to next layer
                        current_loop_idx = 0
                        current_layer += 1
                        current_iz = iz_base + current_layer * layer_cells
                        if current_iz > iz_max:
                            current_layer = n_layers  # stop depositing further
                            break
            # heat step for this dt
            prm.dt = (t_next - t)
            T[:] = adi_step_masked(T, grid, mat, prm, rob_outer, zbc, active,
                                   robin_inner=rob_inner, robin_void=RobinR(args.h_void or args.h_side, args.T_inf))
            t = t_next

        # refresh frame image
        if args.view == 'surface':
            data = T[-1, :, :].T
            im.set_data(data)
        else:
            data = T[:, int(args.iphi_slice % grid.nphi), :].T
            im.set_data(data)
        ax.set_title(f"t={t1:0.2f} s")

        if args.log_stride_cols > 0 and not args.quiet:
            now = deposited_total // args.log_stride_cols
            if now > last_reported:
                last_reported = now
                print(f"[log] deposited columns: {deposited_total}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=args.nframes,
                                  interval=1000.0/args.fps, blit=False, repeat=False)
    ani.save(args.gif_path, writer='pillow', fps=args.fps)
    if not args.quiet:
        print(f"[done] GIF saved to: {args.gif_path}")

if __name__ == "__main__":
    main()
