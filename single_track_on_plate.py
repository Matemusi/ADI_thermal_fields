
# -*- coding: utf-8 -*-
"""
single_track_on_plate.py
Simulate a single-track deposition on a rectangular baseplate with a 3D ADI solver.
- Units: CLI uses millimeters and Celsius; internal solver uses SI (meters, seconds, Kelvin-like deltas).
- Boundary conditions: Robin on all exterior faces toward ambient T_inf (default 20°C).
- Deposition: activate one y-column of a 3x3 voxel track every (dx_mm / v_mm_s) seconds.
- Output: a GIF heatmap of the side cross-section (x=0 plane) over time.

Requires: adi3d_numba_coeff.py in PYTHONPATH or the same directory.
"""
import argparse, math, os, sys, time
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the ADI core
import sys; sys.path.append('/mnt/data/adi_project')
import adi3d_numba_coeff as adi


def build_packs(grid, mat, T_inf, h_all):
    """(Re)build coefficient packs for current mask with uniform Robin on all faces."""
    robin = {'x-': h_all, 'x+': h_all, 'y-': h_all, 'y+': h_all, 'z-': h_all, 'z+': h_all}
    return adi.precompute_coeff_packs_unified(grid, mat, dir_mask=None, dir_value=None,
                                              neumann=None, robin_h=robin, robin_Tinf=T_inf)


def make_figure_cross_section(T, mask, dx_mm, T_inf, vmin=None, vmax=None):
    """
    Prepare a Matplotlib figure of the side cross-section (x=0 plane): array shape (ny, nz).
    We blank-out values where mask is False to emphasize the shape in the heat map.
    """
    # T_slice indexed as (y, z) to put y on horizontal axis, z on vertical
    T_slice = T[0, :, :].T  # now shape (nz, ny) -> transpose later in imshow via extent
    M_slice = mask[0, :, :].T

    # Build masked array for display
    data = np.where(M_slice, T_slice, np.nan)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    # extent: x from 0..(ny*dx_mm), y from 0..(nz*dx_mm); note imshow expects extent [x0,x1,y0,y1]
    extent = [0, data.shape[1]*dx_mm, 0, data.shape[0]*dx_mm]
    im = ax.imshow(data, origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature (°C)")

    ax.set_xlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Side cross-section at x=0")

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Single-track on baseplate using 3D ADI with Robin BCs")
    # Geometry in mm
    ap.add_argument("--plate_x_mm", type=float, default=8.0, help="Plate width along x [mm]")
    ap.add_argument("--plate_y_mm", type=float, default=50.0, help="Plate length along y [mm]")
    ap.add_argument("--plate_z_mm", type=float, default=8.0, help="Plate thickness along z [mm]")
    ap.add_argument("--dx_mm", type=float, default=1.0, help="Voxel size [mm] (1 or 2 suggested)")

    # Track specs in voxels (3x3 default) and position (near x=0 edge so it's visible in the cross-section)
    ap.add_argument("--track_w_vox", type=int, default=3, help="Track width in voxels along x")
    ap.add_argument("--track_h_vox", type=int, default=3, help="Track height in voxels along z")
    ap.add_argument("--track_y_len_mm", type=float, default=50.0, help="Track length along y [mm] (<= plate_y_mm)")
    ap.add_argument("--track_x0_vox", type=int, default=0, help="Track start index along x (0 pins it to the side)")

    # Process: deposition & time
    ap.add_argument("--scan_speed_mm_s", type=float, default=10.0, help="Scan speed [mm/s]")
    ap.add_argument("--theta", type=float, default=0.5, help="ADI theta (0.5 Crank–Nicolson)")
    ap.add_argument("--dt_s", type=float, default=0.02, help="Time step [s]")
    ap.add_argument("--frames_every", type=int, default=1, help="Save a frame every N deposit-steps")

    # Thermal material properties (typical steel-ish defaults)
    ap.add_argument("--rho", type=float, default=7800.0, help="Density [kg/m^3]")
    ap.add_argument("--cp", type=float, default=500.0, help="Heat capacity [J/(kg*K)]")
    ap.add_argument("--k", type=float, default=25.0, help="Thermal conductivity [W/(m*K)]")

    # Boundary & initial conditions
    ap.add_argument("--T_inf", type=float, default=20.0, help="Ambient temperature for Robin [°C]")
    ap.add_argument("--h_conv", type=float, default=10.0, help="Convective h for Robin [W/(m^2*K)] (all faces)")
    ap.add_argument("--T_init", type=float, default=20.0, help="Initial temperature field [°C]")
    ap.add_argument("--T_track_init", type=float, default=1400.0, help="Initial temperature assigned to just-activated track voxels [°C]")

    # Output
    ap.add_argument("--outdir", type=str, default="out_single_track", help="Output directory")
    ap.add_argument("--gif", type=str, default="single_track_side.gif", help="Heatmap GIF filename")

    args = ap.parse_args()

    # Derived sizes (convert to grid counts)
    dx_mm = args.dx_mm
    dx_m = dx_mm * 1e-3

    nx = int(round(args.plate_x_mm / dx_mm))
    ny = int(round(args.plate_y_mm / dx_mm))
    nz_plate = int(round(args.plate_z_mm / dx_mm))
    track_len_vox = int(round(min(args.track_y_len_mm, args.plate_y_mm) / dx_mm))
    nz_total = nz_plate + args.track_h_vox

    # Basic checks
    if args.track_x0_vox + args.track_w_vox > nx:
        raise SystemExit(f"Track exceeds x-dimension: nx={nx}, requested {args.track_x0_vox}+{args.track_w_vox}")

    # Allocate mask and temperature
    mask = np.zeros((nx, ny, nz_total), dtype=bool)
    mask[:, :, :nz_plate] = True  # baseplate solid
    T = np.full((nx, ny, nz_total), args.T_init, dtype=float)

    # Build grid/material/params
    grid = adi.Grid3D(nx, ny, nz_total, dx_m, mask)
    mat = adi.Material(args.rho, args.cp, args.k)
    params = adi.Params(dt=args.dt_s, theta=args.theta)
    packs = build_packs(grid, mat, args.T_inf, args.h_conv)

    # Time stepping controls
    alpha = args.k / (args.rho * args.cp)
    dt_cap = (dx_m*dx_m) / max(alpha, 1e-12)
    if args.dt_s > dt_cap:
        print(f"[warn] dt={args.dt_s:g} > dx^2/alpha={dt_cap:g}; consider reducing dt for stability/accuracy.")

    t = 0.0
    frames = []
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Helper to save a frame
    # vmin/vmax stabilized over time: use min/max of (T_init, T_track_init)
    vmin = min(args.T_init, args.T_track_init)
    vmax = max(args.T_init, args.T_track_init)

    def save_frame(step_idx):
        fig = make_figure_cross_section(T, mask, dx_mm, args.T_inf, vmin=vmin, vmax=vmax)
        png_path = outdir / f"frame_{step_idx:04d}.png"
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        frames.append(imageio.imread(png_path))

    # Initial frame
    save_frame(0)

    # Deposition loop: y from 0..track_len_vox-1
    x0 = args.track_x0_vox
    x1 = x0 + args.track_w_vox
    z0 = nz_plate
    z1 = nz_plate + args.track_h_vox

    # Time between two deposited columns
    t_step = (dx_mm / max(args.scan_speed_mm_s, 1e-9))

    for yi in range(track_len_vox):
        # Activate the next column of the track
        mask[x0:x1, yi:yi+1, z0:z1] = True
        grid.mask = mask  # update in place

        # Rebuild packs to update boundary conditions on new geometry
        packs = build_packs(grid, mat, args.T_inf, args.h_conv)

        # Heat freshly activated voxels to T_track_init (simple approximation)
        T[x0:x1, yi:yi+1, z0:z1] = args.T_track_init

        # Integrate for t_step by sub-stepping with dt
        n_sub = max(1, int(math.ceil(t_step / args.dt_s)))
        dt_eff = t_step / n_sub
        # Temporarily override dt in params for exact period coverage
        dt_orig = params.dt
        params.dt = dt_eff
        for _ in range(n_sub):
            T[...] = adi.adi_step_numba_coeff(T, grid, mat, params, packs, Tinf=args.T_inf)
        params.dt = dt_orig
        t += t_step

        # Save frames according to interval
        if (yi + 1) % args.frames_every == 0:
            save_frame(yi + 1)

    # Write GIF
    gif_path = outdir / args.gif
    imageio.mimsave(gif_path, frames, duration=0.2)
    print(f"[gif] Saved {gif_path} ({len(frames)} frames)")

if __name__ == "__main__":
    main()
