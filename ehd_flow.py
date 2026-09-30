#!/usr/bin/env python3
"""
ehd_flow.py -- lightweight 2D incompressible flow past polygonal obstacles.

Phase 1 of the EAD/ionocraft drag workflow: solves the neutral-air flow
(advection + viscosity + pressure projection) around rigid bodies described
as polygons in a JSON file. No electrostatic/ion forces yet -- see the
``apply_body_force`` hook below, which is where the phase-2 ion-drag force
field (F = rho_c * E) will be added.

Backend: phiFlow (pip install phiflow). Tested with phiflow 3.4.

Usage:
    python ehd_flow.py --obstacles obstacles.json --steps 500 \\
        --grid 256,128 --velocity 5,0 --image final.png --movie flow.mov

JSON format (coordinates in the same physical units as the domain, meters):
    {
      "domain": {"x": 4.0, "y": 2.0},          # optional if --domain given
      "polygons": [
        {"name": "collector", "points": [[1.9, 0.9], [2.1, 0.9], ...]},
        [[0.5, 0.4], [0.6, 0.4], [0.6, 0.6], [0.5, 0.6]]   # bare list also ok
      ]
    }

Boundary conditions: uniform inflow on the left edge (== --velocity),
zero-gradient outflow on the right edge, no-slip walls top/bottom,
no-slip on every polygon (rigid, stationary).
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
from matplotlib import animation

from phi.flow import StaggeredGrid, CenteredGrid, fluid, advect, diffuse
from phi import field, math
from phi.geom import Box
from phi.math import extrapolation, vec, spatial, tensor

AIR_VISCOSITY = 1.5e-5  # kinematic viscosity of air, m^2/s


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def points_in_polygon(pts, poly):
    """Vectorized ray-casting point-in-polygon test.

    pts:  (..., 2) array of query points
    poly: (N, 2) array of polygon vertices (any winding)
    returns: (...,) boolean array
    """
    x = pts[..., 0]
    y = pts[..., 1]
    inside = np.zeros(x.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # edge straddles the horizontal ray through the point?
        cond = ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1
        )
        inside ^= cond
    return inside


def load_case(path):
    """Load domain size and polygons from the JSON case file."""
    with open(path) as f:
        case = json.load(f)
    domain = case.get("domain")
    raw_polys = case.get("polygons", [])
    polygons = []
    for entry in raw_polys:
        if isinstance(entry, dict):
            name = entry.get("name", f"poly{len(polygons)}")
            pts = np.asarray(entry["points"], dtype=float)
        else:
            name = f"poly{len(polygons)}"
            pts = np.asarray(entry, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError(f"polygon '{name}' needs >= 3 (x, y) points")
        polygons.append({"name": name, "points": pts})
    return domain, polygons


# --------------------------------------------------------------------------
# phase-2 hook
# --------------------------------------------------------------------------

def apply_body_force(velocity, step, dt):
    """Add volumetric body forces to the velocity field.

    Phase 1: identity (pure aerodynamics, no ion forcing).
    Phase 2: replace with the EHD force, e.g. a prescribed ion-drag field
    or F = rho_c * E from a coupled Poisson/drift-diffusion solve, added as
        velocity = velocity + dt * force_field
    where force_field is a StaggeredGrid of accelerations (m/s^2).
    """
    return velocity


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_speed(ax, speed, polygons, lx, ly, title, vmin=0.0, vmax=None):
    """Draw speed magnitude with obstacle outlines on the given axes."""
    ax.clear()
    im = ax.imshow(
        speed, origin="lower", cmap="turbo",
        extent=[0, lx, 0, ly], aspect="auto",
        vmin=vmin, vmax=vmax,
    )
    for poly in polygons:
        pts = np.vstack([poly["points"], poly["points"][:1]])
        ax.plot(pts[:, 0], pts[:, 1], "k-", lw=1.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    return im


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="2D incompressible flow past polygonal obstacles (phiFlow)."
    )
    p.add_argument("--obstacles", required=True,
                   help="path to JSON file with domain + polygon obstacles")
    p.add_argument("--steps", type=int, default=200,
                   help="number of timesteps to run before stopping")
    p.add_argument("--grid", default="256,128",
                   help="grid resolution as 'nx,ny' (default 256,128)")
    p.add_argument("--velocity", default="5,0",
                   help="initial + inflow velocity as 'vx,vy' in m/s")
    p.add_argument("--domain", default=None,
                   help="domain size as 'lx,ly' in m (overrides JSON)")
    p.add_argument("--dt", type=float, default=None,
                   help="timestep in s (default: CFL 0.5)")
    p.add_argument("--viscosity", type=float, default=AIR_VISCOSITY,
                   help="kinematic viscosity in m^2/s (default: air)")
    p.add_argument("--image", default=None,
                   help="optional output PNG path for the final frame")
    p.add_argument("--movie", default=None,
                   help="optional output .mov path rendering the flow")
    p.add_argument("--movie-stride", type=int, default=1,
                   help="record every Nth step for the movie (default 1)")
    p.add_argument("--movie-fps", type=int, default=30,
                   help="movie framerate (default 30)")
    p.add_argument("--soft-mask", action="store_true",
                   help="soft (fractional) obstacle boundary instead of crisp no-slip")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ---- inputs ---------------------------------------------------------
    json_domain, polygons = load_case(args.obstacles)
    if args.domain:
        lx, ly = (float(v) for v in args.domain.split(","))
    elif json_domain:
        lx, ly = float(json_domain["x"]), float(json_domain["y"])
    else:
        lx, ly = 4.0, 2.0
    nx, ny = (int(v) for v in args.grid.split(","))
    vx0, vy0 = (float(v) for v in args.velocity.split(","))
    if args.steps <= 0 or nx <= 0 or ny <= 0:
        raise ValueError("--steps and --grid values must be positive")

    dx, dy = lx / nx, ly / ny
    vmag = float(np.hypot(vx0, vy0))
    # floor keeps dt finite if the user passes zero velocity; diffusion
    # stability (dt <= ~dx^2/nu) is easily satisfied at these resolutions
    dt = args.dt or 0.5 * min(dx, dy) / max(vmag, 1.0)

    print(f"domain {lx} x {ly} m | grid {nx} x {ny} | "
          f"v=({vx0}, {vy0}) m/s | dt={dt:.3e} s | steps={args.steps}")
    print(f"obstacles: {', '.join(p['name'] for p in polygons) or '(none)'}")

    # ---- obstacle masks ---------------------------------------------------
    # cell-center rasterization of the polygons (x-first ordering for phiFlow)
    yy, xx = np.mgrid[0:ny, 0:nx]
    pts = np.stack([((xx + 0.5) / nx * lx).T,
                    ((yy + 0.5) / ny * ly).T], axis=-1)
    solid = np.zeros((nx, ny), dtype=bool)
    for poly in polygons:
        solid |= points_in_polygon(pts, poly["points"])
    solid = solid.astype(np.float32)

    domain_box = Box(x=lx, y=ly)
    solid_centered = CenteredGrid(
        tensor(solid, spatial(x=nx, y=ny)), extrapolation.ZERO, bounds=domain_box)
    active = CenteredGrid(
        tensor(1.0 - solid, spatial(x=nx, y=ny)),
        extrapolation.NONE, bounds=domain_box)  # no pressure solve inside solids

    # ---- flow field --------------------------------------------------------
    # left: uniform inflow | right: zero-gradient outflow | top/bottom: walls
    bc = extrapolation.combine_sides(
        x=(vec(x=vx0, y=vy0), extrapolation.ZERO_GRADIENT),
        y=(extrapolation.ZERO, extrapolation.ZERO),
    )
    vel = StaggeredGrid((vx0, vy0), bc, x=nx, y=ny, bounds=domain_box)

    # no-slip mask on the staggered grid (crisp threshold, or soft)
    solid_staggered = field.resample(solid_centered, vel)
    solid_vals = solid_staggered.values
    if args.soft_mask:
        mask_vals = 1 - solid_vals
    else:
        mask_vals = math.where(solid_vals > 0.5, 0.0, 1.0)
    fluid_mask = vel.with_values(mask_vals)

    # ---- time loop ----------------------------------------------------------
    frames = []
    record = args.movie is not None
    stride = max(1, args.movie_stride)
    vmax_hint = 1.6 * vmag

    t0 = time.time()
    for step in range(1, args.steps + 1):
        vel = advect.semi_lagrangian(vel, vel, dt)
        vel = diffuse.explicit(vel, args.viscosity, dt)
        vel = apply_body_force(vel, step, dt)          # phase-2 hook (no-op now)
        vel, _pressure = fluid.make_incompressible(vel, (), active=active)
        vel = field.safe_mul(fluid_mask, vel)          # no-slip inside solids

        if record and step % stride == 0:
            frames.append(np.sqrt((vel.at_centers().numpy() ** 2).sum(axis=-1)).T)

        if step % max(1, args.steps // 10) == 0 or step == args.steps:
            el = time.time() - t0
            print(f"  step {step}/{args.steps}  ({el:.0f}s elapsed)", flush=True)

    sim_time = args.steps * dt
    print(f"done: simulated {sim_time:.3f} s of flow in {time.time() - t0:.0f}s wall time")

    # ---- outputs --------------------------------------------------------------
    speed = np.sqrt((vel.at_centers().numpy() ** 2).sum(axis=-1)).T  # (ny, nx)

    if args.image:
        fig, ax = plt.subplots(figsize=(12, 6))
        im = render_speed(ax, speed, polygons, lx, ly,
                          f"speed after {args.steps} steps (t={sim_time:.3f} s)",
                          vmax=vmax_hint)
        fig.colorbar(im, ax=ax, label="m/s")
        fig.tight_layout()
        fig.savefig(args.image, dpi=100)
        plt.close(fig)
        print(f"wrote image: {args.image}")

    if args.movie:
        if shutil.which("ffmpeg") is None:
            print("warning: ffmpeg not found, skipping movie", file=sys.stderr)
        elif not frames:
            print("warning: no frames recorded, skipping movie", file=sys.stderr)
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            im = render_speed(ax, frames[0], polygons, lx, ly,
                              f"speed, step {stride} (t={stride * dt:.3f} s)",
                              vmax=vmax_hint)
            fig.colorbar(im, ax=ax, label="m/s")

            def update(i):
                s = i * stride
                render_speed(ax, frames[i], polygons, lx, ly,
                             f"speed, step {s} (t={s * dt:.3f} s)",
                             vmax=vmax_hint)

            ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                          interval=1000 // args.movie_fps)
            writer = animation.FFMpegWriter(fps=args.movie_fps, codec="libx264")
            ani.save(args.movie, writer=writer)
            plt.close(fig)
            print(f"wrote movie: {args.movie} "
                  f"({len(frames)} frames, {os.path.getsize(args.movie) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
