#!/usr/bin/env python3
"""Command-line interface for ehd_flow.

Single responsibility: wiring. Parses flags, builds the config, obstacles,
solver, and exporters, then runs. Contains no physics, no plotting, and no
file-format details.

Usage:
    python -m ehd_flow --obstacles obstacles.json --steps 500 \\
        --grid 256,128 --velocity 5,0 --image final.png --movie flow.mov
    (or: python ehd_flow.py --obstacles ...  -- the repo-root shim)

JSON format (coordinates in the same physical units as the domain, meters):
    {
      "domain": {"x": 4.0, "y": 2.0},          # optional if --domain given
      "polygons": [
        {"name": "collector", "points": [[1.9, 0.9], [2.1, 0.9], ...]},
        [[0.5, 0.4], [0.6, 0.4], [0.6, 0.6], [0.5, 0.6]]   # bare list also ok
      ]
    }
"""

import argparse

from .config import AIR_VISCOSITY, SimulationConfig
from .forces import NoBodyForce
from .geometry import ObstacleSet, read_domain
from .rendering import FrameArtist, MovieExporter, PngExporter
# NOTE: .solver is imported inside main() so that --help and config-only
# use don't require phiFlow to be installed.


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
    # Deferred so --help (handled above) never requires phiFlow installed.
    from .solver import FlowSolver

    obstacles = ObstacleSet.from_json(args.obstacles)
    config = SimulationConfig.from_args(args, json_domain=read_domain(args.obstacles))

    print(f"domain {config.lx} x {config.ly} m | grid {config.nx} x {config.ny} | "
          f"v=({config.vx0}, {config.vy0}) m/s | dt={config.dt:.3e} s | steps={config.steps}")
    print(f"obstacles: {', '.join(obstacles.names) or '(none)'}")

    solver = FlowSolver(config, obstacles, NoBodyForce())

    frames = []
    stride = max(1, config.movie_stride)
    on_step = None
    if config.movie:
        def on_step(step, solver):
            if step % stride == 0:
                frames.append(solver.speed())

    solver.run(config.steps, on_step=on_step)

    artist = FrameArtist(obstacles.polygons, config.lx, config.ly)
    vmax_hint = 1.6 * config.vmag
    sim_time = config.steps * config.dt

    if config.image:
        PngExporter(artist).save(
            config.image, solver.speed(),
            f"speed after {config.steps} steps (t={sim_time:.3f} s)",
            vmax=vmax_hint)

    if config.movie:
        MovieExporter(artist).save(
            config.movie, frames, fps=config.movie_fps,
            dt=config.dt, stride=stride, vmax=vmax_hint)


if __name__ == "__main__":
    main()
