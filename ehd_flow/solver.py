"""FlowSolver: phiFlow-backed 2D incompressible flow around obstacles.

Single responsibility: the physics. Owns the velocity field, boundary
conditions, and obstacle masks, and advances them in time. Performs no
file I/O and no rendering -- observe the evolving field through the
``on_step`` callback of :meth:`run`.
"""

import time

import numpy as np

from phi import field, math
from phi.flow import StaggeredGrid, CenteredGrid, fluid, advect, diffuse
from phi.geom import Box
from phi.math import extrapolation, vec, spatial, tensor

from .config import SimulationConfig
from .forces import BodyForce
from .geometry import ObstacleSet


class FlowSolver:
    """2D incompressible Navier-Stokes past polygonal obstacles.

    Boundary conditions: uniform inflow on the left edge, zero-gradient
    outflow on the right edge, no-slip walls top/bottom, no-slip on every
    polygon (rigid, stationary).
    """

    def __init__(self, config: SimulationConfig, obstacles: ObstacleSet,
                 body_force: BodyForce):
        self.config = config
        self.obstacles = obstacles
        self.body_force = body_force

        lx, ly, nx, ny = config.lx, config.ly, config.nx, config.ny
        self.domain_box = Box(x=lx, y=ly)

        solid = obstacles.rasterize(nx, ny, lx, ly).astype(np.float32)
        solid_centered = CenteredGrid(
            tensor(solid, spatial(x=nx, y=ny)),
            extrapolation.ZERO, bounds=self.domain_box)
        self.active = CenteredGrid(
            tensor(1.0 - solid, spatial(x=nx, y=ny)),
            extrapolation.NONE, bounds=self.domain_box)  # no pressure solve inside solids

        # left: uniform inflow | right: zero-gradient outflow | top/bottom: walls
        bc = extrapolation.combine_sides(
            x=(vec(x=config.vx0, y=config.vy0), extrapolation.ZERO_GRADIENT),
            y=(extrapolation.ZERO, extrapolation.ZERO),
        )
        self.velocity = StaggeredGrid(
            (config.vx0, config.vy0), bc, x=nx, y=ny, bounds=self.domain_box)

        # no-slip mask on the staggered grid (crisp threshold, or soft)
        solid_staggered = field.resample(solid_centered, self.velocity)
        solid_vals = solid_staggered.values
        if config.soft_mask:
            mask_vals = 1 - solid_vals
        else:
            mask_vals = math.where(solid_vals > 0.5, 0.0, 1.0)
        self.fluid_mask = self.velocity.with_values(mask_vals)

    def step(self, step: int) -> None:
        """Advance the flow by exactly one timestep."""
        dt = self.config.dt
        vel = advect.semi_lagrangian(self.velocity, self.velocity, dt)
        vel = diffuse.explicit(vel, self.config.viscosity, dt)
        vel = self.body_force.apply(vel, step, dt)
        vel, _pressure = fluid.make_incompressible(vel, (), active=self.active)
        self.velocity = field.safe_mul(self.fluid_mask, vel)  # no-slip inside solids

    def run(self, steps: int, on_step=None) -> None:
        """Advance ``steps`` timesteps.

        ``on_step``, if given, is called as ``on_step(step, self)`` after
        each step -- e.g. to record frames for a movie.
        """
        t0 = time.time()
        report_every = max(1, steps // 10)
        for step in range(1, steps + 1):
            self.step(step)
            if on_step is not None:
                on_step(step, self)
            if step % report_every == 0 or step == steps:
                el = time.time() - t0
                print(f"  step {step}/{steps}  ({el:.0f}s elapsed)", flush=True)
        print(f"done: simulated {steps * self.config.dt:.3f} s of flow "
              f"in {time.time() - t0:.0f}s wall time")

    def speed(self) -> np.ndarray:
        """Speed magnitude at cell centers as a ``(ny, nx)`` array."""
        v = self.velocity.at_centers().numpy()
        return np.sqrt((v ** 2).sum(axis=-1)).T
