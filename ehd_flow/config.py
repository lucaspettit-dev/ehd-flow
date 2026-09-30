"""SimulationConfig: validated run parameters for one flow case.

Single responsibility: hold and validate the numbers that define a run
(domain, grid, inflow, timestep, outputs). Knows nothing about phiFlow,
files, or rendering.
"""

from dataclasses import dataclass

import numpy as np

AIR_VISCOSITY = 1.5e-5  # kinematic viscosity of air, m^2/s


@dataclass
class SimulationConfig:
    lx: float = 4.0
    ly: float = 2.0
    nx: int = 256
    ny: int = 128
    vx0: float = 5.0
    vy0: float = 0.0
    dt: float = None
    viscosity: float = AIR_VISCOSITY
    steps: int = 200
    image: str = None
    movie: str = None
    movie_stride: int = 1
    movie_fps: int = 30
    soft_mask: bool = False

    def __post_init__(self):
        if self.steps <= 0 or self.nx <= 0 or self.ny <= 0:
            raise ValueError("--steps and --grid values must be positive")
        if self.dt is None:
            # floor keeps dt finite if the user passes zero velocity; diffusion
            # stability (dt <= ~dx^2/nu) is easily satisfied at these resolutions
            self.dt = 0.5 * min(self.dx, self.dy) / max(self.vmag, 1.0)

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def vmag(self) -> float:
        return float(np.hypot(self.vx0, self.vy0))

    @classmethod
    def from_args(cls, args, json_domain=None) -> "SimulationConfig":
        """Build from argparse args; --domain overrides the JSON domain."""
        if args.domain:
            lx, ly = (float(v) for v in args.domain.split(","))
        elif json_domain:
            lx, ly = float(json_domain["x"]), float(json_domain["y"])
        else:
            lx, ly = 4.0, 2.0
        nx, ny = (int(v) for v in args.grid.split(","))
        vx0, vy0 = (float(v) for v in args.velocity.split(","))
        return cls(
            lx=lx, ly=ly, nx=nx, ny=ny, vx0=vx0, vy0=vy0,
            dt=args.dt, viscosity=args.viscosity, steps=args.steps,
            image=args.image, movie=args.movie,
            movie_stride=args.movie_stride, movie_fps=args.movie_fps,
            soft_mask=args.soft_mask,
        )
