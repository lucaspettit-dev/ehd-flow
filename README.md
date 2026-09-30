# EHD flow sandbox (phase 1: aerodynamics only)

`ehd_flow.py` runs a lightweight 2D incompressible-flow simulation (phiFlow)
past polygonal obstacles -- the drag half of the ionocraft problem. No ion
forces yet; the `apply_body_force()` hook in the script is where the phase-2
ion-drag force field goes.

## Install

    pip install phiflow matplotlib

(ffmpeg is needed only for `--movie`; check with `which ffmpeg`.)

## Run

    python ehd_flow.py --obstacles sample_wire_collector.json \
        --steps 200 --grid 256,128 --velocity 5,0 \
        --image final.png --movie flow.mov

All flags:

| flag | meaning |
|---|---|
| `--obstacles` | JSON file with `domain` + `polygons` (required) |
| `--steps` | timesteps to run before stopping (default 200) |
| `--grid` | resolution `nx,ny` (default 256,128) |
| `--velocity` | initial + left-inflow velocity `vx,vy` m/s (default 5,0) |
| `--domain` | `lx,ly` in meters, overrides the JSON |
| `--dt` | timestep in s (default: CFL 0.5) |
| `--viscosity` | kinematic viscosity m^2/s (default: air, 1.5e-5) |
| `--image` | optional PNG path for the final speed field |
| `--movie` | optional .mov path rendering the flow |
| `--movie-stride` / `--movie-fps` | record every Nth step / framerate |
| `--soft-mask` | soft obstacle boundary instead of crisp no-slip |

Boundary conditions: uniform inflow left, zero-gradient outflow right,
no-slip walls top/bottom, no-slip on all polygons.

## JSON format

Coordinates are in domain units (meters). Each polygon is either
`{"name": ..., "points": [[x, y], ...]}` or a bare list of points.

```json
{
  "domain": {"x": 2.0, "y": 1.0},
  "polygons": [
    {"name": "collector", "points": [[1.9, 0.9], [2.1, 0.9], [2.1, 1.1], [1.9, 1.1]]}
  ]
}
```

## Cost

The pressure solve dominates: ~0.7 s/step at 128x64, ~2-4 s/step at
256x128 on CPU. Start coarse, refine once the setup looks right.
