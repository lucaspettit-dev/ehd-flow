# EHD flow sandbox (phase 1: aerodynamics only)

`ehd_flow.py` runs a lightweight 2D incompressible-flow simulation (phiFlow)
past polygonal obstacles -- the drag half of the ionocraft problem. No ion
forces yet; subclass `BodyForce` in `src/ehd_flow/forces.py` for the phase-2
ion-drag force field -- the solver needs no changes.

Layout (telempy-style `src/` tree):

    src/ehd_flow/
        config.py, geometry.py, forces.py     # sim config, polygons, body forces
        solver.py, rendering.py, cli.py       # phiFlow solver, output, flow CLI
        polygons.py, polygons_cli.py          # PolygonExtractor + image_to_polygons CLI
        coulomb.py, coulomb_cli.py            # CoulombField + coulomb_field CLI

The root scripts (`ehd_flow.py`, `image_to_polygons.py`, `coulomb_field.py`)
are thin CLI wrappers that add `src/` to the import path, so they run from
any directory with no setup. `image_to_polygons.py` and `coulomb_field.py`
share the same `PolygonExtractor` helper class -- the old
`from image_to_polygons import ...` sibling import is gone.

## Install

    pip install -r requirements.txt

or, to also get importable `ehd_flow` + console scripts from anywhere:

    pip install -e .

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

## Companion tools

`image_to_polygons.py` converts an image into solver-ready polygon JSON
(grayscale, binarize at 127.5, external contours only, Douglas-Peucker
simplify), with a green-bordered preview PNG:

    python image_to_polygons.py input.png obstacles.json --no-window

`coulomb_field.py` renders Coulomb field lines for image-defined electrodes
(red channel -> positive, blue channel -> negative, arrows show +ion drift):

    python coulomb_field.py electrodes.png --charge-ratio 0.5 --no-window

Both are wrappers over `src/ehd_flow/` (`PolygonExtractor`, `CoulombField`).

## Cost

The pressure solve dominates: ~0.7 s/step at 128x64, ~2-4 s/step at
256x128 on CPU. Start coarse, refine once the setup looks right.
