"""ehd_flow: 2D incompressible flow past polygonal obstacles (phiFlow backend).

Phase 1 of the EAD/ionocraft drag workflow: solves the neutral-air flow
(advection + viscosity + pressure projection) around rigid bodies described
as polygons in a JSON file. Phase-2 ion forcing plugs in via
:mod:`ehd_flow.forces` without touching the solver.

Exports are lazy (PEP 562): importing ``ehd_flow.config`` does not pull in
phiFlow; only ``ehd_flow.solver`` needs it, and only ``ehd_flow.polygons``
/ ``ehd_flow.coulomb`` need OpenCV.
"""

__all__ = [
    "AIR_VISCOSITY",
    "SimulationConfig",
    "BodyForce",
    "NoBodyForce",
    "ObstacleSet",
    "Polygon",
    "FrameArtist",
    "MovieExporter",
    "PngExporter",
    "FlowSolver",
    "PolygonExtractor",
    "CoulombField",
]

_LAZY_IMPORTS = {
    "AIR_VISCOSITY": ".config",
    "SimulationConfig": ".config",
    "BodyForce": ".forces",
    "NoBodyForce": ".forces",
    "ObstacleSet": ".geometry",
    "Polygon": ".geometry",
    "FrameArtist": ".rendering",
    "MovieExporter": ".rendering",
    "PngExporter": ".rendering",
    "FlowSolver": ".solver",
    "PolygonExtractor": ".polygons",
    "CoulombField": ".coulomb",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
