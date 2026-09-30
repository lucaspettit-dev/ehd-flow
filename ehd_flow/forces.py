"""Volumetric body forces.

Single responsibility: how external forcing modifies the velocity field.
Strategy pattern -- the solver calls ``BodyForce.apply`` each step and does
not care which force it is. Phase 2 adds a subclass here (e.g. an ion-drag
field or F = rho_c * E from a coupled Poisson/drift-diffusion solve) with
zero changes to the solver.
"""

from abc import ABC, abstractmethod


class BodyForce(ABC):
    """Applies a volumetric force to the velocity field over one timestep."""

    @abstractmethod
    def apply(self, velocity, step: int, dt: float):
        """Return the velocity field after forcing.

        ``velocity`` is a phiFlow StaggeredGrid; the returned value must be
        one too (accelerations in m/s^2 enter as ``velocity + dt * force``).
        """


class NoBodyForce(BodyForce):
    """Phase 1: pure aerodynamics, no ion forcing (identity)."""

    def apply(self, velocity, step: int, dt: float):
        return velocity
