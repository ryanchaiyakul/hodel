from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import jax
from jax.tree_util import register_dataclass


@register_dataclass
@dataclass
class EnergyModel(ABC):
    """Stationary energy."""

    @abstractmethod
    def get_energy(
        self,
        xf: jax.Array,
        xb: Any = None,
        Theta: Any = None,
        aux: Any = None,
    ) -> jax.Array: ...

    def get_grad_energy(
        self,
        xf: jax.Array,
        xb: Any = None,
        Theta: Any = None,
        aux: Any = None,
    ) -> jax.Array:
        return jax.grad(self.get_energy, 0)(xf, xb, Theta, aux)

    def get_hess_energy(
        self,
        xf: jax.Array,
        xb: Any = None,
        Theta: Any = None,
        aux: Any = None,
    ) -> jax.Array:
        return jax.hessian(self.get_energy, 0)(xf, xb, Theta, aux)

    def get_mixed_hess_energy(
        self,
        xf: jax.Array,
        xb: Any = None,
        Theta: Any = None,
        aux: Any = None,
    ) -> jax.Array:
        return jax.jacobian(self.get_grad_energy, 1)(xf, xb, Theta, aux)


@register_dataclass
@dataclass
class ExternalForce(ABC):
    """Lambda depedent external force."""

    @abstractmethod
    def get_grad_energy(self, lambda_: jax.Array, aux: Any = None) -> jax.Array: ...
