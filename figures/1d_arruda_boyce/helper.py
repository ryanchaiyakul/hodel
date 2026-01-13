from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import csv

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
import jaxtyping
import flax.linen as nn

import hodel


@register_dataclass
@dataclass
class TripletAux:
    """Parametrized external force."""

    f: jax.Array
    c: jax.Array


@register_dataclass
@dataclass
class Triplet:
    """3 node spring with [K_1, K_2]"""

    l_k: jax.Array

    @classmethod
    def init(cls, xf0: jax.Array, xb0: jax.Array, **kwargs):
        x0 = xb0[0]
        x1, x2 = xf0
        return cls(jnp.array([x1 - x0, x2 - x1]), **kwargs)

    @staticmethod
    def get_strain(x0, x1, x2, l_k) -> jax.Array:
        return jnp.array([x1 - x0, x2 - x1]) / l_k

    def get_K(self, del_strain: jax.Array, Theta: jaxtyping.PyTree) -> jax.Array:
        return jnp.diag(jnp.abs(Theta))

    def get_energy(
        self,
        xf: jax.Array,
        xb: jax.Array,
        Theta: jax.Array,
        aux: TripletAux,
        carry: None,
    ) -> jax.Array:
        # xb = [x0], xf = [x1, x2]
        x0 = xb[0]
        x1, x2 = xf
        del_strain = self.get_strain(x0, x1, x2, self.l_k) - 1.0
        return self.core_energy_fn(del_strain, Theta)

    def core_energy_fn(self, del_strain, Theta) -> jax.Array:
        return 0.5 * del_strain @ self.get_K(del_strain, Theta) @ del_strain


@register_dataclass
@dataclass
class NNTriplet(Triplet):
    """3 node spring where Theta is parameters for a neural network."""

    model: nn.Module

    def get_K(self, del_strain: jax.Array, Theta: nn.FrozenDict) -> jax.Array:
        a = self.model.apply(Theta, del_strain[0:1])
        b = self.model.apply(Theta, del_strain[1:2])
        return jnp.array([[a, 0.0], [0.0, b]])  # type: ignore


class HoDELNN(nn.Module):
    hidden_size: int
    warm_Theta: jax.Array

    @nn.compact
    def __call__(self, del_strain: jax.Array) -> jax.Array:
        hidden_init = nn.initializers.variance_scaling(
            scale=0.01, mode="fan_in", distribution="truncated_normal"
        )
        final_init = nn.initializers.truncated_normal(stddev=1e-4)
        x = nn.softplus(nn.Dense(self.hidden_size, kernel_init=hidden_init)(del_strain))
        x = nn.softplus(nn.Dense(self.hidden_size, kernel_init=hidden_init)(x))
        x = nn.Dense(1, kernel_init=final_init)(x)[0]
        return self.warm_Theta * (1 + x) ** 2


class BaselineNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, del_strain: jax.Array) -> jax.Array:
        hidden_init = nn.initializers.variance_scaling(
            scale=0.01, mode="fan_in", distribution="truncated_normal"
        )
        final_init = nn.initializers.truncated_normal(stddev=1e-4)
        x = nn.softplus(nn.Dense(self.hidden_size, kernel_init=hidden_init)(del_strain))
        x = nn.softplus(nn.Dense(self.hidden_size, kernel_init=hidden_init)(x))
        x = nn.Dense(1, kernel_init=final_init)(x)[0]
        return x**2


class TrainConfig(NamedTuple):
    method: hodel.Method
    nepochs: int
    # Optimizer
    lr: float
    decay_steps: int
    decay_alpha: float = 0.1
    # Solver
    nsteps: int = 2
    # Model
    triplet_cls: type[Triplet] = NNTriplet
    model_cls: type[nn.Module] = BaselineNN
    init_shape: int = 1
    prewarm: bool = False
    hidden_dim: int = 10
    # Noise
    eta: float = 0.0


def get_W(lambda_: jax.Array, aux: TripletAux) -> jax.Array:
    return lambda_ * aux.f + aux.c


def fixed_0(lambda_: jax.Array, aux: TripletAux) -> jax.Array:
    return jnp.array([0.0])


def get_stress(
    strains: jax.Array,
    energy: Triplet,
    Theta: jaxtyping.PyTree,
    r: float = 1e-3,
    L: float = 0.1,
) -> jax.Array:
    """Get true stress from engineering strain"""
    del_strains = strains.repeat(2).reshape(-1, 2)  # 2 equal strains because homogenous
    dEdstrain = jnp.sum(
        jax.vmap(lambda strain_: jax.grad(energy.core_energy_fn, 0)(strain_, Theta))(
            del_strains
        ),
        axis=1,
    )
    # Our stress -> Abaqus edge stress has 4/3 factor
    A = jnp.pi * r**2
    eng_stress = dEdstrain / (A * L) * (4 / 3)
    return eng_stress * (1 + jnp.abs(del_strains[:, 0])) / 1e6  # MPa


def from_csv(
    train_path: str = "data/1d_train.csv", truth_path: str = "data/1d_strain.csv"
):
    """Get training (lambda, xf), truth (strain, stress), and aux object."""
    # Load training trajectory
    with open(train_path) as f:
        reader = csv.reader(f)
        lambda_list = []
        xf_list = []
        for row in reader:
            lambda_list.append(float(row[0]))
            xf_list.append([float(i) for i in row[2:]])
        lambdas = jnp.asarray(lambda_list)
        xf_stars = jnp.asarray(xf_list)

    # Abaqus applied force
    aux = TripletAux(jnp.array([0.0, 1.5]), jnp.array([0.0, 0.0]))

    # Load ground truth (includes (σ, ϵ) pairs in and out-of-distribution)
    with open(truth_path) as f:
        reader = csv.reader(f)
        strain_list = []
        stress_list = []
        for row in reader:
            strain_list.append(float(row[0]))
            stress_list.append(float(row[1]))
        strain_stars = jnp.asarray(strain_list)
        stress_stars = jnp.asarray(stress_list) / 1e6  # to MPa

    return (lambdas, xf_stars), (strain_stars, stress_stars), aux
