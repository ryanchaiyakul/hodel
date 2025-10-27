from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
import optax
import flax.linen as nn

import hodel


@register_dataclass
@dataclass
class Triplet(hodel.EnergyModel):
    """3 node spring where Theta is [K_1, K_2]"""

    l_k: jax.Array

    @classmethod
    def init(cls, xf0, xb0) -> Triplet:
        x0 = xb0[0]
        x1, x2 = xf0
        return cls(jnp.array([x1 - x0, x2 - x1]))

    @staticmethod
    def get_strain(x0, x1, x2, l_k) -> jax.Array:
        return jnp.array([x1 - x0, x2 - x1]) / l_k

    def get_K(self, del_strain: jax.Array, Theta: jax.Array) -> jax.Array:
        return jnp.diag(jnp.abs(Theta))

    def get_energy(
        self,
        xf: jax.Array,
        xb: jax.Array = jnp.array([]),
        Theta: jax.Array = jnp.array([]),
        aux=None,
    ) -> jax.Array:
        # xb = [x0], xf = [x1, x2]
        x0 = xb[0]
        x1, x2 = xf
        del_strain = self.get_strain(x0, x1, x2, self.l_k) - 1.0
        return 0.5 * del_strain @ self.get_K(del_strain, Theta) @ del_strain


@register_dataclass
@dataclass
class LinearForce(hodel.ExternalForce):
    """Linear force with offset"""

    f: jax.Array
    c: jax.Array

    def get_grad_energy(self, lambda_: jax.Array, aux: Any = None) -> jax.Array:
        return lambda_ * self.f + self.c


def fixed_0(lambda_: jax.Array, _=None) -> jax.Array:
    return jnp.array([0.0])


@register_dataclass
@dataclass
class KSTriplet(Triplet):
    def get_K(self, del_strain, Theta):
        # Theta = [K0_1, alpha_1, K0_2, alpha_2]
        K0_1, alpha_1, K0_2, alpha_2 = Theta
        K1 = K0_1 * jnp.exp(alpha_1 * del_strain[0])
        K2 = K0_2 * jnp.exp(alpha_2 * del_strain[1])
        return jnp.diag(jnp.array([K1, K2]))


class KNet(nn.Module):
    """Simple 2 x 2 PSD module"""

    hidden_size: int

    @nn.compact
    def __call__(self, del_strain):
        x = nn.softplus(
            nn.Dense(
                self.hidden_size,
            )(del_strain)
        )
        x = nn.softplus(
            nn.Dense(
                self.hidden_size,
            )(x)
        )
        # Initialize bias to K=4 to make first executions reasonable.
        # Kernel to 0 is just experimental. It makes the first run strictly DER.
        x = nn.Dense(
            2,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(4.0),
        )(x)
        a, b = x
        a = jax.nn.softplus(a)
        b = jax.nn.softplus(b)
        return jnp.array([[a, 0.0], [0.0, b]])

    @nn.compact
    def psd(self, del_strain):
        # Unused, off diagonals were messing things up but with proper init it should probably work
        x = nn.tanh(nn.Dense(self.hidden_size)(del_strain))
        x = nn.Dense(3)(x)
        a, b, c = x
        a = jax.nn.softplus(a)
        c = jax.nn.softplus(c)
        L = jnp.array([[a, 0.0], [b, c]])
        return L @ L.T


model = KNet(10)


@register_dataclass
@dataclass
class NNTriplet(Triplet):
    """3 node spring where Theta is parameters for a neural network."""

    def get_K(self, del_strain: jax.Array, Theta: Any) -> jax.Array:
        return model.apply(Theta, del_strain)  # type: ignore


if __name__ == "__main__":
    # Setup
    # force is setup to mimic gravity + lambd varying load
    # assume total spring weighs 0.03 kg and each node has equal contribution
    xf0 = jnp.array([1.0, 2.0])
    lambdas = jnp.linspace(0, 1, 10)
    force = LinearForce(jnp.array([0.0, 20.0]), jnp.array([9.81 * 1e-2, 9.81 * 1e-2]))

    # xf*
    Theta_star = jnp.array([5.0, 0.4, 2.0, 0.8])
    ks_energy = KSTriplet.init(xf0, fixed_0(jnp.array([0.0])))
    ks_sim = hodel.HODEL(ks_energy, (force,), fixed_0)
    xf_stars = ks_sim.sim(lambdas, xf0, Theta_star)

    # TODO: Figure out how to make this more seamless
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros(2))
    nn_energy = NNTriplet.init(xf0, fixed_0(jnp.array([0.0])))
    nn_sim = hodel.HODEL(nn_energy, (force,), fixed_0)

    lr = 1e-2
    nepochs = 50

    jax.profiler.start_trace("jax-trace")
    final_params, L = nn_sim.learn(
        lambdas, xf0, xf_stars, params, optim=optax.adam(lr), nepochs=nepochs
    )
    jax.block_until_ready(final_params)
    jax.profiler.stop_trace()