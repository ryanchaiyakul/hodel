from typing import Any, Callable, Sequence
import jax
import jax.numpy as jnp
import optax
from .interfaces import EnergyModel, ExternalForce


def two_norm(xf: jax.Array, xf_star: jax.Array, Theta: Any = None) -> jax.Array:
    return jnp.linalg.norm(xf_star - xf) ** 2


class HODEL:
    def __init__(
        self,
        energy: EnergyModel,
        forces: Sequence[ExternalForce] = [],
        get_xb: Callable[[jax.Array, Any], jax.Array] | None = None,
        loss: Callable[[jax.Array, jax.Array, Any], jax.Array] = two_norm,
    ):
        self._energy = energy
        self._forces = forces
        self._get_xb = get_xb
        self._loss = loss

    def get_W(self, lambda_: jax.Array, aux: Any = None) -> jax.Array:
        return jnp.sum(
            jnp.stack([f.get_grad_energy(lambda_, aux) for f in self._forces]), axis=0
        )

    def get_xb(self, lambda_: jax.Array, aux: Any = None) -> jax.Array | None:
        return self._get_xb(lambda_, aux) if self._get_xb else None

    def get_residual(
        self,
        lambda_: jax.Array,
        xf: jax.Array,
        Theta: Any = None,
        aux: Any = None,
    ) -> jax.Array:
        xb = self.get_xb(lambda_)
        return self._energy.get_grad_energy(xf, xb, Theta, aux) - self.get_W(
            lambda_, aux
        )

    def get_evo_op(
        self, lambda_: jax.Array, xf0: jax.Array, Theta: Any = None, aux: Any = None
    ):
        xb = self.get_xb(lambda_)
        xf = self.solve(lambda_, xf0, Theta, aux)
        dxfdxfE = self._energy.get_hess_energy(xf, xb, Theta, aux)
        dxfdxbE = self._energy.get_mixed_hess_energy(xf, xb, Theta, aux)
        dxbdlambda = jax.jacobian(self.get_xb, 0)(lambda_)
        dWdlambda = jax.jacobian(self.get_W, 0)(lambda_, aux)
        rhs = (dxfdxbE @ dxbdlambda - dWdlambda).squeeze()
        return jnp.linalg.solve(dxfdxfE, rhs)

    def solve(
        self,
        lambda_: jax.Array,
        xf0_init: jax.Array,
        Theta: Any = None,
        aux: Any = None,
        nsteps: int = 20,
    ) -> jax.Array:
        """Fixed scan to support autodifferentiation"""
        # TODO: Use custom vjp to allow for early tolerance exit etc.
        xb = self.get_xb(lambda_)

        def body_fn(xf: jax.Array, _):
            r = self.get_residual(lambda_, xf, Theta, aux)
            H = self._energy.get_hess_energy(xf, xb, Theta, aux)
            dx = jnp.linalg.solve(H, r)
            xf_new = xf - dx
            return xf_new, None

        xf_final, _ = jax.lax.scan(body_fn, xf0_init, jnp.arange(nsteps))
        return xf_final

    def sim(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        Theta: Any = None,
        aux: Any = None,
        nsteps: int = 20,
    ) -> jax.Array:
        # xf0 = self.solve(jnp.array([0.0]), xf0, Theta)
        def step_fn(
            xf_prev: jax.Array, lambda_: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            xf_new = self.solve(lambda_, xf_prev, Theta, aux, nsteps)
            return xf_new, xf_new

        _, xfs = jax.lax.scan(step_fn, xf0_init, lambdas)
        return xfs

    def get_loss(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        xf_stars: jax.Array,
        Theta: Any = None,
        aux: Any = None,
    ) -> jax.Array:
        def step_fn(
            xf_prev: jax.Array, inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            lambda_, xf_star = inputs
            xf = self.solve(lambda_, xf_prev, Theta, aux)
            loss = self._loss(xf, xf_star, Theta)
            return xf, loss

        _, losses = jax.lax.scan(step_fn, xf0_init, (lambdas, xf_stars))
        return jnp.mean(losses)

    def learn(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        xf_stars: jax.Array,
        Theta0: Any = None,
        aux: Any = None,
        optim: optax.GradientTransformation = optax.adam(1e-2),
        n_epochs: int = 50,
    ):
        grad_loss_fn = jax.value_and_grad(self.get_loss, 3)
        opt_state = optim.init(Theta0)

        def body_fn(carry: tuple[jax.Array, optax.OptState], _: jax.Array):
            Theta, opt_state = carry
            L, g = grad_loss_fn(lambdas, xf0_init, xf_stars, Theta, aux)
            updates, opt_state = optim.update(g, opt_state, Theta)
            Theta = optax.apply_updates(Theta, updates)
            return (Theta, opt_state), L

        (Theta_final, _), losses = jax.lax.scan(
            body_fn, (Theta0, opt_state), jnp.arange(n_epochs)
        )
        return Theta_final, losses
