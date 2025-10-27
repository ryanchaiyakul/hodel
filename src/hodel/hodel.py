from __future__ import annotations
from typing import Any, Callable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
import optax
import diffrax

from .interfaces import EnergyModel, ExternalForce
from .root_finders import newton


@partial(
    register_dataclass,
    data_fields=["energy", "forces"],
    meta_fields=["get_xb_fn", "loss_fn", "update_fn"],
)
@dataclass
class HODEL:
    energy: EnergyModel
    forces: Sequence[ExternalForce]
    get_xb_fn: Callable[[jax.Array, Any], jax.Array] | None = None
    loss_fn: Callable[[jax.Array, jax.Array, Any], jax.Array] = (
        lambda xf, xf_star, _: optax.squared_error(xf, xf_star)  # type: ignore
    )
    update_fn: Callable[
        [jax.Array, Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]],
        tuple[jax.Array, Any],
    ] = newton

    @jax.jit
    def get_W(self, lambda_: jax.Array, aux: Any = None) -> jax.Array:
        """W(λ)"""
        Ws = jnp.stack([f.get_grad_energy(lambda_, aux) for f in self.forces])
        return jnp.sum(Ws, axis=0)

    @jax.jit
    def get_xb(self, lambda_: jax.Array, aux: Any = None) -> jax.Array | None:
        """x_b(λ)"""
        return self.get_xb_fn(lambda_, aux) if self.get_xb_fn else None

    @jax.jit
    def get_residual(
        self,
        lambda_: jax.Array,
        xf: jax.Array,
        Theta: Any = None,
        aux: Any = None,
    ) -> jax.Array:
        """F_f(x,Θ,λ)"""
        xb = self.get_xb(lambda_)
        return self.energy.get_grad_energy(xf, xb, Theta, aux) - self.get_W(
            lambda_, aux
        )

    @jax.jit
    def get_dxf_dlambda(
        self,
        lambda_: jax.Array,
        xf0_init: jax.Array,
        Theta: Any = None,
        aux: Any = None,
    ) -> jax.Array:
        """dx_f/dλ"""
        xb = self.get_xb(lambda_)
        xf = self.solve(lambda_, xf0_init, Theta, aux)
        dxfdxfE = self.energy.get_hess_energy(xf, xb, Theta, aux)
        dxfdxbE = self.energy.get_mixed_hess_energy(xf, xb, Theta, aux)
        dxbdlambda = jax.jacobian(self.get_xb, 0)(lambda_)
        dWdlambda = jax.jacobian(self.get_W, 0)(lambda_, aux)
        rhs = (dxfdxbE @ dxbdlambda - dWdlambda).squeeze()
        return jnp.linalg.solve(dxfdxfE, rhs)

    def get_ode_term(self, Theta: Any = None, aux: Any = None) -> diffrax.ODETerm:
        """Diffrax ODETerm for integration."""
        return diffrax.ODETerm(
            lambda t, x, args: -self.get_dxf_dlambda(
                jnp.asarray(t), jnp.asarray(x), Theta, aux
            )
        )

    @partial(jax.jit, static_argnames=["nsteps"])
    def solve(
        self,
        lambda_: jax.Array,
        xf0_init: jax.Array,
        Theta: Any = None,
        aux: Any = None,
        nsteps: int = 20,
    ) -> jax.Array:
        """Solves for x_f which satisfies F_f(x,Θ,λ)=0"""
        return _solve(self, lambda_, xf0_init, Theta, aux, nsteps)

    @partial(jax.jit, static_argnames=["nsteps"])
    def sim(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        Theta: Any = None,
        aux: Any = None,
        nsteps: int = 20,
    ) -> jax.Array:
        """Solve λs where the guess x_f is the prior x_f"""

        def body_fn(
            xf_prev: jax.Array, lambda_: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            xf_new = _solve(self, lambda_, xf_prev, Theta, aux, nsteps)
            return xf_new, xf_new

        _, xfs = jax.lax.scan(body_fn, xf0_init, lambdas)
        return xfs

    @partial(jax.jit, static_argnames=["nsteps"])
    def get_loss(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        xf_stars: jax.Array,
        Theta: Any = None,
        aux: Any = None,
        nsteps: int = 20,
    ) -> jax.Array:
        def body_fn(
            xf_prev: jax.Array, inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            lambda_, xf_star = inputs
            xf = self.solve(lambda_, xf_prev, Theta, aux, nsteps)
            loss = self.loss_fn(xf, xf_star, Theta)
            return xf, loss

        _, losses = jax.lax.scan(body_fn, xf0_init, (lambdas, xf_stars))
        return jnp.mean(losses)

    @partial(jax.jit, static_argnames=["nsteps", "nepochs", "optim"])
    def learn(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        xf_stars: jax.Array,
        Theta0: Any = None,
        aux: Any = None,
        nsteps: int = 20,
        optim: optax.GradientTransformation = optax.adam(1e-2),
        nepochs: int = 50,
    ):
        grad_loss_fn = jax.value_and_grad(self.get_loss, 3)
        opt_state = optim.init(Theta0)

        def body_fn(carry: tuple[jax.Array, optax.OptState], _: jax.Array):
            Theta, opt_state = carry
            L, g = grad_loss_fn(lambdas, xf0_init, xf_stars, Theta, aux, nsteps)
            updates, opt_state = optim.update(g, opt_state, Theta)
            Theta = optax.apply_updates(Theta, updates)
            return (Theta, opt_state), L

        (Theta_final, _), losses = jax.lax.scan(
            body_fn, (Theta0, opt_state), jnp.arange(nepochs)
        )
        return Theta_final, losses


# FIXME: disables jvp or forward differentiation. Should be ok but just a warning
# Solve implementation outside of class because of jax.custom_vjp works poorly with self.solve
@partial(jax.custom_vjp, nondiff_argnames=["aux", "nsteps"])
@partial(jax.jit, static_argnames=["nsteps"])
def _solve(
    self: HODEL,
    lambda_: jax.Array,
    xf0_init: jax.Array,
    Theta: Any = None,
    aux: Any = None,
    nsteps: int = 20,
) -> jax.Array:
    """x_f=argmin E(x_f,x_b,Θ) subject to F_f(x,Θ,λ)=0"""
    # TODO: add early exit
    xb = self.get_xb(lambda_, aux)

    def body_fn(xf: jax.Array, _: jax.Array) -> tuple[jax.Array, None]:
        return self.update_fn(
            xf,
            lambda x: self.get_residual(lambda_, x, Theta, aux),
            lambda x: self.energy.get_hess_energy(x, xb, Theta, aux),
        )

    xf_star, _ = jax.lax.scan(body_fn, xf0_init, jnp.arange(nsteps))
    return xf_star


def _solve_fwd(
    self: HODEL,
    lambda_: jax.Array,
    xf0_init: jax.Array,
    Theta: Any = None,
    aux: Any = None,
    nsteps: int = 20,
) -> tuple[jax.Array, tuple[HODEL, jax.Array, Any, jax.Array]]:
    xf_star = self.solve(lambda_, xf0_init, Theta, aux, nsteps)
    return xf_star, (self, lambda_, Theta, xf_star)


# Signature is nondiff_args, res from fwd, pertubation vector
def _solve_bwd(
    aux: Any,
    nsteps: int,
    res: tuple[HODEL, jax.Array, Any, jax.Array],
    xf_star_bar: jax.Array,
):
    self, lambda_, Theta, xf_star = res
    xb = self.get_xb(lambda_)
    H = self.energy.get_hess_energy(xf_star, xb, Theta, aux)
    x_bar = jnp.linalg.solve(H, xf_star_bar)

    _, vjp_fn = jax.vjp(
        lambda lambda__: self.get_residual(lambda__, xf_star, Theta, aux), lambda_
    )
    (lambda_bar,) = vjp_fn(x_bar)
    lambda_bar = -lambda_bar

    if Theta is not None:
        _, vjp_fn = jax.vjp(
            lambda Theta_: self.get_residual(lambda_, xf_star, Theta_, aux), Theta
        )
        (Theta_bar,) = vjp_fn(x_bar)
        Theta_bar = jax.tree.map(lambda x: -x, Theta_bar)
    else:
        Theta_bar = None

    def zero_or_none(x):
        """Empty gradient for complex pytree aka HODEL"""
        if isinstance(x, (int, float, complex)):
            return 0.0
        if isinstance(x, jax.Array):
            return jnp.zeros_like(x)
        return None  # for callables, objects, etc.

    return (
        jax.tree.map(zero_or_none, self),
        lambda_bar,
        jnp.zeros_like(xf_star),
        Theta_bar,
    )


_solve.defvjp(_solve_fwd, _solve_bwd)
