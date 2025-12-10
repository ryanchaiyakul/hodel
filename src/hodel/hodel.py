from __future__ import annotations
from jax.tree_util import register_dataclass
from dataclasses import dataclass, field
from typing import cast, Any, Callable
from functools import partial
from enum import Enum

import jax
import jax.numpy as jnp
import optax
import diffrax
import jaxtyping

from .config import SolverConfig
from .root_finders import newton


class Method(Enum):
    Residual = 0
    Minimization = 1
    ODE = 2


@partial(
    register_dataclass,
    data_fields=[],
    meta_fields=[
        "get_energy",
        "get_W_fn",
        "get_xb_fn",
        "carry_fn",
        "loss_fn",
        "update_fn",
    ],
)
@dataclass
class HODEL:
    """
    PyTree which "glues" the various methods which make up HoDEL
    """

    get_energy: Callable[
        [
            jax.Array,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
        ],
        jax.Array,
    ]
    get_W_fn: Callable[[jax.Array, jaxtyping.PyTree], jax.Array] | None = None
    get_xb_fn: Callable[[jax.Array, jaxtyping.PyTree], jax.Array] | None = None
    carry_fn: (
        Callable[
            [jax.Array, jaxtyping.PyTree, jaxtyping.PyTree, jaxtyping.PyTree],
            jaxtyping.PyTree,
        ]
        | None
    ) = None
    loss_fn: Callable[[jax.Array, jax.Array, jaxtyping.PyTree], jax.Array] = (
        lambda xf, xf_star, _: jnp.mean(jnp.linalg.norm(xf - xf_star) ** 2)
    )
    update_fn: Callable[
        [
            jax.Array,
            Callable[[jax.Array], jax.Array],
            Callable[[jax.Array], jaxtyping.PyTree],
            jaxtyping.PyTree,
        ],
        tuple[jax.Array, Any],
    ] = newton
    solve_fn: Callable[
        [
            jax.Array,
            jax.Array,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
            int,
        ],
        jax.Array,
    ] = field(init=False, repr=False)

    def __post_init__(self):
        self.solve_fn = get_solve(self)

    def get_W(
        self, lambda_: jax.Array, aux: jaxtyping.PyTree = None
    ) -> jax.Array | None:
        """W(λ)"""
        return self.get_W_fn(lambda_, aux) if self.get_W_fn else None

    def get_xb(
        self, lambda_: jax.Array, aux: jaxtyping.PyTree = None
    ) -> jax.Array | None:
        """x_b(λ)"""
        return self.get_xb_fn(lambda_, aux) if self.get_xb_fn else None

    def get_residual(
        self,
        lambda_: jax.Array,
        xf: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
    ) -> jax.Array:
        """F_f(x,Θ,λ)"""
        xb = self.get_xb(lambda_, aux)
        dxE = jax.grad(self.get_energy, 0)(xf, xb, Theta, aux, carry)
        w = self.get_W(lambda_, aux)
        return dxE - w if w is not None else dxE

    def get_dxf_dlambda(
        self,
        lambda_: jax.Array,
        xf: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
    ) -> jax.Array:
        """dx_f/dλ"""
        xb = self.get_xb(lambda_, aux)
        dxfdxfE = jax.hessian(self.get_energy, 0)(xf, xb, Theta, aux, carry)
        dxfdxbE = jax.jacobian(jax.grad(self.get_energy, 0), 1)(
            xf, xb, Theta, aux, carry
        )
        dxbdlambda = jax.jacobian(self.get_xb, 0)(lambda_, aux)
        dWdlambda = jax.jacobian(self.get_W, 0)(lambda_, aux)
        rhs = (dxfdxbE @ dxbdlambda - dWdlambda).squeeze()
        return -jnp.linalg.solve(dxfdxfE, rhs)

    def update_carry(
        self,
        lambda_: jax.Array,
        xf: jax.Array,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
    ) -> jaxtyping.PyTree:
        if self.carry_fn:
            return self.carry_fn(xf, self.get_xb(lambda_, aux), aux, carry)
        return carry

    def get_ode_term(
        self,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
    ) -> diffrax.ODETerm:
        """Diffrax ODETerm for integration. Carry updates from initial only."""

        def term(
            t: jaxtyping.PyTree, x: jaxtyping.PyTree, args: jaxtyping.PyTree
        ) -> jaxtyping.PyTree:
            lambda_ = jnp.asarray(t)
            xf = jnp.asarray(x)
            return self.get_dxf_dlambda(lambda_, xf, Theta, aux, carry)

        return diffrax.ODETerm(term)

    @partial(jax.jit, static_argnames=["config"])
    def solve(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        config: SolverConfig = SolverConfig(),
    ) -> jax.Array:
        """Iteratively solve λs for x_f which minimizes F_f(x_f,Θ,λ)=0."""
        hodel_carry = carry

        def body_fn(
            carry: tuple[jax.Array, jaxtyping.PyTree], lambda_: jax.Array
        ) -> tuple[tuple[jax.Array, jaxtyping.PyTree], jax.Array]:
            xf_prev, carry_prev = carry
            xf_new = self.solve_fn(
                lambda_, xf_prev, Theta, aux, carry_prev, config.nsteps
            )
            carry_new = self.update_carry(lambda_, xf_new, aux, carry_prev)
            return (xf_new, carry_new), xf_new

        _, xfs = jax.lax.scan(body_fn, (xf0_init, hodel_carry), lambdas)
        return xfs

    @partial(jax.jit, static_argnames=["config"])
    def ode_solve(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        config: SolverConfig = SolverConfig(),
    ) -> jax.Array:
        """Using the `self.get_ode_term`, forward propogate from λ=[0.0, 1.0]"""
        term = self.get_ode_term(Theta, aux, carry)
        saveat = diffrax.SaveAt(ts=lambdas)
        if config.solve_xf0:
            y0 = self.solve(jnp.array([0.0]), xf0_init, Theta, aux, carry, config)[0]
        else:
            y0 = xf0_init

        return cast(
            jax.Array,
            diffrax.diffeqsolve(
                term,
                config.solver,
                t0=jnp.min(lambdas),
                t1=jnp.max(lambdas),
                dt0=config.dt0,
                y0=y0,
                saveat=saveat,
                max_steps=config.max_steps,
                stepsize_controller=config.stepsize_controller,
            ).ys,
        )

    @partial(jax.jit, static_argnames=["method", "config"])
    def loss(
        self,
        lambdas: jax.Array,
        xf0: jax.Array,
        xf_stars: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        method: Method = Method.ODE,
        config: SolverConfig = SolverConfig(),
    ) -> jax.Array:
        if method == Method.Residual:

            def get_batch_residual(lambdas, xfs):
                return jax.vmap(
                    lambda lambda_, xf_: self.get_residual(
                        lambda_, xf_, Theta, aux, carry
                    )
                )(lambdas, xfs)

            pred = get_batch_residual(lambdas, xf_stars)
            y = jnp.zeros_like(pred)
        if method == Method.Minimization:
            pred = self.solve(lambdas, xf0, Theta, aux, carry, config)
            y = xf_stars
        else:
            pred = self.ode_solve(lambdas, xf0, Theta, aux, carry, config)
            y = xf_stars
        return self.loss_fn(pred, y, Theta)

    @partial(jax.jit, static_argnames=["method", "config", "optim", "nepochs"])
    def learn(
        self,
        lambdas: jax.Array,
        xf0: jax.Array,
        xf_stars: jax.Array,
        Theta0: jaxtyping.PyTree,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        method: Method = Method.Minimization,
        config: SolverConfig = SolverConfig(solve_xf0=True, dt0=1e-3, max_steps=10000),
        optim: optax.GradientTransformation = optax.adam(1e-2),
        nepochs: int = 10,
    ):
        grad_loss_fn = jax.value_and_grad(self.loss, 3)
        opt_state = optim.init(Theta0)

        hodel_carry = carry

        def body_fn(carry: tuple[jaxtyping.PyTree, optax.OptState], _: jax.Array):
            Theta, opt_state = carry
            L, g = grad_loss_fn(
                lambdas, xf0, xf_stars, Theta, aux, hodel_carry, method, config
            )
            updates, new_opt_state = optim.update(g, opt_state, Theta)
            new_Theta = optax.apply_updates(Theta, updates)
            return (new_Theta, new_opt_state), L

        (Theta_final, _), losses = jax.lax.scan(
            body_fn, (Theta0, opt_state), jnp.arange(nepochs)
        )
        return Theta_final, losses

    def batch_loss(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        batch_xf_stars: jax.Array,
        Theta: jaxtyping.PyTree = None,
        batch_aux: jaxtyping.PyTree = None,
        batch_carry: jaxtyping.PyTree = None,
        method: Method = Method.Minimization,
        **kwargs,
    ):
        if method == Method.Residual:

            def get_batch_residual(lambdas, xfs, aux_, carry_):
                return jax.vmap(
                    lambda lambda_, xf_: self.get_residual(
                        lambda_, xf_, Theta, aux_, carry_
                    )
                )(lambdas, xfs)

            raise NotImplementedError
        if method == Method.Minimization:
            batch_solve_fn = jax.vmap(
                lambda aux_, carry_: self.solve(
                    lambdas, xf0_init, Theta, aux_, carry_, **kwargs
                )
            )
            pred = batch_solve_fn(batch_aux, batch_carry)
            y = batch_xf_stars
        else:
            batch_solve_fn = jax.vmap(
                lambda aux_, carry_: self.ode_solve(
                    lambdas, xf0_init, Theta, aux_, carry_, **kwargs
                )
            )
            pred = batch_solve_fn(batch_aux, batch_carry)
            y = batch_xf_stars
        return self.loss_fn(pred, y, Theta)

    # TODO: Batch carry
    @partial(
        jax.jit,
        static_argnames=[
            "method",
            "nsteps",
            "optim",
            "nepochs",
            "batch_size",
            "kwargs",
        ],
    )
    def batch_learn(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        batch_xf_stars: jax.Array,
        Theta0: jaxtyping.PyTree,
        batch_aux: jaxtyping.PyTree = None,
        batch_carry: jaxtyping.PyTree = None,
        method: Method = Method.ODE,
        optim: optax.GradientTransformation = optax.adam(1e-2),
        nepochs: int = 50,
        batch_size: int = 64,
        key: jax.Array = jax.random.PRNGKey(0),
        **kwargs,
    ):
        grad_loss_fn = jax.value_and_grad(self.batch_loss, 3)
        opt_state = optim.init(Theta0)

        # Aux is pytree so check leaf
        n_samples = jax.tree_util.tree_leaves(batch_aux)[0].shape[0]
        n_batches = max(1, n_samples // batch_size)

        hodel_carry = batch_carry  # alias

        def epoch_fn(
            carry: tuple[jaxtyping.PyTree, optax.OptState, jax.Array], _: jax.Array
        ):
            Theta, opt_state, key = carry

            # Shuffle x_f* and aux every epoch
            new_key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, n_samples)
            shuffled_xf_stars = batch_xf_stars[perm]
            shuffled_aux = jax.tree.map(lambda x: x[perm], batch_aux)

            def body_fn(carry: tuple[jaxtyping.PyTree, optax.OptState], idx: jax.Array):
                # Dynamic slicing
                xf_stars = jax.lax.dynamic_slice(
                    shuffled_xf_stars,
                    (idx * batch_size,) + (0,) * (shuffled_xf_stars.ndim - 1),
                    (batch_size,) + shuffled_xf_stars.shape[1:],
                )
                aux = jax.tree.map(
                    lambda x: jax.lax.dynamic_slice(
                        x,
                        (idx * batch_size,) + (0,) * (x.ndim - 1),
                        (batch_size,) + x.shape[1:],
                    ),
                    shuffled_aux,
                )

                Theta, opt_state = carry
                L, g = grad_loss_fn(
                    lambdas,
                    xf0_init,
                    xf_stars,
                    Theta,
                    aux,
                    hodel_carry,
                    method,
                    **kwargs,
                )
                updates, opt_state = optim.update(g, opt_state, Theta)
                Theta = optax.apply_updates(Theta, updates)
                return (Theta, opt_state), L

            (Theta_new, new_opt_state), epoch_loss = jax.lax.scan(
                body_fn, (Theta, opt_state), jnp.arange(n_batches)
            )
            return (Theta_new, new_opt_state, new_key), jnp.mean(epoch_loss)

        (Theta_final, _, _), losses = jax.lax.scan(
            epoch_fn, (Theta0, opt_state, key), jnp.arange(nepochs)
        )
        return Theta_final, losses


# Solve implementation outside of class because of jax.custom_vjp works poorly with self.solve
def get_solve(
    self: HODEL,
) -> Callable[
    [jax.Array, jax.Array, jaxtyping.PyTree, jaxtyping.PyTree, jaxtyping.PyTree, int],
    jax.Array,
]:
    # FIXME: disables jvp or forward differentiation
    @partial(jax.custom_vjp, nondiff_argnames=["nsteps"])
    def _solve(
        lambda_: jax.Array,
        xf0_init: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        nsteps: int = 20,
    ) -> jax.Array:
        """x_f=argmin_{x_f} E(x_f,x_b,Θ)-w(λ)^Tx_f subject to F_f(x,Θ,λ)=0"""
        # TODO: add early exit
        xb = self.get_xb(lambda_, aux)

        def body_fn(xf: jax.Array, _: jax.Array) -> tuple[jax.Array, None]:
            return self.update_fn(
                xf,
                lambda x: self.get_residual(lambda_, x, Theta, aux, carry),
                lambda x: jax.hessian(self.get_energy, 0)(x, xb, Theta, aux, carry),
                aux,
            )

        xf_star, _ = jax.lax.scan(body_fn, xf0_init, jnp.arange(nsteps))
        """jax.debug.print(
            "err: {}",
            jnp.linalg.norm(self.get_residual(lambda_, xf_star, Theta, aux, carry)),
        )"""
        return xf_star

    def _solve_fwd(
        lambda_: jax.Array,
        xf0_init: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        nsteps: int = 20,
    ) -> tuple[
        jax.Array,
        tuple[
            jax.Array, jaxtyping.PyTree, jax.Array, jaxtyping.PyTree, jaxtyping.PyTree
        ],
    ]:
        xf_star = _solve(lambda_, xf0_init, Theta, aux, carry, nsteps)
        return xf_star, (xf_star, lambda_, Theta, aux, carry)

    # Signature is nondiff_args, res from fwd, pertubation vector
    def _solve_bwd(
        nsteps: int,
        res: tuple[
            jax.Array, jax.Array, jaxtyping.PyTree, jaxtyping.PyTree, jaxtyping.PyTree
        ],
        xf_star_bar: jax.Array,
    ):
        xf_star, lambda_, Theta, aux, carry = res
        xb = self.get_xb(lambda_, aux)
        H = jax.hessian(self.get_energy, 0)(xf_star, xb, Theta, aux, carry)
        x_bar = jnp.linalg.solve(H, xf_star_bar)

        _, vjp_fn = jax.vjp(
            lambda lambda__: self.get_residual(lambda__, xf_star, Theta, aux, carry),
            lambda_,
        )
        (lambda_bar,) = vjp_fn(x_bar)
        lambda_bar = jax.tree.map(lambda x: -x, lambda_bar)

        if Theta is not None:
            _, vjp_fn = jax.vjp(
                lambda Theta_: self.get_residual(lambda_, xf_star, Theta_, aux, carry),
                Theta,
            )
            (Theta_bar,) = vjp_fn(x_bar)
            Theta_bar = jax.tree.map(lambda x: -x, Theta_bar)
        else:
            Theta_bar = None

        # None is cheaper than zeros_like
        xf0_init_bar = jnp.zeros_like(xf_star)  # just a guess
        aux_bar = None
        carry_bar = None
        return (
            lambda_bar,
            xf0_init_bar,
            Theta_bar,
            aux_bar,
            carry_bar,
        )

    _solve.defvjp(_solve_fwd, _solve_bwd)

    return _solve
