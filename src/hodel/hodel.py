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

from .root_finders import newton


@dataclass(frozen=True)
class SolverConfig:
    nsteps: int = 10
    dt0: float = 1e-2
    max_steps: int = 4096
    solver: diffrax.AbstractSolver = diffrax.Tsit5()
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()


class Method(Enum):
    """Choose loss formulation."""

    Residual = 0
    Minimization = 1  # HoDEL
    ODE = 2
    DEQ = 3  # Minimization w/o HoDEL


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
    PyTree which "glues" the various methods that make up HoDEL.
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
    get_W_fn: (
        Callable[
            [
                jax.Array,
                jaxtyping.PyTree,
            ],
            jax.Array,
        ]
        | None
    ) = None
    get_xb_fn: (
        Callable[
            [
                jax.Array,
                jaxtyping.PyTree,
            ],
            jax.Array,
        ]
        | None
    ) = None
    carry_fn: (
        Callable[
            [
                jax.Array,
                jaxtyping.PyTree,
                jaxtyping.PyTree,
                jaxtyping.PyTree,
            ],
            jaxtyping.PyTree,
        ]
        | None
    ) = None
    loss_fn: Callable[
        [
            jax.Array,
            jax.Array,
            jaxtyping.PyTree,
        ],
        jax.Array,
    ] = lambda xf, xf_star, _: jnp.mean(jnp.linalg.norm(xf - xf_star) ** 2)  # MSE
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
            jaxtyping.PyTree,
            jaxtyping.PyTree,
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
        W = self.get_W(lambda_, aux)
        return self._residual_core(xf, xb, W, Theta, aux, carry)

    @jax.jit
    def _residual_core(
        self,
        xf: jax.Array,
        xb: jaxtyping.PyTree,
        W: jaxtyping.PyTree,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
    ):
        dxE = jax.grad(self.get_energy, 0)(xf, xb, Theta, aux, carry)
        return dxE - W if W is not None else dxE

    def get_dxf_dlambda(
        self,
        lambda_: jax.Array,
        xf: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
    ) -> jax.Array:
        """dx_f/dλ"""
        # We do not flatten xb/W because this is called by Diffrax solver.
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
        xf: jax.Array,
        xb: jaxtyping.PyTree,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
    ) -> jaxtyping.PyTree:
        if self.carry_fn:
            return self.carry_fn(xf, xb, aux, carry)
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
        xbs = jax.vmap(self.get_xb, in_axes=(0, None))(lambdas, aux)
        Ws = jax.vmap(self.get_W, in_axes=(0, None))(lambdas, aux)
        return self._solve_core(xf0_init, (xbs, Ws), Theta, aux, carry, config)

    @partial(jax.jit, static_argnames=["config"])
    def _solve_core(
        self,
        xf0_init: jax.Array,
        inputs: tuple[jaxtyping.PyTree, jaxtyping.PyTree],
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        config: SolverConfig = SolverConfig(),
    ) -> jax.Array:
        def body_fn(state, cur_inputs):
            xf_prev, carry_prev = state
            xb, W = cur_inputs

            xf_new = self.solve_fn(
                xf_prev, xb, W, Theta, aux, carry_prev, config.nsteps
            )
            carry_new = self.update_carry(xf_new, xb, aux, carry_prev)
            return (xf_new, carry_new), xf_new

        _, xfs = jax.lax.scan(body_fn, (xf0_init, carry), inputs)
        return xfs

    @partial(jax.jit, static_argnames=["config"])
    def ode_solve(
        self,
        lambdas: jax.Array,
        xf0: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        config: SolverConfig = SolverConfig(),
    ) -> jax.Array:
        """Using the `self.get_ode_term`, forward propogate from λ=[0.0, 1.0]"""
        term = self.get_ode_term(Theta, aux, carry)
        saveat = diffrax.SaveAt(ts=lambdas)

        return cast(
            jax.Array,
            diffrax.diffeqsolve(
                term,
                config.solver,
                t0=jnp.min(lambdas),
                t1=jnp.max(lambdas),
                dt0=config.dt0,
                y0=xf0,
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
        method: Method = Method.Minimization,
        config: SolverConfig = SolverConfig(),
    ) -> jax.Array:
        xbs = jax.vmap(self.get_xb, in_axes=(0, None))(lambdas, aux)
        Ws = jax.vmap(self.get_W, in_axes=(0, None))(lambdas, aux)
        return self._loss_core(
            lambdas, xf0, xf_stars, (xbs, Ws), Theta, aux, carry, method, config
        )

    @partial(jax.jit, static_argnames=["method", "config"])
    def _loss_core(
        self,
        lambdas: jax.Array,
        xf0: jax.Array,
        xf_stars: jax.Array,
        inputs: tuple[jaxtyping.PyTree, jaxtyping.PyTree],
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        method: Method = Method.Minimization,
        config: SolverConfig = SolverConfig(),
    ) -> jax.Array:
        if method == Method.Residual:
            residual_fn = partial(
                self._residual_core, Theta=Theta, aux=aux, carry=carry
            )
            pred = jax.vmap(residual_fn)(xf_stars, *inputs)
            y = jnp.zeros_like(pred)
        elif method == Method.Minimization:
            pred = self._solve_core(xf0, inputs, Theta, aux, carry, config)
            y = xf_stars
        elif method == Method.ODE:
            pred = self.ode_solve(lambdas, xf0, Theta, aux, carry, config)
            y = xf_stars
        elif method == Method.DEQ:
            # Use undeformed as reference
            xbs, Ws = inputs
            pred = jax.vmap(
                lambda xb, W: self.solve_fn(
                    xf0, xb, W, Theta, aux, carry, config.nsteps
                )
            )(xbs, Ws)
            y = xf_stars
        else:
            raise ValueError(f"Unsupported method: {method}")

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
        config: SolverConfig = SolverConfig(),
        optim: optax.GradientTransformation = optax.adam(1e-2),
        nepochs: int = 10,
    ):
        grad_loss_fn = jax.value_and_grad(self.loss, 3)
        opt_state = optim.init(Theta0)
        hodel_carry = carry  # alias for body_fn convention

        def body_fn(carry: tuple[jaxtyping.PyTree, optax.OptState], _: jax.Array):
            Theta, opt_state = carry
            L, g = grad_loss_fn(
                lambdas,
                xf0,
                xf_stars,
                Theta,
                aux,
                hodel_carry,
                method,
                config,
            )
            updates, new_opt_state = optim.update(g, opt_state, Theta)
            new_Theta = optax.apply_updates(Theta, updates)
            return (new_Theta, new_opt_state), L

        (Theta_final, _), L = jax.lax.scan(
            body_fn, (Theta0, opt_state), jnp.arange(nepochs)
        )
        return Theta_final, L

    @partial(
        jax.jit, static_argnames=["method", "config", "optim", "nepochs", "eval_fn"]
    )
    def learn_and_eval(
        self,
        lambdas: jax.Array,
        xf0: jax.Array,
        xf_stars: jax.Array,
        Theta0: jaxtyping.PyTree,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        method: Method = Method.Minimization,
        config: SolverConfig = SolverConfig(),
        optim: optax.GradientTransformation = optax.adam(1e-2),
        nepochs: int = 10,
        eval_fn: Callable[[jaxtyping.PyTree], jax.Array] = lambda _: jnp.array(0.0),
    ):
        grad_loss_fn = jax.value_and_grad(self.loss, 3)
        opt_state = optim.init(Theta0)
        hodel_carry = carry  # alias for body_fn convention

        def body_fn(carry: tuple[jaxtyping.PyTree, optax.OptState], _: jax.Array):
            Theta, opt_state = carry
            L, g = grad_loss_fn(
                lambdas,
                xf0,
                xf_stars,
                Theta,
                aux,
                hodel_carry,
                method,
                config,
            )
            updates, new_opt_state = optim.update(g, opt_state, Theta)
            new_Theta = optax.apply_updates(Theta, updates)
            L_test = eval_fn(new_Theta)
            return (new_Theta, new_opt_state), (L, L_test)

        (Theta_final, _), (L, L_test) = jax.lax.scan(
            body_fn, (Theta0, opt_state), jnp.arange(nepochs)
        )
        return Theta_final, L, L_test

    # TODO: Fix batched learning eventually

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
    [
        jax.Array,
        jaxtyping.PyTree,
        jaxtyping.PyTree,
        jaxtyping.PyTree,
        jaxtyping.PyTree,
        jaxtyping.PyTree,
        int,
    ],
    jax.Array,
]:
    # FIXME: disables jvp or forward differentiation
    @partial(jax.custom_vjp, nondiff_argnames=["nsteps"])
    def _solve(
        xf0_init: jax.Array,
        xb: jaxtyping.PyTree,
        W: jaxtyping.PyTree,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        nsteps: int = 20,
    ) -> jax.Array:
        """x_f=argmin_{x_f} E(x_f,x_b,Θ)-w(λ)^Tx_f subject to F_f(x,Θ,λ)=0"""

        # TODO: add early exit
        def body_fn(xf: jax.Array, _: jax.Array) -> tuple[jax.Array, None]:
            return self.update_fn(
                xf,
                lambda x: self._residual_core(x, xb, W, Theta, aux, carry),
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
        xf0_init: jax.Array,
        xb: jaxtyping.PyTree,
        W: jaxtyping.PyTree,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        carry: jaxtyping.PyTree = None,
        nsteps: int = 20,
    ) -> tuple[
        jax.Array,
        tuple[
            jax.Array,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
        ],
    ]:
        xf_star = _solve(xf0_init, xb, W, Theta, aux, carry, nsteps)
        return xf_star, (xf_star, xb, W, Theta, aux, carry)

    # Signature is nondiff_args, res from fwd, pertubation vector
    def _solve_bwd(
        nsteps: int,
        res: tuple[
            jax.Array,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
            jaxtyping.PyTree,
        ],
        xf_star_bar: jax.Array,
    ):
        xf_star, xb, W, Theta, aux, carry = res
        H = jax.hessian(self.get_energy, 0)(xf_star, xb, Theta, aux, carry)
        x_bar = jnp.linalg.solve(H, xf_star_bar)

        if Theta is not None:
            _, vjp_fn = jax.vjp(
                lambda Theta_: self._residual_core(xf_star, xb, W, Theta_, aux, carry),
                Theta,
            )
            (Theta_bar,) = vjp_fn(x_bar)
            Theta_bar = jax.tree.map(lambda x: -x, Theta_bar)
        else:
            Theta_bar = None

        # None is cheaper than zeros_like
        xb_bar = None
        W_bar = None
        xf0_init_bar = jnp.zeros_like(xf_star)  # just a guess
        aux_bar = None
        carry_bar = None
        return (
            xf0_init_bar,
            xb_bar,
            W_bar,
            Theta_bar,
            aux_bar,
            carry_bar,
        )

    _solve.defvjp(_solve_fwd, _solve_bwd)

    return _solve
