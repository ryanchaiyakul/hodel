from __future__ import annotations
from jax.tree_util import register_dataclass
from dataclasses import dataclass, field
from typing import Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import diffrax
import jaxtyping

from .root_finders import newton


@partial(
    register_dataclass,
    data_fields=[],
    meta_fields=["get_energy", "get_W_fn", "get_xb_fn", "loss_fn", "update_fn"],
)
@dataclass
class HODEL:
    get_energy: Callable[
        [jax.Array, jaxtyping.PyTree, jaxtyping.PyTree, jaxtyping.PyTree], jax.Array
    ]
    get_W_fn: Callable[[jax.Array, jaxtyping.PyTree], jax.Array] | None = None
    get_xb_fn: Callable[[jax.Array, jaxtyping.PyTree], jax.Array] | None = None
    loss_fn: Callable[[jax.Array, jax.Array, jaxtyping.PyTree], jax.Array] = (
        lambda xf, xf_star, _: optax.squared_error(xf, xf_star)  # type: ignore
    )
    update_fn: Callable[
        [jax.Array, Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]],
        tuple[jax.Array, Any],
    ] = newton
    solve_fn: Callable = field(init=False, repr=False)

    def __post_init__(self):
        self.solve_fn = get_solve(self)

    @jax.jit
    def get_W(
        self, lambda_: jax.Array, aux: jaxtyping.PyTree = None
    ) -> jax.Array | None:
        """W(λ)"""
        return self.get_W_fn(lambda_, aux) if self.get_W_fn else None

    @jax.jit
    def get_xb(
        self, lambda_: jax.Array, aux: jaxtyping.PyTree = None
    ) -> jax.Array | None:
        """x_b(λ)"""
        return self.get_xb_fn(lambda_, aux) if self.get_xb_fn else None

    @jax.jit
    def get_residual(
        self,
        lambda_: jax.Array,
        xf: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
    ) -> jax.Array:
        """F_f(x,Θ,λ)"""
        xb = self.get_xb(lambda_, aux)
        dxE = jax.grad(self.get_energy, 0)(xf, xb, Theta, aux)
        w = self.get_W(lambda_, aux)
        return dxE - w if w is not None else dxE

    @jax.jit
    def get_dxf_dlambda(
        self,
        lambda_: jax.Array,
        xf0_init: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
    ) -> jax.Array:
        """dx_f/dλ"""
        xb = self.get_xb(lambda_, aux)
        xf = self.solve(lambda_, xf0_init, Theta, aux)
        dxfdxfE = jax.hessian(self.get_energy, 0)(xf, xb, Theta, aux)
        dxfdxbE = jax.jacobian(jax.grad(self.get_energy, 0), 1)(xf, xb, Theta, aux)
        dxbdlambda = jax.jacobian(self.get_xb, 0)(lambda_, aux)
        dWdlambda = jax.jacobian(self.get_W, 0)(lambda_, aux)
        rhs = (dxfdxbE @ dxbdlambda - dWdlambda).squeeze()
        return jnp.linalg.solve(dxfdxfE, rhs)

    @jax.jit
    def get_ode_term(
        self, Theta: jaxtyping.PyTree = None, aux: jaxtyping.PyTree = None
    ) -> diffrax.ODETerm:
        """Diffrax ODETerm for integration"""
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
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        nsteps: int = 20,
    ) -> jax.Array:
        """Solves for x_f which satisfies F_f(x,Θ,λ)=0"""
        return self.solve_fn(lambda_, xf0_init, Theta, aux, nsteps)

    @partial(jax.jit, static_argnames=["nsteps"])
    def sim(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        nsteps: int = 20,
    ) -> jax.Array:
        """Solve λs where the guess x_f is the prior x_f"""

        def body_fn(
            xf_prev: jax.Array, lambda_: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            xf_new = self.solve(lambda_, xf_prev, Theta, aux, nsteps)
            return xf_new, xf_new

        _, xfs = jax.lax.scan(body_fn, xf0_init, lambdas)
        return xfs

    @partial(jax.jit, static_argnames=["nsteps"])
    def loss(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        xf_stars: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        nsteps: int = 20,
    ) -> jax.Array:
        xfs = self.sim(lambdas, xf0_init, Theta, aux, nsteps)
        return self.loss_fn(xfs, xf_stars, Theta)

    @partial(jax.jit, static_argnames=["nsteps"])
    def batch_loss(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        batch_xf_stars: jax.Array,
        Theta: jaxtyping.PyTree = None,
        batch_aux: jaxtyping.PyTree = None,
        nsteps: int = 20,
    ):
        batch_solve_fn = jax.vmap(
            lambda aux_: self.sim(lambdas, xf0_init, Theta, aux_, nsteps)
        )
        batch_xfs = batch_solve_fn(batch_aux)
        losses = jax.vmap(self.loss_fn, (0, 0, None))(batch_xfs, batch_xf_stars, Theta)
        return jnp.mean(losses)

    @partial(jax.jit, static_argnames=["nsteps", "optim", "nepochs"])
    def learn(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        xf_stars: jax.Array,
        Theta0: jaxtyping.PyTree,
        aux: jaxtyping.PyTree,
        nsteps: int = 20,
        optim: optax.GradientTransformation = optax.adam(1e-2),
        nepochs: int = 50,
    ):
        grad_loss_fn = jax.value_and_grad(self.loss, 3)
        opt_state = optim.init(Theta0)

        def body_fn(carry: tuple[jaxtyping.PyTree, optax.OptState], _: jax.Array):
            Theta, opt_state = carry
            L, g = grad_loss_fn(lambdas, xf0_init, xf_stars, Theta, aux, nsteps)
            updates, opt_state = optim.update(g, opt_state, Theta)
            Theta = optax.apply_updates(Theta, updates)
            return (Theta, opt_state), L

        (Theta_final, _), losses = jax.lax.scan(
            body_fn, (Theta0, opt_state), jnp.arange(nepochs)
        )
        return Theta_final, losses

    @partial(jax.jit, static_argnames=["batch_size", "nsteps", "optim", "nepochs"])
    def batch_learn(
        self,
        lambdas: jax.Array,
        xf0_init: jax.Array,
        batch_xf_stars: jax.Array,
        Theta0: jaxtyping.PyTree,
        batch_aux: jaxtyping.PyTree,
        batch_size: int = 64,
        nsteps: int = 20,
        optim: optax.GradientTransformation = optax.adam(1e-2),
        nepochs: int = 50,
        key: jax.Array = jax.random.PRNGKey(0),
        test_xf_stars: jax.Array | None = None,
        test_aux: jaxtyping.PyTree | None = None,
    ):
        grad_loss_fn = jax.value_and_grad(self.batch_loss, 3)
        opt_state = optim.init(Theta0)

        # Aux is pytree so check leaf
        n_samples = jax.tree_util.tree_leaves(batch_aux)[0].shape[0]
        n_batches = max(1, n_samples // batch_size)

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
                L, g = grad_loss_fn(lambdas, xf0_init, xf_stars, Theta, aux, nsteps)
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
def get_solve(self: HODEL):
    # FIXME: disables jvp or forward differentiation
    @partial(jax.custom_vjp, nondiff_argnames=["nsteps"])
    def _solve(
        lambda_: jax.Array,
        xf0_init: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        nsteps: int = 20,
    ) -> jax.Array:
        """x_f=argmin_{x_f} E(x_f,x_b,Θ) subject to F_f(x,Θ,λ)=0"""
        # TODO: add early exit
        xb = self.get_xb(lambda_, aux)

        def body_fn(xf: jax.Array, _: jax.Array) -> tuple[jax.Array, None]:
            return self.update_fn(
                xf,
                lambda x: self.get_residual(lambda_, x, Theta, aux),
                lambda x: jax.hessian(self.get_energy, 0)(x, xb, Theta, aux),
            )

        xf_star, _ = jax.lax.scan(body_fn, xf0_init, jnp.arange(nsteps))
        return xf_star

    def _solve_fwd(
        lambda_: jax.Array,
        xf0_init: jax.Array,
        Theta: jaxtyping.PyTree = None,
        aux: jaxtyping.PyTree = None,
        nsteps: int = 20,
    ) -> tuple[
        jax.Array, tuple[jax.Array, jaxtyping.PyTree, jax.Array, jaxtyping.PyTree]
    ]:
        xf_star = _solve(lambda_, xf0_init, Theta, aux, nsteps)
        return xf_star, (xf_star, lambda_, Theta, aux)

    # Signature is nondiff_args, res from fwd, pertubation vector
    def _solve_bwd(
        nsteps: int,
        res: tuple[jax.Array, jax.Array, jaxtyping.PyTree, jaxtyping.PyTree],
        xf_star_bar: jax.Array,
    ):
        xf_star, lambda_, Theta, aux = res
        xb = self.get_xb(lambda_, aux)
        H = jax.hessian(self.get_energy, 0)(xf_star, xb, Theta, aux)
        x_bar = jnp.linalg.solve(H, xf_star_bar)

        _, vjp_fn = jax.vjp(
            lambda lambda__: self.get_residual(lambda__, xf_star, Theta, aux), lambda_
        )
        (lambda_bar,) = vjp_fn(x_bar)
        lambda_bar = jax.tree.map(lambda x: -x, lambda_bar)

        if Theta is not None:
            _, vjp_fn = jax.vjp(
                lambda Theta_: self.get_residual(lambda_, xf_star, Theta_, aux), Theta
            )
            (Theta_bar,) = vjp_fn(x_bar)
            Theta_bar = jax.tree.map(lambda x: -x, Theta_bar)
        else:
            Theta_bar = None

        if aux is not None:
            _, vjp_fn = jax.vjp(
                lambda aux_: self.get_residual(lambda_, xf_star, Theta, aux_), aux
            )
            (aux_bar,) = vjp_fn(x_bar)
            aux_bar = jax.tree.map(lambda x: -x, aux_bar)
        else:
            aux_bar = None

        return (
            lambda_bar,
            jnp.zeros_like(xf_star),  # just a guess
            Theta_bar,
            aux_bar,
        )

    _solve.defvjp(_solve_fwd, _solve_bwd)

    return _solve
