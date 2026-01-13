import argparse
import time
import jax
import jaxtyping
import jax.numpy as jnp
import optax
import tqdm
import hodel
from helper import (
    Triplet,
    TrainConfig,
    get_W,
    fixed_0,
    HoDELNN,
    get_stress,
    from_csv,
)


# nepochs set for >=30 seconds wall-clock
hodel_3_config = TrainConfig(
    method=hodel.Method.Minimization,
    nepochs=6000,
    lr=1e-3,
    decay_steps=2000,
    decay_alpha=1e-2,
    nsteps=2,
    prewarm=True,
    model_cls=HoDELNN,
)
hodel_1_config = TrainConfig(
    method=hodel.Method.Minimization,
    nepochs=6000,
    lr=1e-2,
    decay_steps=2000,
    decay_alpha=1e-2,
    nsteps=2,
    prewarm=True,
    model_cls=HoDELNN,
)
pinn_config = TrainConfig(
    method=hodel.Method.Residual,
    nepochs=650000,
    lr=1e-3,
    decay_steps=30000,
    decay_alpha=1e-1,
)

deq_config = TrainConfig(
    method=hodel.Method.DEQ,
    nepochs=45000,
    lr=1e-3,
    decay_steps=15000,
    decay_alpha=1e-2,
    nsteps=5,
)
node_config = TrainConfig(
    method=hodel.Method.ODE,
    nepochs=50,
    lr=1e-2,
    decay_steps=1,
    decay_alpha=0.0,
)

config_map = {
    "hodel": {0.01: hodel_1_config, 0.03: hodel_3_config},
    "pinn": pinn_config,
    "deq": deq_config,
    "node": node_config,
}


def run_warm_start(
    key: jax.Array,
    lambdas: jax.Array,
    xf_stars: jax.Array,
    xf0: jax.Array,
    aux: jaxtyping.PyTree,
    lr: float = 1e-1,
    nepochs: int = 100,
):
    ts_energy = Triplet.init(xf0, jnp.array([0.0]))
    sim = hodel.HODEL(ts_energy.get_energy, get_W, fixed_0)

    optim = optax.chain(
        optax.adam(lr),
        optax.keep_params_nonnegative(),
    )

    theta_final, _ = sim.learn(
        lambdas,
        xf0,
        xf_stars,
        jax.random.uniform(key, (2,)),
        aux,
        method=hodel.Method.Residual,
        optim=optim,
        nepochs=nepochs,
    )

    start_K = jnp.mean(theta_final)
    return start_K


def get_start_K(
    key: jax.Array,
    config: TrainConfig,
    lambdas: jax.Array,
    xf_stars: jax.Array,
    xf0: jax.Array,
    aux: jaxtyping.PyTree,
):
    if config.prewarm:
        return run_warm_start(key, lambdas, xf_stars, xf0, aux)
    return jnp.array(1.0)


def build_model_and_solver(
    key: jax.Array,
    config: TrainConfig,
    xf0: jax.Array,
    start_K: jax.Array,
):
    if config.model_cls is HoDELNN:
        model = config.model_cls(config.hidden_dim, start_K)
    else:
        model = config.model_cls(config.hidden_dim)
    theta0 = model.init(key, jnp.zeros(config.init_shape))
    nn_energy = config.triplet_cls.init(xf0, jnp.array([0.0]), model=model)
    sim = hodel.HODEL(nn_energy.get_energy, get_W, fixed_0)
    schedule = optax.cosine_decay_schedule(
        init_value=config.lr,
        decay_steps=config.decay_steps,
        alpha=config.decay_alpha,
    )
    optim = optax.adam(schedule)
    solver_config = hodel.SolverConfig(nsteps=config.nsteps)
    return theta0, nn_energy, sim, optim, solver_config


def run(
    key: jax.Array,
    config: TrainConfig,
    lambdas: jax.Array,
    xf_stars: jax.Array,
    strain_stars: jax.Array,
    stress_stars: jax.Array,
    aux: jaxtyping.PyTree,
):
    xf_stars = xf_stars * (
        1 + config.eta * jax.random.normal(key, shape=xf_stars.shape)
    )
    xf0 = xf_stars[0]
    start_K = get_start_K(key, config, lambdas, xf_stars, xf0, aux)
    theta0, nn_energy, sim, optim, solver_config = build_model_and_solver(
        key, config, xf0, start_K
    )

    def eval_fn(Theta: jaxtyping.PyTree):
        stress_pred = get_stress(strain_stars, nn_energy, Theta)
        return jnp.linalg.norm(stress_stars - stress_pred) / jnp.linalg.norm(
            stress_stars
        )

    _, _, L_test = sim.learn_and_eval(
        lambdas,
        xf0,
        xf_stars,
        theta0,
        aux,
        method=config.method,
        optim=optim,
        nepochs=config.nepochs,
        config=solver_config,
        eval_fn=eval_fn,
    )
    return L_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 1D benchmark.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["hodel", "pinn", "deq", "node"],
        help="The learning framework to use.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=100,
        help="Number of random seeds to run (default: 100).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.01,
        help="Eta noise scaling factor (defaults: 0.01).",
    )
    args = parser.parse_args()
    template_config = config_map[args.method]
    if type(template_config) is dict:
        template_config = template_config[args.eta]
    config = template_config._replace(eta=args.eta)
    (LAMBDAS, XF_STARS), (STRAIN_STARS, STRESS_STARS), AUX = from_csv()
    print(
        f"Starting experiment: {args.method} for {args.seeds} seeds with η={args.eta}..."
    )
    all_test_errors = []
    t0 = time.time()
    for i in tqdm.trange(args.seeds):
        L_test = run(
            jax.random.PRNGKey(i),
            config,
            LAMBDAS,
            XF_STARS,
            STRAIN_STARS,
            STRESS_STARS,
            AUX,
        )
        print(L_test[-1])
        all_test_errors.append(L_test)
    total_time = time.time() - t0
    filename = f"data/results_{args.method}_eta={args.eta}.npz"
    jnp.savez(
        filename,
        test_errors=jnp.array(all_test_errors),
        total_time=total_time,
        method=args.method,
    )
    print(f"Successfully saved results to {filename}")
