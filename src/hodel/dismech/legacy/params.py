import dataclasses


@dataclasses.dataclass
class Geometry:
    rod_r0: float
    shell_h: float
    axs: float | None = None
    jxs: float | None = None
    ixs1: float | None = None
    ixs2: float | None = None


@dataclasses.dataclass
class Material:
    density: float
    youngs_rod: float
    youngs_shell: float
    poisson_rod: float
    poisson_shell: float


@dataclasses.dataclass
class SimParams:
    # static_sim: bool
    # two_d_sim: bool
    # log_data: bool
    # log_step: int
    # dt: float
    max_iter: int = 25
    # total_time: float
    # plot_step: int
    tol: float = 1e-4
    ftol: float = 1e-4
    dtol: float = 1e-2
