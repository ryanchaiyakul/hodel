from dataclasses import dataclass

import diffrax


@dataclass(frozen=True)
class SolverConfig:
    nsteps: int = 10
    dt0: float = 1e-2
    max_steps: int = 4096
    solve_xf0: bool = False
    solver: diffrax.AbstractSolver = diffrax.Tsit5()
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()
