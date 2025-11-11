import typing
import numpy as np

from .params import Geometry, Material


def get_rod_stiffness(
    geom: Geometry, material: Material
) -> typing.Tuple[float, float, float, float]:
    A = geom.axs if geom.axs else np.pi * geom.rod_r0**2
    EA = material.youngs_rod * A

    if geom.ixs1 and geom.ixs2:
        EI1 = material.youngs_rod * geom.ixs1
        EI2 = material.youngs_rod * geom.ixs2
    else:
        EI1 = EI2 = material.youngs_rod * np.pi * geom.rod_r0**4 / 4

    # TODO: what is proper name
    something = geom.jxs if geom.jxs else np.pi * geom.rod_r0**4 / 2
    GJ = material.youngs_rod / (2 * (1 + material.poisson_rod)) * something

    return EA, EI1, EI2, GJ


def get_shell_stiffness(
    geom: Geometry, material: Material, ref_len: np.ndarray, use_mid_edge: bool
) -> typing.Tuple[float, float]:
    # TODO: rewrite this because it doesn't make sense
    if use_mid_edge:
        ks = (
            2
            * material.youngs_shell
            * geom.shell_h
            / (1 - material.poisson_shell**2)
            * ref_len
        )
        kb = (
            material.youngs_shell
            * geom.shell_h**3
            / (24 * (1 - material.poisson_shell**2))
        )
    else:
        ks = (3**0.5 / 2) * material.youngs_shell * geom.shell_h * ref_len
        kb = (2 / 3**0.5) * material.youngs_shell * (geom.shell_h**3) / 12
    return ks, kb  # type: ignore
