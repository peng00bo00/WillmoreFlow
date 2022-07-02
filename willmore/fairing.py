import numpy as np

from .mesh import Mesh
from .spin import spin
from .solver import solveEigen, solvePoisson
from .util import timeit


@timeit
def WillmoreFlow(mesh: Mesh, tau: float = 0.1, cupy=False) -> np.ndarray:
    """One step Willmore Flow fairing.

    Args:
        mesh: mesh data structure
        tau: step size
        cupy: whether to use cupy for acceleration
    
    Returns:
        V_new: vertices after Willmore flow fairing.
    """

    ## use face curvature as the potential function
    ## rho = -tau * curvature
    rho =-tau * mesh.face_curvature

    ## solve Eigen system
    qs = solveEigen(mesh, rho, cupy)

    ## spin transform
    q_hedge = spin(mesh, qs, cupy)

    ## solve Poisson system
    ## it's faster to solve Poisson system on CPU when
    ## the system is small
    V_new = solvePoisson(mesh, q_hedge, cupy)

    return V_new