import numpy as np

from .mesh import Mesh
from .spin import spin
from .solver import solveEigen, solvePoisson
from .util import timeit


def curvature_transfer_step(V: np.ndarray, F: np.ndarray, target: Mesh, tau: float = 0.1, cupy=False) -> np.ndarray:
    """
    One step curvature transfer.

    Args:
        V: vertex array
        F: faces array
        target: target mesh data structure (TODO: actually we only needs mean curvature half density)
        tau: step size
        cupy: whether to use cupy for acceleration
    
    Returns:
        V_new: vertices after Willmore flow fairing.
    """

    ## set up (face) curvature potential function
    mesh = Mesh(V, F)
    rho = target.face_curvature * np.sqrt(mesh.face_area / target.face_area) - mesh.face_curvature
    rho*= tau

    ## solve Eigen system
    qs = solveEigen(mesh, rho, cupy)

    ## spin transform
    q_hedge = spin(mesh, qs, cupy)
    # print(q_hedge)

    ## solve Poisson system
    ## it's faster to solve Poisson system on CPU when
    ## the system is small
    V_new = solvePoisson(mesh, q_hedge, cupy)

    return V_new