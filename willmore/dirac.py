import numpy as np

from .mesh import Mesh
from .quaternion import QuaternionMatrix

from .util import timeit

def DiracInstrinsic(mesh: Mesh) -> QuaternionMatrix:
    """Instrinsic Dirac operator.

    Args:
        mesh: mesh data structure.
    
    Returns:
        D: intrinsic Dirac operaor
    """

    F = mesh.F
    D = QuaternionMatrix((len(F), len(F)))

    for i, he, j in mesh.face_he_face:
        D[i, j] = mesh.qHE[he]
    
    return D

def DiracExtrinsic(mesh: Mesh) -> QuaternionMatrix:
    """Instrinsic Dirac operator.

    Args:
        mesh: mesh data structure.
    
    Returns:
        D: extrinsic Dirac operaor
    """

    EPS = 1e-7

    D = DiracInstrinsic(mesh)

    for i in range(len(mesh.F)):
        D[i, i] = (EPS - mesh.face_curvature[i])
    
    return D

@timeit
def buildDirac(mesh: Mesh, rho: np.ndarray) -> QuaternionMatrix:
    """Create Dirac operator with curvature potential.

    Args:
        mesh: mesh data structure
        rho: curvature potential defined on faces
    
    Returns:
        D: Df-rho operator for Dirac equation
    """
    
    D = DiracExtrinsic(mesh)
    
    ## project to some space so that the equation has a solution
    rho = rho - rho.sum()/mesh.face_area.sum() * mesh.face_area
    
    ## substract rho for Dirac equation
    for i in range(len(mesh.F)):
        D[i, i]-= rho[i]
    
    return D