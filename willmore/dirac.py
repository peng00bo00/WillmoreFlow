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
    
    ## R
    V = mesh.V
    F = mesh.F

    TT, TTi = igl.triangle_triangle_adjacency(F)
    edge_mapping = np.array([[0, 1], [1, 2], [2, 0]])
    left_vertex  = np.array([2, 0, 1])

    R = QuaternionMatrix((len(F), len(F)))

    ## weights c
    edges = igl.edges(F)
    edges = V[edges[:,1]] - V[edges[:,0]]
    edges = np.sqrt(np.sum(edges*edges, axis=1))
    c = 0.001*np.max(edges)

    for Fi in range(len(F)):
        for j in range(3):
            Fj = TT[Fi, j]

            ## find common edge of the two adjacent faces
            vi, vj = F[Fj][edge_mapping[TTi[Fi, j]]]
            vk = F[Fj][left_vertex[TTi[Fi, j]]]

            u = V[vi] - V[vk]
            v = V[vj] - V[vk]

            ## (half) dual edge
            cot = np.dot(u, v) / np.linalg.norm(np.cross(u, v))
            w = 0.5 * cot * np.linalg.norm(V[vi] - V[vj]) * c

            R[Fi, Fi] += 0.5*w
            R[Fi, Fj] -= 0.5*w
            R[Fj, Fi] -= 0.5*w
            R[Fj, Fj] += 0.5*w
    
    D = D + R
    
    return D
