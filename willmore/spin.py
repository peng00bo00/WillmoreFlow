import numpy as np

from .mesh import Mesh
from .quaternion import QList
from .util import buildAdj, timeit


@timeit
def spin(mesh: Mesh, qs: QList, cupy=False) -> QList:
    """Perform spin transform on the mesh.

    Args:
        mesh: mesh data structure
        qs; vertex quaternions
    
    Returns:
        q_hedge: quaternion on half edges
    """
    
    V, F = mesh.V, mesh.F

    ## vertex quaternion -> face quaternion
    adj_matrix = buildAdj(V, F, cupy)
    qs = qs.toReal(cupy)

    qphi = adj_matrix @ qs

    if cupy:
        import cupy as cp
        from cupyx.scipy import sparse as cusp

        ## normalize
        scale = cp.linalg.norm(qphi)
        qphi = qphi * (scale / cp.sqrt(len(F)))
        
        ## spin transform with matrix multiplication
        qphi = QList(cp.asnumpy(qphi))

        q1 = (~qphi).data[mesh.face_he_face[:, 0]]
        qHE= mesh.qHE[mesh.face_he_face[:, 1]]
        q2 = qphi[mesh.face_he_face[:, 2]]

        q1 = QList(q1.tolist()).toDiag(cupy)
        qHE= QList(qHE.tolist()).toDiag(cupy)
        q2 = QList(q2.tolist()).toReal(cupy)

        q_hedge = cp.asnumpy(q1 @ qHE @ q2)
        q_hedge = QList(q_hedge)

    else:
        ## normalize
        qphi = QList(qphi)
        scale= np.sqrt(sum([q.norm2() for q in qphi]))
        qphi = qphi * (scale / np.sqrt(len(F)))

        ## perform spin transform on half-edges
        q_hedge = QList([(~qphi[i]) * mesh.qHE[he] * qphi[j] for (i, he, j) in mesh.face_he_face])

        # q1 = (~qphi).data[mesh.face_he_face[:, 0]]
        # qHE= mesh.qHE.data[mesh.face_he_face[:, 1]]
        # q2 = qphi.data[mesh.face_he_face[:, 2]]

        # q_hedge = QList((q1 * qHE * q2).tolist())

    return q_hedge