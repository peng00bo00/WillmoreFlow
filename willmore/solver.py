import numpy as np
from scipy import sparse

from .mesh import Mesh
from .quaternion import QuaternionMatrix, QList
from .util import buildAdj, vertex_area, ismember, timeit
from .dirac import buildDirac


@timeit
def solveEigen(mesh: Mesh, rho: np.ndarray, cupy=False) -> QList:
    """Solve generalized eigen-value problem to find vertex quaternions.

    Args:
        mesh: mesh data structure
        rho: curvature potential defined on faces
        cupy: whether to use cupy for acceleration
    
    Return:
        qs: vertex quaternions
    """

    ## build Eigen system
    V, F = mesh.V, mesh.F

    VA         = vertex_area(V, F, cupy)
    adj_matrix = buildAdj(V, F, cupy)

    ## TODO: build Dirac and move to GPU is the most time consuming process
    Dirac = buildDirac(mesh, rho).toReal(cupy)

    M1 = adj_matrix.T @ Dirac @ Dirac @ adj_matrix
    M1 = 0.5 * (M1 + M1.T)

    ## solve the smallest eigen value and the corresponding eigen vector
    if cupy:
        import cupy as cp
        from cupyx.scipy import sparse as cusp
        from cupyx.scipy.sparse.linalg import cgs

        ## M2 is the inverse vertex area, different from the scipy version
        M2 = cp.array(1./VA)
        M2 = cp.tile(M2, (4,1)).reshape(-1, order="F")
        M2 = cusp.diags(M2)

        M2 = 0.5 * (M2 + M2.T)

        ## solve eig with inverse iteration
        A = M2 @ M1
        b = cp.ones(4*len(V))

        for i in range(3):
            b /= cp.linalg.norm(b)
            x, _ = cgs(A, b)
            b = x

        eigen_f = cp.asnumpy(x)

        ## move VA back to CPU
        VA = cp.asnumpy(VA)

    else:
        M2 = QuaternionMatrix((len(V), len(V)))

        for i in range(len(V)):
            M2[i, i] += VA[i]
        
        M2 = M2.toReal()
        M2 = 0.5 * (M2 + M2.T)

        _, eigen_f = sparse.linalg.eigsh(M1, k=1, M=M2, which='SM', v0=np.ones(4*len(V)))
    

    ## Dirac alignment
    ## normalize the quaternions with vertex area
    qs = QList(eigen_f.flatten())

    qscale = sum([q.inv()*w for q, w in zip(qs, VA)])
    qscale.normalize()

    ## vertex quaternion
    qs *= qscale
    
    return qs

@timeit
def solvePoisson(mesh: Mesh, q_hedge: QList, cupy=False) -> np.ndarray:
    """Solve Poisson problem to reconstruct vertex coordinates.

    Args:
        mesh: mesh data structure
        q_hedge: quaternion on half edges
        cupy: whether to use cupy for acceleration
    
    Return:
        x: vertex coordinates
    """

    V = mesh.V
    half_edges = mesh.half_edges

    ## face index of each edge
    edges = np.unique(np.sort(half_edges, axis=1), axis=0)
    _, edges_halfedges = ismember(edges, half_edges)

    ## solve linear system Ax=b
    if cupy:
        import cupy as cp
        from cupyx.scipy import sparse as cusp
        from cupyx.scipy.sparse.linalg import cgs

        ## LHS
        index_x = cp.arange(len(edges))
        index_y_minus= cp.array(edges[:, 0])
        index_y_plus = cp.array(edges[:, 1])

        A = cusp.csr_matrix((-cp.ones_like(index_x, dtype=float), (index_x, index_y_minus)), (len(edges), len(V))) + \
            cusp.csr_matrix(( cp.ones_like(index_x, dtype=float), (index_x, index_y_plus )), (len(edges), len(V)))

        A = A[:, :-1]

        AA = A.T @ A

        ## RHS
        b = cp.array(q_hedge.im[edges_halfedges,:])

        x = cp.zeros((len(V)-1, 3))
        for i in range(3):
            Ab = A.T @ b[:, i]
            x[:, i], _ = cgs(AA, Ab)
        
        x = cp.asnumpy(x)
    
    else:
        ## face index of each edge
        edges = np.unique(np.sort(half_edges, axis=1), axis=0)
        _, edges_halfedges = ismember(edges, half_edges)

        ## LHS
        index_x = np.arange(len(edges))
        index_y_minus= edges[:, 0]
        index_y_plus = edges[:, 1]

        A = sparse.csr_matrix((-np.ones_like(index_x), (index_x, index_y_minus)), (len(edges), len(V))) + \
            sparse.csr_matrix(( np.ones_like(index_x), (index_x, index_y_plus )), (len(edges), len(V)))

        A = A[:, :-1]

        ## RHS
        b = q_hedge.im[edges_halfedges,:]

        x = sparse.linalg.spsolve(A.T @ A, A.T @ b)
    
    x = np.concatenate([x, np.zeros((1, 3))], axis=0)

    return x