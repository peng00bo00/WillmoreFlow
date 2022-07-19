import igl
import numpy as np
from scipy.sparse import csr_matrix

from .quaternion import QuaternionMatrix


import time
from functools import wraps


def timeit(func):
    """Time cost profiling.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} s')
        return result
    return timeit_wrapper

def ismember(A, B):
    """Find the index of each element of A in B. Assume that the elements are unique in both A and B.
    Steal from:
        https://github.com/erdogant/ismember/blob/277d1e2907abd0bb7dcbcaf4633535bc40bcee6a/ismember/ismember.py#L102
    """

    def is_row_in(a, b):
        # Get the unique row index
        _, rev = np.unique(np.concatenate((b,a)),axis=0,return_inverse=True)
        # Split the index
        a_rev = rev[len(b):]
        b_rev = rev[:len(b)]
        # Return the result:
        return np.isin(a_rev,b_rev)
    
    # Step 1: Find row-wise the elements of a_vec in b_vec
    bool_ind = is_row_in(A, B)
    common = A[bool_ind]

    # Step 2: Find the indices for b_vec
    # In case multiple-similar rows are detected, take only the unique ones
    common_unique, common_inv = np.unique(common, return_inverse=True, axis=0)
    b_unique, b_ind = np.unique(B, return_index=True, axis=0)
    common_ind = b_ind[is_row_in(b_unique, common_unique)]
    
    return bool_ind, common_ind[common_inv]


def buildAdj(V: np.ndarray, F: np.ndarray, cupy=False) -> csr_matrix:
    """Create vertex-to-face adjacent matrix.

    Args:
        V: vertices
        F: faces
        cupy: whether to use cupy acceleration
    
    Returns:
        Adj: vertex-to-face adjacent matrix
    """
    if cupy:
        import cupy as cp
        from cupyx.scipy import sparse as cusp

        rows = cp.arange((len(F)))*4
        cols = cp.array(F)*4

        ## row indices
        rows = cp.tile(rows, (4,1)) + cp.arange(4).reshape((4,1))
        rows = rows.reshape((-1, 1), order="F")
        rows = cp.tile(rows, (1,3)).reshape((-1, 1), order="F")
        rows = rows.flatten()

        ## col indices
        cols = cols.reshape((-1, 1), order="F")
        cols = cp.tile(cols, (1, 4)) + cp.arange(4).reshape((1,4))
        cols = cols.flatten()

        Adj = cusp.csr_matrix((cp.ones_like(rows, dtype=float), (rows, cols)), shape=(4*len(F), 4*len(V)))
    
    else:
        ## still much easier to work with QuaternionMatrix
        Adj = QuaternionMatrix((len(F), len(V)))

        for i in range(len(F)):
            v0, v1, v2 = F[i]

            Adj[i, v0] += 1
            Adj[i, v1] += 1
            Adj[i, v2] += 1
        
        Adj = Adj.toReal()

    return Adj

def vertex_area(V: np.ndarray, F: np.ndarray, cupy=False) -> np.ndarray:
    """A helper function to find vertex area.

    Args:
        V: vertices
        F: faces
    
    Returns:
        area: vertex area
    """

    face_area = igl.doublearea(V, F)/2
    
    index_x = F.flatten("F")
    index_y = np.tile(np.arange(len(F)), 3).flatten("F")

    data= np.ones_like(index_x)

    if cupy:
        import cupy as cp
        from cupyx.scipy import sparse as cusp

        adj = cusp.csr_matrix((cp.array(data, dtype=float), (cp.array(index_x), cp.array(index_y))), (len(V), len(F)))
        area = adj @ cp.array(face_area) / 3
    
    else:
        adj = csr_matrix((data, (index_x, index_y)), (len(V), len(F)))
        area = adj @ face_area / 3

    return area

def area_normalization(V, F) -> QuaternionMatrix:
    """A helper function to build area normalization matrix R.
    
    Args:
        V: vertices
        F: faces
    
    Returns:
        R: area normalization
    """

    TT, TTi = igl.triangle_triangle_adjacency(F)
    edge_mapping = np.array([[0, 1], [1, 2], [2, 0]])
    left_vertex  = np.array([2, 0, 1])

    R = QuaternionMatrix((len(F), len(F)))

    ## weight c
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
    
    return R