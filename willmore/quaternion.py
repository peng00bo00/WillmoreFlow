from time import time
from typing import Tuple

import numpy as np

import scipy
from scipy.sparse import csr_matrix

############################################################################
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
############################################################################

class Quaternion:
    """Quaternion implementation.
    """
    def __init__(self, *args):
        self.s, self.v = self._create_quaternion(*args)
    
    def __str__(self):
        return f"Quaternion: ({self.s}, {self.v[0]}, {self.v[1]}, {self.v[2]})"
    
    def __repr__(self):
        return f"Quaternion: ({self.s}, {self.v[0]}, {self.v[1]}, {self.v[2]})"
    
    def __getitem__(self, key: int):
        assert 0<=key<4 and isinstance(key, int), f"invalid index: {key}"

        if key == 0:
            return self.s
        else:
            return self.v[key-1]
    
    def __setitem__(self, key: int, value: float):
        assert 0<=key<4 and isinstance(key, int), f"invalid index: {key}"
        
        if key == 0:
            self.s = float(value)
        else:
            self.v[key-1] = float(value)
    
    ## properties
    @property
    def im(self):
        return self.v
    
    @property
    def re(self):
        return self.s
    
    @property
    def conj(self):
        return Quaternion(self.re, -self.im)
    
    @property
    def w(self):
        return self.s

    @property
    def x(self):
        return self.v[0]
    
    @property
    def y(self):
        return self.v[1]
    
    @property
    def z(self):
        return self.v[2]
    
    @property
    def data(self):
        x = np.zeros(4)
        x[0] = self.s
        x[1:]= self.v

        return x
    
    ## operators
    def __add__(self, q):
        """Quaternion add.
        """
        if isinstance(q, QList):
            return QList([self+qq for qq in q])
        elif isinstance(q, Quaternion):
            return Quaternion(self.s+q.s, self.v+q.v)
        
        return self+Quaternion(q)
    
    def __radd__(self, q):
        """Quaternion right add.
        """
        if isinstance(q, Quaternion):
            return Quaternion(q.s+self.s, q.v+self.v)

        return Quaternion(q)+self
    
    def __sub__(self, q):
        """Quaternion subtraction.
        """
        if isinstance(q, QList):
            return QList([self-qq for qq in q])
        elif isinstance(q, Quaternion):
            return Quaternion(self.s-q.s, self.v-q.v)
        
        return self-Quaternion(q)
    
    def __rsub__(self, q):
        """Quaternion right subtraction.
        """
        if isinstance(q, Quaternion):
            return Quaternion(q.s-self.s, q.v-self.v)
        
        return Quaternion(q)-self
    
    def __mul__(self, q):
        """Quaternion multiplication.
        """
        if isinstance(q, QList):
            return QList([self*qq for qq in q])
        elif isinstance(q, Quaternion):
            s1 = self.s
            s2 = q.s

            v1 = self.v
            v2 = q.v

            s = s1*s2 - v1@v2
            v = s1*v2 + s2*v1 + np.cross(v1, v2)

            return Quaternion(s, v)

        elif isinstance(q, (int, float)):
            return Quaternion(self.s*q, self.v*q)
        
        raise TypeError(f"invalid quaternion input q: {q}!")
    
    def __rmul__(self, q):
        """Quaternion right multiplication.
        """
        if isinstance(q, Quaternion):
            ## note that p*q != q*p when p and q are quaternions
            return q * self

        elif isinstance(q, (int, float)):
            return self * q
        
        raise TypeError(f"invalid quaternion input q: {q}!")
    
    def __truediv__(self, s):
        if isinstance(s, (int, float)):
            return Quaternion(self.s/s, self.v/s)

        raise TypeError(f"invalid quaternion input s: {s}!")
    
    def __neg__(self):
        return Quaternion(-self.s, -self.v)
    
    def __pos__(self):
        return Quaternion(self.s, self.v)

    def __invert__(self):
        """Quaternion conjugate.
        """
        return self.conj
    
    @staticmethod
    def _create_quaternion(*args):
        if len(args) == 0:
            return 0., np.zeros(3, dtype=float)
        elif len(args) == 1:
            q = args[0]
            if isinstance(q, np.ndarray):
                assert len(q) == 4, "invalid quaternion input!"
                q = q.astype(float)
                return q[0], q[1:]
            elif isinstance(q, list):
                assert len(q) == 4, "invalid quaternion input!"
                return float(q[0]), np.array(q[1:], dtype=float)
            elif isinstance(q, (int, float)):
                return q, np.zeros(3, dtype=float)

        elif len(args) == 2:
            s, v = args
            assert isinstance(s, (int, float)), "invalid quaternion input!"

            if isinstance(v, list):
                assert len(v) == 3, "invalid quaternion input!"
                v = np.array(v, dtype=float)
                return float(s), v
            elif isinstance(v, np.ndarray):
                assert len(v) == 3, "invalid quaternion input!"
                v = v.astype(float)
                return float(s), v
        
        elif len(args) == 4:
            s, x, y, z = args
            v = np.array([x, y, z], dtype=float)
            return float(s), v
        
        raise TypeError("invalid quaternion input!")
    
    ## useful functions
    def norm2(self):
        return self.s*self.s + self.v @ self.v
    
    def norm(self):
        return np.sqrt(self.norm2())
    
    def normalize(self):
        s = self.norm()
        self.s /= s
        self.v /= s
    
    def unit(self):
        return self / self.norm()
    
    def inv(self):
        return ~self / self.norm2()

    def toMatrix(self) -> np.ndarray:
        """
        Convert the quaternion to matrix representation.
        """
        Q = np.zeros((4, 4))

        s = self.re
        x, y, z = self.im

        Q[0, 0] = s; Q[0, 1] =-x; Q[0, 2] =-y; Q[0,3] =-z;
        Q[1, 0] = x; Q[1, 1] = s; Q[1, 2] =-z; Q[1,3] = y;
        Q[2, 0] = y; Q[2, 1] = z; Q[2, 2] = s; Q[2,3] =-x;
        Q[3, 0] = z; Q[3, 1] =-y; Q[3, 2] = x; Q[3,3] = s;

        return Q


class QuaternionMatrix:
    """Quaternion matrix implementation.
    """

    def __init__(self, size: Tuple[int, int]):
        """Initialize the matrix.

        Args:
            size: size of the array
        """
        self.size = size
        self.data = {}
    
    def __getitem__(self, key: Tuple[int, int]):
        i, j = key
        h, w = self.size

        assert 0<=i<h and 0<=j<w, f"invalid index: {key}"

        return self.data.get(key, Quaternion())
    
    def __setitem__(self, key: Tuple[int, int], value):
        i, j = key
        h, w = self.size

        assert 0<=i<h and 0<=j<w, f"invalid index: {key}"

        if isinstance(value, Quaternion):
            self.data[key] = value
        else:
            self.data[key] = Quaternion(value)
    
    ## operators
    def __add__(self, Q):
        """Quaternion matrix add.
        """
        if isinstance(Q, QuaternionMatrix):
            assert self.shape == Q.shape, "shape not matched!"

            M = QuaternionMatrix(self.size)

            for (i, j), q in self.items():
                M[i, j] += q
            
            for (i, j), q in Q.items():
                M[i, j] += q
            
            return M
        
        raise TypeError("invalid quaternion matrix input!")
    
    def __sub__(self, Q):
        """Quaternion matrix subtraction.
        """
        if isinstance(Q, QuaternionMatrix):
            assert self.shape == Q.shape, "shape not matched!"

            M = QuaternionMatrix(self.size)

            for (i, j), q in self.items():
                M[i, j] += q
            
            for (i, j), q in Q.items():
                M[i, j] -= q
            
            return M
        
        raise TypeError("invalid quaternion matrix input!")
    
    def __neg__(self):
        M = QuaternionMatrix(self.size)
        M.data = {k:-v for k, v in self.items()}
        return M
    
    def __pos__(self):
        M = QuaternionMatrix(self.size)
        M.data = self.data.copy()
        return M
    
    @property
    def shape(self):
        return self.size
    
    ## useful functions
    def items(self):
        return self.data.items()

    def copy(self):
        M = QuaternionMatrix(self.size)
        M.data = self.data.copy()
        return M

    ## TODO: this is time consuming, try to optimize with joblib
    @timeit
    def toReal(self, cupy=False) -> csr_matrix:
        """Convert the quaternion matrix to a sparse matrix.
        """

        h, w = self.size        
        N = len(self.data)

        value, rows, cols = np.zeros(N*16, dtype=float), np.zeros(N*16, dtype=int), np.zeros(N*16, dtype=int)

        ## indices offsets
        _rows = np.tile(np.arange(4, dtype=int), (4, 1)).T.flatten()
        _cols = np.tile(np.arange(4, dtype=int), (4, 1)).flatten()

        for k, ((i, j), q) in enumerate(self.items()):
            Q = q.toMatrix()

            value[16*k: 16*k+16] = Q.flatten()

            rows[16*k: 16*k+16] = 4*i + _rows
            cols[16*k: 16*k+16] = 4*j + _cols

        if cupy:
            import cupy as cp
            from cupyx.scipy import sparse as csp

            M = csp.csr_matrix((cp.array(value), (cp.array(rows), cp.array(cols))), shape=(4*h, 4*w))
        
        else:            
            M = csr_matrix((value, (rows, cols)), shape=(4*h, 4*w))

        return M


class QList:
    """Quaternion list implementation.
    """

    def __init__(self, arg):
        """Initialize QList.
        """
        self.data = self._create_list(arg)
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()
    
    def __getitem__(self, key: int):
        return self.data[key]
    
    def __setitem__(self, key: int, value: Quaternion):
        self.data[key] = value
    
    def __len__(self):
        return len(self.data)
    
    @property
    def im(self):
        return np.array([q.im for q in self.data])
    
    @property
    def re(self):
        return np.array([q.re for q in self.data])
    
    @property
    def conj(self):
        return QList([q.conj for q in self.data])
    
    ## operators
    def __add__(self, q):
        """Element-wise add.
        """
        if isinstance(q, Quaternion) or isinstance(q, (int, float)):
            return QList([qq+q for qq in self.data])
        elif isinstance(q, QList):
            assert len(self) == len(q), "Length unmatched!"
            return QList([q1+q2 for q1, q2 in zip(self.data, q.data)])
        
        raise TypeError(f"Invalid sum with {q}")
    
    def __radd__(self, q):
        """Element-wise add.
        """
        if isinstance(q, Quaternion) or isinstance(q, (int, float)):
            return QList([q+qq for qq in self.data])
        elif isinstance(q, QList):
            assert len(self) == len(q), "Length unmatched!"
            return QList([q1+q2 for q1, q2 in zip(q.data, self.data)])
        
        raise TypeError(f"Invalid sum with {q}")
    
    def __sub__(self, q):
        """Element-wise subtraction.
        """
        if isinstance(q, Quaternion) or isinstance(q, (int, float)):
            return QList([qq-q for qq in self.data])
        
        elif isinstance(q, QList):
            assert len(self) == len(q), "Length unmatched!"
            return QList([q1-q2 for q1, q2 in zip(self.data, q.data)])

        raise TypeError(f"Invalid subtraction with {q}")
    
    def __rsub__(self, q):
        """Element-wise subtraction.
        """
        if isinstance(q, Quaternion) or isinstance(q, (int, float)):
            return QList([q-qq for qq in self.data])
        
        elif isinstance(q, QList):
            assert len(self) == len(q), "Length unmatched!"
            return QList([q1-q2 for q1, q2 in zip(q.data, self.data)])

        raise TypeError(f"Invalid subtraction with {q}")
    
    def __mul__(self, q):
        """Element-wise product.
        """
        if isinstance(q, (Quaternion, int, float)):
            return QList([qq*q for qq in self.data])
        
        raise TypeError(f"Invalid product with {q}")
    
    def __rmul__(self, q):
        """Element-wise product.
        """
        if isinstance(q, (Quaternion, int, float)):
            return QList([q*qq for qq in self.data])
        
        raise TypeError(f"Invalid product with {q}")
    
    def __truediv__(self, q):
        """Element-wise division.
        """
        if isinstance(q, (int, float)):
            return QList([qq/q for qq in self.data])

        raise TypeError(f"Invalid division with {q}")
    
    def __neg__(self):
        return QList([-q for q in self.data])
    
    def __pos__(self):
        return QList([+q for q in self.data])
    
    def __invert__(self):
        return self.conj
    
    @staticmethod
    def _create_list(arg):
        if isinstance(arg, list):
            for q in arg:
                assert isinstance(q, Quaternion), "Invalid list input!"
            
            return np.array(arg, dtype=object)

        elif isinstance(arg, np.ndarray):
            assert len(arg) % 4 == 0, "Invalid list input!"

            N = len(arg) // 4
            
            return np.array([Quaternion(arg[4*i: 4*i+4]) for i in range(N)], dtype=object)
        
        raise TypeError("invalid quaternion list input!")

    def toReal(self, cupy=False):
        """Convert the list to a vector.
        """

        data = np.array([q.data for q in self.data]).flatten()
        
        if cupy:
            import cupy as cp
            
            data = cp.array(data)
        
        return data
    
    @timeit
    def toDiag(self, cupy=False):
        """Convert the list to a block diagonal matrix.
        """

        N = len(self)

        value, rows, cols = np.zeros(N*16, dtype=float), np.zeros(N*16, dtype=int), np.zeros(N*16, dtype=int)

        ## indices offsets
        _rows = np.tile(np.arange(4, dtype=int), (4, 1)).T.flatten()
        _cols = np.tile(np.arange(4, dtype=int), (4, 1)).flatten()

        for i in range(N):
            Q = self.data[i].toMatrix()
            value[16*i: 16*i+16] = Q.flatten()

            rows[16*i: 16*i+16] = 4*i + _rows
            cols[16*i: 16*i+16] = 4*i + _cols

        if cupy:
            import cupy as cp
            from cupyx.scipy import sparse as csp

            M = csp.csr_matrix((cp.array(value), (cp.array(rows), cp.array(cols))), shape=(4*N, 4*N))

        else:
            M = csr_matrix((value, (rows, cols)), shape=(4*N, 4*N))
        
        return M