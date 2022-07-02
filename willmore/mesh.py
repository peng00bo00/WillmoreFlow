import igl
import numpy as np
from .quaternion import Quaternion, QList

class Mesh:
    """Mesh data structure for spin transform.
    """

    def __init__(self, V, F):
        self.V = V.copy()
        self.F = F.copy()

        self.num_vertices = len(V)
        self.num_faces    = len(F)

        ## half edges
        self.half_edges = igl.oriented_facets(F)

        ## face normals
        Z = np.sqrt(np.array([1/3, 1/3, 1/3]))
        self.normal_face = igl.per_face_normals(V, F, Z)

        ## neighbor face of each triangle
        neighbor_face, _ = igl.triangle_triangle_adjacency(self.F)

        ## swap face cols
        col = neighbor_face[:, 0].copy()
        neighbor_face[:, 0] = neighbor_face[:, 1]
        neighbor_face[:, 1] = col

        col = neighbor_face[:, 1].copy()
        neighbor_face[:, 1] = neighbor_face[:, 2]
        neighbor_face[:, 2] = col
        
        self.neighbor_face = neighbor_face.flatten('F')

        ## initialize other useful attributes
        self.face_area            = igl.doublearea(V, F)/2
        self.face_he_face         = self.get_face_he_face()
        self.halfedges_coordinate = self.get_halfedges_coordinate()
        self.meancurvature_edge   = self.get_meancurvature_edge()
        self.hyperedges           = np.concatenate([self.meancurvature_edge, self.halfedges_coordinate], axis=1)
        self.face_curvature       = self.get_face_curvature()

        self.qHE = QList([Quaternion(self.meancurvature_edge[i, 0], self.halfedges_coordinate[i]) \
                            for i in range(len(self.meancurvature_edge))])
    
    def get_face_he_face(self):
        """Initialize face_he_face.
        """

        face_he_face = np.stack([np.tile(np.arange((self.F.shape[0])), 3), np.arange(len(self.neighbor_face)), self.neighbor_face], axis=1)
        face_he_face = face_he_face[self.neighbor_face > -1, :]
        
        return face_he_face
    
    def get_halfedges_coordinate(self):
        """Initialize halfedges_coordinate.
        """

        V, F = self.V, self.F

        ## e_ij = v_j - v_i
        halfedges_coordinate = -V[F[:, [1, 2, 0]].flatten('F'),:] + V[F[:,[2, 0, 1]].flatten('F'), :]
        halfedges_coordinate = halfedges_coordinate[self.face_he_face[:, 1], :]

        return halfedges_coordinate
    
    def get_meancurvature_edge(self):
        """Initialize meancurvature on edges.
        """

        normals_right = self.normal_face[self.face_he_face[:, 2], :]
        normal_face_extend = self.normal_face[self.face_he_face[:, 0], :]

        edge_norm_squre = np.sum(self.halfedges_coordinate*self.halfedges_coordinate, axis=1, keepdims=True)

        A = normal_face_extend
        B = normal_face_extend - normals_right# - 2*(self.halfedges_coordinate / edge_norm_squre)
        meancurvature_edge = np.sum(A*B, axis=1, keepdims=True)

        A = np.cross(normal_face_extend, normals_right)
        B = self.halfedges_coordinate / edge_norm_squre
        meancurvature_edge = meancurvature_edge / np.sum(A*B, axis=1, keepdims=True)

        plane_index = (np.abs(np.sum(A*B, axis=1, keepdims=True)) < (1e-5)* edge_norm_squre.min())
        meancurvature_edge[plane_index] = 0

        return meancurvature_edge
    
    def get_face_curvature(self):
        """Initialize face curvature.
        """

        ## construct the mean curvature integrated on the faces
        face_curvature = np.zeros((3*self.num_faces, 1))
        face_curvature[self.face_he_face[:, 1], :] = self.meancurvature_edge

        face_curvature = np.sum(np.reshape(face_curvature, (-1, 3), order='F'), axis=1)

        return face_curvature
    
    @property
    def willmore_energy(self):
        """Return the willmore energy of the surface.
        """

        willmore_energy = np.sum(self.face_curvature**2 / self.face_area) / 4

        return willmore_energy