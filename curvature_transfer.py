import os
import igl
import numpy as np

import time

from willmore import curvature_transfer_step, Mesh

MESH_ROOT = "./meshes"
MESH      = "spot.obj"
SPHERE    = "spot_sphere.obj"
SAVE_PATH = "./parameterization"

STEP = 15
TAU  = 0.3
CUPY = True


## target mesh
V, F = igl.read_triangle_mesh(os.path.join(MESH_ROOT, MESH))
target = Mesh(V, F)
target = target.face_curvature / np.sqrt(target.face_area)

## spherical parameterization
V, F = igl.read_triangle_mesh(os.path.join(SAVE_PATH, SPHERE))


begin = time.perf_counter()

for i in range(STEP):
    ## normalize vertices for numerical stability
    V = V - np.mean(V, axis=0)
    V = V / V.max()

    V = curvature_transfer_step(V, F, target, TAU, CUPY)
    print()
    # igl.write_triangle_mesh(os.path.join(SAVE_PATH, f"spot_recon_{str(i).zfill(2)}.obj"), V, F)

V = V - np.mean(V, axis=0)
V = V / V.max()

end = time.perf_counter()
print(f"Parameterization finished! Time Cost: {(end-begin):.3f} s")

igl.write_triangle_mesh(os.path.join(SAVE_PATH, f"spot_recon.obj"), V, F)