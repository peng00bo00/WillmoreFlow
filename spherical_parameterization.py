import os
import igl
import numpy as np

import time

from willmore import WillmoreFlow, Mesh

MESH_ROOT = "./meshes"
MESH = "spot.obj"
SAVE_PATH = "./parameterization"

STEP = 10
TAU  = 0.3
CUPY = True

V, F = igl.read_triangle_mesh(os.path.join(MESH_ROOT, MESH))

begin = time.perf_counter()

for i in range(STEP):
    ## normalize vertices for numerical stability
    V = V - np.mean(V, axis=0)
    V = V / V.max()

    mesh = Mesh(V, F)
    print(f"\nStep {i}, Willmore Energy: {mesh.willmore_energy:.3f}")

    V = WillmoreFlow(mesh, TAU, CUPY)

    # igl.write_triangle_mesh(os.path.join(SAVE_PATH, f"spot_sphere_{str(i).zfill(2)}.obj"), V, F)

end = time.perf_counter() 
print(f"Parameterization finished! Time Cost: {(end-begin):.3f} s")