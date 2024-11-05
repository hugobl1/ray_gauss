import numpy as np
from ply import write_ply, read_ply

NB_MIN_TRACK_IMAGES = 4 #number of images where the point is visible

file = open("dataset-colmap\\reconstruction-txt\\points3D.txt", "r")

file.readline()
file.readline()

data_xyz_ply = np.array([[0,0,0]], dtype=float)
data_normals_ply = np.array([[0.0,0,0]], dtype=float)
data_rgb_ply = np.array([[0,0,0]], dtype=np.uint8)

while True:
    content=file.readline()
    if not content:
        break
    tab_line = content.split()
    if len(tab_line) >= 8+2*NB_MIN_TRACK_IMAGES:
        data_xyz_ply = np.append(data_xyz_ply, [np.array(tab_line[1:4], dtype = float)], axis = 0)
        data_normals_ply = np.append(data_normals_ply, [[0.0, 0.0, 0.0]], axis = 0)
        data_rgb_ply = np.append(data_rgb_ply, [np.array(tab_line[4:7], dtype = np.uint8)], axis = 0)


print("Number points as sparse point cloud: ", len(data_xyz_ply))


write_ply("dataset-colmap\\sparse\\0\\points3D.ply", [data_xyz_ply, data_normals_ply, data_rgb_ply], ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue'])
