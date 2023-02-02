import argparse
import os

from hw1.utils import PathPlanMode, plot_GVD, neighbors
import hw1.path_finding as pf
import numpy as np
import matplotlib.pyplot as plt

# directory_name = '../worlds'

# for filename in os.listdir(directory_name):
#     f = os.path.join(directory_name, filename)
#     world = np.load(file=f)
#     plt.imshow(world)
#     plt.show()



#
# test = pf.GVD_path()


world_file = '../worlds/world_1.npy'
gvd_path_file = '../worlds/world_1_gvd.npy'
world = np.load(world_file)
gvd_path = np.load(gvd_path_file)
plot_GVD(world, 1)
gvd_path_tuple = set([tuple(cell) for cell in gvd_path])
gvd_path = [tuple(cell) for cell in gvd_path]
a = neighbors(world, 10, 10)
mid_path, reached, frontier_size = pf.GVD_path(world, gvd_path_tuple, (15, 30), (91, 7), PathPlanMode.DFS)
plot_GVD(world, 1, gvd_path, mid_path)
mid_path, reached, frontier_size = pf.GVD_path(world, gvd_path_tuple, (15, 30), (91, 7), PathPlanMode.BFS)
plot_GVD(world, 1, gvd_path, mid_path)


# test grad access
path = pf.cell_to_GVD_gradient_ascent(world, gvd_path_tuple, (10,20))
plot_GVD(world, 1, gvd_path, path)

path = pf.cell_to_GVD_gradient_ascent(world, gvd_path_tuple, (40,20))
plot_GVD(world, 1, gvd_path, path)


