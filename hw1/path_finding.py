from pathlib import Path
from queue import PriorityQueue
from typing import Set, Tuple, List

import numpy as np
import numpy.typing as npt

from hw1.utils import neighbors, plot_GVD, PathPlanMode, distance


def cell_to_GVD_gradient_ascent(
        grid: npt.ArrayLike, GVD: Set[Tuple[int, int]], cell: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Find the shortest path from any cell in the enviroment to a cell on the
    GVD using gradient ascent.
    Args:
        grid (numpy): NxN numpy array representing the world, with obstacles,
        walls, and the distance from each cell to the obstacles.
        GVD (set[tuple]): A set of tuples containing the cells on the GVD.
        cell (tuple): The starting/ending cell of this path.
    Returns:
        list<tuple>: list of tuples of the path.
    """

    path = [cell]
    # TODO: Implement this method
    #
    #
    # while
    # current_cell = path[-1]
    # neighbor_list = neighbors(grid, current_cell[0], path[-1][0])
    # max_distance = 0
    # for neighbor_list:

    # best first search
    current_cell = path[-1]
    while current_cell not in GVD:
        neighbor_list = neighbors(grid, current_cell[0], current_cell[1])
        # current_cell = path[-1]
        max_value = 0
        for neighbor in neighbor_list:
            value = grid[neighbor[0], neighbor[1]]
            if value>max_value:
                max_value=value
                current_cell=neighbor

        path.append(current_cell)

    return path


def cell_to_GVD_a_star(
        grid: npt.ArrayLike, GVD: Set[Tuple[int, int]], cell: Tuple[int, int],
        goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Find the shortest path from any cell in the enviroment to the GVD using
    A* with L2 distance heurstic.
    Args:
        grid (numpy): NxN numpy array representing the world, with obstacles,
        walls, and the distance from each cell to the obstacles.
        GVD (set<tuple>): A set of tuples containing the cells on the GVD.
        cell (tuple): The starting/ending cell of this path.
    Returns:
        list[tuple], dict, list[int]: list of tuples of the path, and the reached 
        dictionary, and the list of frontier sizes. 
    """

    # define a priority queue
    frontier = PriorityQueue()
    frontier.put((0, cell))
    frontier_size = [0]

    # construct a reached table using python dictionary. The key is the (x, y)
    # tuple of the cell position, the value is dictionary with the cell's cost,
    # and cell parent.
    reached = {cell: {"cost": 0, "parent": None}}
    gvd_cell = None
    while not frontier.empty():
        _, vertex = frontier.get()
        if vertex in GVD:
            gvd_cell=vertex
            break
        frontier_size.append(frontier.qsize())
        neighbor_list = neighbors(grid=grid,
                                  i=vertex[0],
                                  j=vertex[1])  # append to the list
        for neighbor in neighbor_list:
            g = reached[vertex]['cost'] + distance(vertex, neighbor)
            if neighbor not in reached or g < reached[neighbor][
                'cost']:  # if g smaller than the real cost to this pixel
                h = distance(neighbor, goal)
                f = g + h
                frontier.put((f, neighbor))
                reached[neighbor] = {"cost": g, "parent": vertex}
                # if neighbor in GVD:
                #     find_gvd = True
                #     gvd_cell = neighbor
                #     break



    # TODO: implement this to use the reached table (back pointers) to find
    # the path once you have reached a cell on the GVD.
    # path = None
    path = [gvd_cell]
    parent = gvd_cell
    while parent != cell:
        parent = reached[parent]["parent"]
        path.append(parent)
    path.reverse()


    return path, reached, frontier_size


def GVD_path(
        grid: npt.ArrayLike,
        GVD: Set[Tuple[int, int]],
        A: Tuple[int, int],
        B: Tuple[int, int],
        mode: PathPlanMode
) -> List[Tuple[int, int]]:
    """Find the shortest path between two points on the GVD using
    Breadth-First-Search
    Args:
        grid (numpy): NxN numpy array representing the world, with obstacles,
        walls, and the distance from each cell to the obstacles.
        A (tuple): The starting cell of the path.
        B (tuple): The ending cell of the path.
    Returns:
        list[tuple], dict, list[int]: return the path, pointers, and frontier 
        size array. 
    """

    # the set of cells on the GVD
    global neighbor_list
    GVD = set(GVD)

    # the set of visited cells
    closed = set([])

    # the set of cells on the current frontier
    frontier = [A]

    # back pointers to find the path once reached the goal B. The keys
    # should both be tuples of cell positions (x, y)
    pointers = {}

    # the length of the frontier array, update this variable at each step. 
    frontier_size = [0]

    # dfs:

    while len(frontier) > 0:
        vertex = None
        if mode == PathPlanMode.BFS:
            vertex = frontier.pop(0)
        if mode == PathPlanMode.DFS:
            vertex = frontier.pop(-1)

        if vertex == B:
            break

        frontier_size.append(len(frontier))
        # print(frontier_size)

        neighbor_list = neighbors(grid=grid,
                                  i=vertex[0],
                                  j=vertex[1])


        for neighbor in neighbor_list:
            if neighbor in GVD:
                if neighbor not in pointers and neighbor not in frontier:
                    frontier.append(neighbor)
                    pointers[neighbor] = vertex


    # triv back
    path = [B]
    parent = B
    while parent != A:
        parent = pointers[parent]
        path.append(parent)
    path.reverse()

    return path, pointers, frontier_size


def compute_path(
        grid,
        GVD: set[tuple],
        start: tuple,
        goal: tuple,
        outmode: PathPlanMode = PathPlanMode.GRAD,
        inmode: PathPlanMode = PathPlanMode.DFS):
    """ Compute the path on the grid from start to goal using the methods
    implemented in this file. 
    Returns:
        list: a list of tuples represent the planned path. 
    """

    if outmode == PathPlanMode.GRAD:
        start_path = cell_to_GVD_gradient_ascent(grid, GVD, start)
        end_path = list(reversed(cell_to_GVD_gradient_ascent(grid, GVD, goal)))
    else:
        start_path = cell_to_GVD_a_star(grid, GVD, start, goal)[0]
        end_path = list(reversed(cell_to_GVD_a_star(grid, GVD, goal, start)[0]))
    mid_path, reached, frontier_size = GVD_path(
        grid, GVD, start_path[-1], end_path[0], inmode)
    return start_path + mid_path[1:-1] + end_path


def test_world(
        world_id,
        start,
        goal,
        outmode: PathPlanMode = PathPlanMode.GRAD,
        inmode: PathPlanMode = PathPlanMode.DFS,
        world_dir="worlds"):
    print(f"Testing world {world_id} with modes {inmode} and {outmode}")
    grid = np.load(f"{world_dir}/world_{world_id}.npy")
    GVD = set([tuple(cell) for cell in np.load(
        f"{world_dir}/world_{world_id}_gvd.npy")])
    path = compute_path(grid, GVD, start, goal, outmode=outmode, inmode=inmode)
    print(f"Path length: {len(path)} steps")
    plot_GVD(grid, world_id, GVD, path)
