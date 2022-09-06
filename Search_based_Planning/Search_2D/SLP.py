"""
SLP
@author: yinliang chen
"""

import os
import sys
import math
import heapq
import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env

from Astar import AStar as AS

class SLP:
    """SLP is a hybrid path planning algorithm
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.line_slope = math.inf
        self.heuristic_type = heuristic_type

        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come
        self.astar_sg = [] # astar start and goal ele:[start,goal]

    def LINEAR_PATH_CALCULATOR(self):
        """
        SLP LINEAR_PATH_CALCULATOR.
        :return: path, intersection_points
        """
        path = [self.s_start]
        inter = []
        # exclude vertical line case
        if self.s_start[0] - self.s_goal[0] != 0:
            self.line_slope = (self.s_start[1] - self.s_goal[1]) / (self.s_start[0] - self.s_goal[0])
            dx_delta = math.copysign(1, self.s_goal[0] - self.s_start[0])
            path_dx = 0
            while True:
                if self.s_start[0] + path_dx == self.s_goal[0]:
                    break
                path_dx += dx_delta
                point = (self.s_start[0] + path_dx, math.ceil(self.s_start[1] + self.line_slope * path_dx))
                if point in self.obs:
                    inter.append(point)
                    self.astar_sg.append(
                        [(self.s_start[0] + path_dx - 1, math.ceil(self.s_start[1] + self.line_slope * (path_dx - 1))),
                         (self.s_start[0] + path_dx + 1, math.ceil(self.s_start[1] + self.line_slope * (path_dx + 1)))])
                else:
                    path.append(point)


        else:
            path_dy = 0
            dy_delta = math.copysign(1, self.s_goal[1] - self.s_start[1])
            while True:
                if self.s_start[1] + path_dy == self.s_goal[1]:
                    break
                path_dy += dy_delta
                point = (self.s_start[0], self.s_start[1] + path_dy)
                if point in self.obs:
                    inter.append(point)
                else:
                    path.append(point)
        path.append(self.s_goal)

        return path, inter

    def BASIS_ALGORITHM_PLANNER(self, path):
        """
        USING A*
        """
        slp_path = np.array(path)
        for sg in self.astar_sg:
            astar = AS(sg[0], sg[1], "euclidean")
            a_path, visited = astar.searching()
            a_path = np.array(a_path)
            a_path = a_path[::-1][1:-1] # reverse and remove start and goal
            insert_place = np.where(slp_path == sg[0])[0]
            insert_place = stats.mode(insert_place)[0][0]  # the most element
            slp_path_a = slp_path[0:insert_place]
            slp_path_b = slp_path[insert_place+1:]
            slp_path = np.concatenate((slp_path_a,a_path))
            slp_path = np.concatenate((slp_path,slp_path_b)) # insert astar path
        return slp_path.tolist()

    def PATH_LINEARIZER(self, path):
        assert path[-1] != self.s_goal
        assert path[0] != self.s_start

        if len(path) < 4:
            return path

        linearized_path = []

        i = len(path) - 1  # parse adversely
        j = i - 1
        pre = j
        linearized_path.append(path[i])


        while True:
            # if j == 0:
            #     if self.is_collision(tuple(path[i]), tuple(path[j])):
            #         linearized_path.append(path[pre])
            #     break
            if self.is_collision(tuple(path[i]), tuple(path[j])):
                print("{} and {} collide".format(path[i],path[j]))
                linearized_path.append(path[pre])
                if j == 0:
                    break
                else:
                    i = pre
                    j = pre-1
                    pre = j
            else:
                if j == 0:
                    break
                pre = j
                j -= 1


        linearized_path.append(path[0])
        linearized_path = np.array(linearized_path)
        print(path)
        print(linearized_path[::-1].tolist())
        return linearized_path[::-1].tolist()


    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """
        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0]-s_end[0] != 0:
            slope = (s_start[1]-s_end[1])/(s_start[0]-s_end[0])
            path_dx = 0
            dx_delta = math.copysign(1, s_end[0]-s_start[0])

            while True:
                if s_start[0] + path_dx == s_end[0]:
                    break
                path_dx += dx_delta
                ceil_point = (s_start[0] + path_dx, math.ceil(s_start[1] + slope*path_dx))
                floor_point = (s_start[0] + path_dx, math.floor(s_start[1] + slope*path_dx))
                if ceil_point in self.obs or floor_point in self.obs:
                    print("collided at {} or {}".format(ceil_point,floor_point))
                    return True
        return False

def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    slp = SLP(s_start, s_goal, "euclidean")
    # print(slp.is_collision(s_start,s_start))
    plot = plotting.Plotting(s_start, s_goal)
    path, visited = slp.LINEAR_PATH_CALCULATOR()
    slp_path = slp.BASIS_ALGORITHM_PLANNER(path)

    linearized_path = slp.PATH_LINEARIZER(slp_path)
    pre_path_len = len(linearized_path)
    while True:
        linearized_path = slp.PATH_LINEARIZER(linearized_path)
        cur_path_len = len(linearized_path)
        if pre_path_len == cur_path_len:
            break
        else:
            pre_path_len = cur_path_len

    plot.animation(slp_path, visited, "SLP")  # animation
    plot.animation(linearized_path, visited, "SLP")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


if __name__ == '__main__':
    main()
