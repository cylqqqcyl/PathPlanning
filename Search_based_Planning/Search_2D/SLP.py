"""
A_star 2D
@author: huiming zhou
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
        if self.s_start[0]-self.s_goal[0] != 0:
            self.line_slope = (self.s_start[1]-self.s_goal[1])/(self.s_start[0]-self.s_goal[0])
            path_dx = 0
            while True:
                if self.s_start[0] + path_dx == self.s_goal[0]:
                    break
                path_dx += 1
                point = (self.s_start[0] + path_dx, math.ceil(self.s_start[1] + self.line_slope*path_dx))
                if point in self.obs:
                    inter.append(point)
                    self.astar_sg .append([(self.s_start[0] + path_dx-1, math.ceil(self.s_start[1] + self.line_slope*(path_dx-1))),
                                     (self.s_start[0] + path_dx+1, math.ceil(self.s_start[1] + self.line_slope*(path_dx+1)))])
                else:
                    path.append(point)

        else:
            path_dy = 0
            while True:
                if self.s_start[1] + path_dy == self.s_goal[1]:
                    break
                path_dy += 1
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
            insert_place = stats.mode(insert_place)[0][0]
            slp_path_a = slp_path[0:insert_place]
            slp_path_b = slp_path[insert_place+1:]
            slp_path = np.concatenate((slp_path_a,a_path))
            slp_path = np.concatenate((slp_path,slp_path_b)) # insert astar path
        return slp_path.tolist()

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    slp = SLP(s_start, s_goal, "euclidean")
    plot = plotting.Plotting(s_start, s_goal)

    path, visited = slp.LINEAR_PATH_CALCULATOR()
    slp_path = slp.BASIS_ALGORITHM_PLANNER(path)
    plot.animation(slp_path, visited, "SLP")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


if __name__ == '__main__':
    main()
