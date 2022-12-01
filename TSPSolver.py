#!/usr/bin/python3

from which_pyqt import PYQT_VER
from DistanceTable import DistanceTable
import heapq
import copy

if PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setup_with_scenario(self, scenario):
        self._scenario = scenario

    # Untouched from provided code
    def default_random_tour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.get_cities()
        n_cities = len(cities)
        found_tour = False
        count = 0
        bssf = None

        start_time = time.time()

        while not found_tour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(n_cities)
            route = []
            # Now build the route using the random permutation
            for i in range(n_cities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                found_tour = True
        end_time = time.time()
        results['cost'] = bssf.cost if found_tour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                  Optimization                                 #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    # Included for optimization. This type of function would typically be found in the DistanceTable constructor.
    # It's removal means that I only have to call len(cities) once per algorithm. It also means that an integer is
    # stored where a pointer would have been stored in the DistanceTable class. The change resulted in a surprising 50%
    # speedup.
    def initialize_dist_table(self) -> DistanceTable:
        cities = self._scenario.get_cities()
        n_cities = len(cities)
        distances = []

        for i in range(n_cities):
            distances.append([])

            for j in range(n_cities):
                distances[i].append(cities[i].cost_to(cities[j]))

        return DistanceTable(n_cities, distances)

    # Also included for optimization, as described above.
    # Time: O(n), Space: O(n)
    def dist_table_to_solution(self, dist_table: DistanceTable) -> TSPSolution:
        cities = self._scenario.get_cities()
        route_cities = []

        # Convert route indexes to cities
        for index in dist_table.route:
            route_cities.append(cities[index])

        return TSPSolution(route_cities)

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                     Greedy                                    #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    # A helper function for the greedy algorithm
    # Time: O(n^2), Space: O(n)
    def solve_greedy(self, start_city_index: int, dist_table: DistanceTable) -> TSPSolution:
        dist_table.set_start_city(start_city_index)
        next_city_index = dist_table.get_nearest_city() # Time: O(n), Space: O(1)
        solution = None

        # Time: O(n^2), Space: O(1)
        while next_city_index is not None:
            dist_table.visit(next_city_index)
            next_city_index = dist_table.get_nearest_city()

        # Time: O(n), Space: O(n)
        if dist_table.has_solution():
            solution = self.dist_table_to_solution(dist_table)

        return solution

    # Time: O(n^3), Space: O(n^2)
    def greedy(self, time_allowance=60.0):
        dist_table = self.initialize_dist_table()
        solution_count = 0
        bssf = None
        results = {}

        start_time = time.time()

        # Time: O(n^3), Space: O(n^2)
        for city_index in range(dist_table.n_cities):       # Iterates O(n) times
            if time.time() - start_time > time_allowance:
                break

            # Time: O(n^2), Space: O(n^2)
            # "Deep copies" dist_table of O(n^2) size
            current_solution = self.solve_greedy(city_index, copy.deepcopy(dist_table))

            if current_solution is not None:
                solution_count += 1

                if bssf is None or current_solution.cost < bssf.cost:
                    bssf = current_solution

        end_time = time.time()

        results['cost'] = bssf.cost if solution_count > 0 else math.inf
        results['time'] = end_time - start_time
        results['count'] = solution_count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None

        return results

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                Branch and Bound                               #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    # Time: O(n), Space: O(n)
    def prune(self, priority_queue, upper_bound):
        pruned_queue = []

        for i in range(len(priority_queue)):
            if priority_queue[i].lower_bound < upper_bound:
                pruned_queue.append(priority_queue[i])

        return pruned_queue, len(priority_queue) - len(pruned_queue)

    def branch_and_bound(self, time_allowance=60.0):
        dist_table = self.initialize_dist_table()
        priority_queue = []
        bssf = None
        solution_count = 0
        max_queue_size = 0
        pruned_count = 0
        total_states = 0
        results = {}

        # Find initial upper bound and solution
        greedy_results = self.greedy(time_allowance)
        upper_bound = greedy_results['cost']
        bssf = greedy_results['soln']

        start_time = time.time()

        # Find initial lower bound
        dist_table.set_start_city(0)
        dist_table.reduce()
        heapq.heappush(priority_queue, dist_table)

        # Solve
        # Time: O(n^2 * b^n), Space: O(n^2 * b^n)
        while len(priority_queue) > 0 and time.time() - start_time < time_allowance:
            parent_table = heapq.heappop(priority_queue)

            # Time : O(n^3), Space: O(n^3)
            for city_index in parent_table.unvisited:
                # Time: O(n^2), Space: O(n^2)
                branch_table = copy.deepcopy(parent_table)
                total_states += 1

                # Time: O(n), Space: O(1)
                branch_table.visit(city_index)

                # Time: O(n^2), Space: O(1)
                branch_table.reduce()

                # print(branch_table)

                # Evaluates solution
                if branch_table.has_solution():
                    branch_table.finalize_lower_bound()

                    # Saves solution, else prunes
                    if branch_table.lower_bound < upper_bound:
                        upper_bound = branch_table.lower_bound
                        bssf = self.dist_table_to_solution(branch_table)
                        solution_count += 1
                        self.prune(priority_queue, upper_bound)
                    else:
                        pruned_count += 1

                # Evaluates branch
                elif branch_table.lower_bound < upper_bound:
                    heapq.heappush(priority_queue, branch_table)
                    max_queue_size = max(max_queue_size, len(priority_queue))

                # Prunes branch
                else:
                    pruned_count += 1

        end_time = time.time()

        results['cost'] = bssf.cost if solution_count > 0 else math.inf
        results['time'] = end_time - start_time
        results['count'] = solution_count
        results['soln'] = bssf
        results['max'] = max_queue_size
        results['total'] = total_states
        results['pruned'] = pruned_count

        return results

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                     Fancy                                     #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    # Will implement in Project 6
    def fancy(self, time_allowance=60.0):
        pass
