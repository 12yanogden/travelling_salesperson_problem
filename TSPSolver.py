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

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

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

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''
    def solve_greedy(self, start_city_index: int, dist_table: DistanceTable) -> TSPSolution:
        dist_table.set_start_city(start_city_index)
        next_city_index = dist_table.get_nearest_city()
        solution = None

        while next_city_index is not None:
            dist_table.visit(next_city_index)
            next_city_index = dist_table.get_nearest_city()

        if dist_table.has_solution():
            solution = dist_table.to_solution()

        return solution

    def greedy(self, time_allowance=60.0):
        dist_table = DistanceTable(self._scenario.get_cities())
        solution_count = 0
        bssf = None
        results = {}

        start_time = time.time()

        for city_index in range(len(dist_table.cities)):
            if time.time() - start_time > time_allowance:
                break

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

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    def prune(self, priority_queue, upper_bound):
        pruned_queue = []

        # print("Priority Queue before pruning")
        # print("length:", len(priority_queue))
        # for dist_table in priority_queue:
        #     print(str(dist_table.lower_bound) + " ", end="")

        for i in range(len(priority_queue)):
            if priority_queue[i].lower_bound < upper_bound:
                pruned_queue.append(priority_queue[i])

        # print()
        # print()

        # print("Priority Queue after pruning")
        # print("length:", len(pruned_queue))
        # for dist_table in pruned_queue:
        #     print(str(dist_table.lower_bound) + " ", end="")

        return pruned_queue, len(priority_queue) - len(pruned_queue)

    def branch_and_bound(self, time_allowance=60.0):
        dist_table = DistanceTable(self._scenario.get_cities())
        priority_queue = []
        bssf = None
        solution_count = 0
        max_queue_size = 0
        pruned_count = 0
        total_states = 0
        results = {}

        # Find upper_bound
        greedy_results = self.greedy(time_allowance)
        upper_bound = greedy_results['cost']
        bssf = greedy_results['soln']

        start_time = time.time()

        # Find lower_bound
        dist_table.set_start_city(0)
        dist_table.reduce()
        heapq.heappush(priority_queue, dist_table)

        # Solve
        while len(priority_queue) > 0 and time.time() - start_time < time_allowance:
            parent_table = heapq.heappop(priority_queue)

            # print("Evaluating " + str(parent_table.route[-1]))
            # print(parent_table)

            for city_index in parent_table.unvisited:
                branch_table = copy.deepcopy(parent_table)
                # print(str(branch_table.route[-1]) + " -> " + str(city_index))
                total_states += 1

                branch_table.visit(city_index)
                branch_table.reduce()

                # print(branch_table)

                if branch_table.has_solution():
                    solution_count += 1
                    branch_table.complete_cycle()

                    if branch_table.lower_bound < upper_bound:
                        upper_bound = branch_table.lower_bound
                        bssf = branch_table.to_solution()
                        self.prune(priority_queue, upper_bound)
                    else:
                        pruned_count += 1

                elif branch_table.lower_bound < upper_bound:
                    heapq.heappush(priority_queue, branch_table)
                    max_queue_size = max(max_queue_size, len(priority_queue))
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

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        pass
