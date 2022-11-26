#!/usr/bin/python3

import math
import numpy as np
import random
import time


class TSPSolution:
    def __init__(self, list_of_cities):
        self.route = list_of_cities
        self.cost = self._cost_of_route()

    # print([c._index for c in listOfCities])

    def _cost_of_route(self):
        cost = 0
        last = self.route[0]

        for city in self.route[1:]:
            cost += last.cost_to(city)
            last = city

        cost += self.route[-1].cost_to(self.route[0])

        return cost

    def enumerate_edges(self):
        elist = []
        c1 = self.route[0]

        for c2 in self.route[1:]:
            dist = c1.cost_to(c2)
            if dist == np.inf:
                return None
            elist.append((c1, c2, int(math.ceil(dist))))
            c1 = c2

        dist = self.route[-1].cost_to(self.route[0])

        if dist == np.inf:
            return None
        elist.append((self.route[-1], self.route[0], int(math.ceil(dist))))

        return elist


def name_for_int(num):
    if num == 0:
        return ''
    elif num <= 26:
        return chr(ord('A') + num - 1)
    else:
        return name_for_int((num - 1) // 26) + name_for_int((num - 1) % 26 + 1)


class Scenario:
    HARD_MODE_FRACTION_TO_REMOVE = 0.20  # Remove 20% of the edges

    def __init__(self, city_locations, difficulty, rand_seed):
        self._difficulty = difficulty

        if difficulty == "Normal" or difficulty == "Hard":
            self._cities = [City(pt.x(), pt.y(),
                                 random.uniform(0.0, 1.0)
                                 ) for pt in city_locations]
        elif difficulty == "Hard (Deterministic)":
            random.seed(rand_seed)
            self._cities = [City(pt.x(), pt.y(),
                                 random.uniform(0.0, 1.0)
                                 ) for pt in city_locations]
        else:
            self._cities = [City(pt.x(), pt.y()) for pt in city_locations]

        num = 0
        for city in self._cities:
            # if difficulty == "Hard":
            city.set_scenario(self)
            city.set_index_and_name(num, name_for_int(num + 1))
            num += 1

        # Assume all edges exists except self-edges
        ncities = len(self._cities)
        self._edge_exists = (np.ones((ncities, ncities)) - np.diag(np.ones(ncities))) > 0

        if difficulty == "Hard":
            self.thin_edges()
        elif difficulty == "Hard (Deterministic)":
            self.thin_edges(deterministic=True)

    def get_cities(self):
        return self._cities

    def randperm(self, n):  # isn't there a numpy function that does this and even gets called in Solver?
        perm = np.arange(n)
        for i in range(n):
            rand_index = random.randint(i, n - 1)
            save = perm[i]
            perm[i] = perm[rand_index]
            perm[rand_index] = save
        return perm

    def thin_edges(self, deterministic=False):
        n_cities = len(self._cities)
        edge_count = n_cities * (n_cities - 1)  # can't have self-edge
        num_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE * edge_count)

        can_delete = self._edge_exists.copy()

        # Set aside a route to ensure at least one tour exists
        route_keep = np.random.permutation(n_cities)
        if deterministic:
            route_keep = self.randperm(n_cities)
        for i in range(n_cities):
            can_delete[route_keep[i], route_keep[(i + 1) % n_cities]] = False

        # Now remove edges until 
        while num_to_remove > 0:
            if deterministic:
                src = random.randint(0, n_cities - 1)
                dst = random.randint(0, n_cities - 1)
            else:
                src = np.random.randint(n_cities)
                dst = np.random.randint(n_cities)
            if self._edge_exists[src, dst] and can_delete[src, dst]:
                self._edge_exists[src, dst] = False
                num_to_remove -= 1


class City:
    def __init__(self, x, y, elevation=0.0):
        self._x = x
        self._y = y
        self._elevation = elevation
        self._scenario = None
        self._index = -1
        self._name = None

    def set_index_and_name(self, index, name):
        self._index = index
        self._name = name

    def set_scenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        How much does it cost to get from this city to the destination?
        Note that this is an asymmetric cost function.
        
        In advanced mode, it returns infinity when there is no connection.
        </summary> '''

    MAP_SCALE = 1000.0

    def cost_to(self, other_city):

        assert (type(other_city) == City)

        # In hard mode, remove edges; this slows down the calculation...
        # Use this in all difficulties, it ensures INF for self-edge
        if not self._scenario._edge_exists[self._index, other_city._index]:
            return np.inf

        # Euclidean Distance
        cost = math.sqrt((other_city._x - self._x) ** 2 +
                         (other_city._y - self._y) ** 2)

        # For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
        if not self._scenario._difficulty == 'Easy':
            cost += (other_city._elevation - self._elevation)
            if cost < 0.0:
                cost = 0.0  # Shouldn't it cost something to go downhill, no matter how steep??????

        return int(math.ceil(cost * self.MAP_SCALE))
