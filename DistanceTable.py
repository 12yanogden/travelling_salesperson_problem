from typing import Optional, Any

import math
from TSPClasses import TSPSolution


class DistanceTable:
    def __init__(self, cities: list = None):
        self.cities = cities
        self.distances = self.init_distances(cities) if cities is not None else None
        self.route = []
        self.unvisited = [*range(len(cities))]
        self.lower_bound = 0

    def init_distances(self, cities: list) -> list:
        distances = []

        for i in range(len(cities)):
            distances.append([])

            for j in range(len(cities)):
                distances[i].append(cities[i].cost_to(cities[j]))

        return distances

    def set_start_city(self, start_city_index: int) -> None:
        self.unvisited.remove(start_city_index)
        self.route.append(start_city_index)

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                  Visit a city                                 #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    def visit(self, to_index: int) -> None:
        from_index = self.route[-1]
        distance = self.distances[from_index][to_index]

        self.infinity_row(from_index)
        self.infinity_column(to_index)
        self.infinity_inverse(from_index, to_index)

        self.route.append(to_index)
        self.unvisited.remove(to_index)

        self.lower_bound += distance

    def infinity_row(self, row: int) -> None:
        row = self.distances[row]

        for i in range(len(row)):
            if row[i] != math.inf:
                row[i] = math.inf

    def infinity_column(self, column: int) -> None:
        for row in self.distances:
            if row[column] != math.inf:
                row[column] = math.inf

    def infinity_inverse(self, from_index: int, to_index: int) -> None:
        self.distances[to_index][from_index] = math.inf

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                  Reduce Table                                 #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    def reduce(self) -> None:
        self.reduce_rows()
        self.reduce_columns()

    def reduce_rows(self) -> None:
        for row in self.distances:
            min_distance = math.inf
            found_zero = False

            # Find the minimum distance in the row
            for distance in row:
                if distance == 0:
                    found_zero = True
                    break
                elif distance < min_distance:
                    min_distance = distance

            if found_zero:
                continue

            # Subtract the minimum distance from each distance in the row
            if min_distance < math.inf:
                for i in range(len(row)):
                    row[i] -= min_distance

                self.lower_bound += min_distance

    def reduce_columns(self) -> None:
        for j in range(len(self.cities)):
            min_distance = math.inf
            found_zero = False

            # Find the minimum distance in the column
            for i in range(len(self.cities)):
                distance = self.distances[i][j]

                if distance == 0:
                    found_zero = True
                    break
                elif distance < min_distance:
                    min_distance = distance

            if found_zero:
                continue

            # Subtract the minimum distance from each distance in the column
            if min_distance < math.inf:
                for i in range(len(self.distances)):
                    self.distances[i][j] -= min_distance

                self.lower_bound += min_distance

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                     Greedy                                    #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    def get_nearest_city(self) -> Optional[int]:
        nearest_city_index = None

        if len(self.route) < len(self.cities):
            row = self.distances[self.route[-1]]
            min_distance = math.inf

            for i in range(len(row)):
                if i != self.route[0] and row[i] < min_distance:
                    nearest_city_index = i
                    min_distance = row[i]

        return nearest_city_index

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                    Solution                                   #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    def has_solution(self) -> bool:
        return len(self.route) == len(self.cities) and self.distances[self.route[-1]][self.route[0]] < math.inf

    def complete_cycle(self) -> None:
        self.lower_bound += self.distances[self.route[-1]][self.route[0]]

    def to_solution(self) -> TSPSolution:
        route_cities = []

        # Convert route indexes to cities
        for index in self.route:
            route_cities.append(self.cities[index])

        return TSPSolution(route_cities)

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                      Heap                                     #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    def __lt__(self, other) -> bool:
        return self.lower_bound < other.lower_bound

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                     Debug                                     #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    def __str__(self) -> str:
        string = ""

        string += "\t"

        for city in self.cities:
            string += str(city._index) + "\t"

        string += "\n"

        for i in range(len(self.distances)):
            string += str(self.cities[i]._index) + "\t"

            for distance in self.distances[i]:
                string += str(distance) + "\t"

            string += "\n"

        string += "lower bound: " + str(self.lower_bound)

        return string.expandtabs(8)

