from typing import Optional, Any

import math


class DistanceTable:
    def __init__(self, n_cities: int = 0, distances: list = None):
        self.n_cities = n_cities
        self.distances = distances
        self.route = []
        self.unvisited = [*range(n_cities)]
        self.lower_bound = 0

    # Time: O(n), Space: O(1)
    def set_start_city(self, start_city_index: int) -> None:
        self.unvisited.remove(start_city_index)
        self.route.append(start_city_index)

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                  Visit a city                                 #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    # Time: O(n), Space: O(1)
    def visit(self, to_index: int) -> None:
        from_index = self.route[-1]
        distance = self.distances[from_index][to_index]

        # Updates distance table
        self.infinity_row(from_index)
        self.infinity_column(to_index)
        self.infinity_inverse(from_index, to_index)

        # Transfers unvisited to route
        self.unvisited.remove(to_index)
        self.route.append(to_index)

        # Updates lower bound
        self.lower_bound += distance

    # Time: O(n), Space: O(1)
    def infinity_row(self, row: int) -> None:
        row = self.distances[row]

        for i in range(len(row)):
            if row[i] != math.inf:
                row[i] = math.inf

    # Time: O(n), Space: O(1)
    def infinity_column(self, column: int) -> None:
        for row in self.distances:
            if row[column] != math.inf:
                row[column] = math.inf

    # Time: O(1), Space: O(1)
    def infinity_inverse(self, from_index: int, to_index: int) -> None:
        self.distances[to_index][from_index] = math.inf

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                  Reduce Table                                 #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    # Time: O(n^2), Space: O(1)
    def reduce(self) -> None:
        self.reduce_rows()
        self.reduce_columns()

    # Time: O(n^2), Space: O(1)
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

            # Skips row if a zero was found for efficiency
            if found_zero:
                continue

            # Subtract the minimum distance from each distance in the row
            if min_distance < math.inf:
                for i in range(len(row)):
                    row[i] -= min_distance

                self.lower_bound += min_distance

    # Time: O(n^2), Space: O(1)
    def reduce_columns(self) -> None:
        for j in range(self.n_cities):
            min_distance = math.inf
            found_zero = False

            # Find the minimum distance in the column
            for i in range(self.n_cities):
                distance = self.distances[i][j]

                if distance == 0:
                    found_zero = True
                    break
                elif distance < min_distance:
                    min_distance = distance

            # Skips column if a zero was found for efficiency
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
    # Time: O(n), Space: O(1)
    def get_nearest_city(self) -> Optional[int]:
        nearest_city_index = None

        if len(self.route) < self.n_cities:
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
    # Time: O(1), Space: O(1)
    def has_solution(self) -> bool:
        return len(self.route) == self.n_cities and self.distances[self.route[-1]][self.route[0]] < math.inf

    # Time: O(1), Space: O(1)
    def finalize_lower_bound(self) -> None:
        self.lower_bound += self.distances[self.route[-1]][self.route[0]]

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                      Heap                                     #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    # Time: O(1), Space: O(1)
    def __lt__(self, other) -> bool:
        return self.lower_bound / len(self.route) < other.lower_bound / len(other.route)

    # ----------------------------------------------------------------------------- #
    #                                                                               #
    #                                     Debug                                     #
    #                                                                               #
    # ----------------------------------------------------------------------------- #
    # Time: O(n^2), Space: O(n^2)
    def __str__(self) -> str:
        string = ""

        string += "\t"

        for city_index in range(self.n_cities):
            string += str(city_index) + "\t"

        string += "\n"

        for i in range(len(self.distances)):
            string += str(i) + "\t"

            for distance in self.distances[i]:
                string += str(distance) + "\t"

            string += "\n"

        string += "lower bound: " + str(self.lower_bound)

        return string.expandtabs(8)

