import glob
import os
from typing import List

import highspy
# Set PMIP_HIGHS_LIBRARY automatically to make mip find the HiGHS solver library.
# This is more robust than hardcoding the filename.
try:
    highs_lib_dir = os.path.dirname(highspy.__file__)
    so_file = glob.glob(os.path.join(highs_lib_dir, "_core.*.so"))[0]
    os.environ["PMIP_HIGHS_LIBRARY"] = so_file
except IndexError:
    # This may happen in some environments, but we'll let mip try to find it.
    print("Warning: Could not automatically find the HiGHS shared library.")
import time

from mip import BINARY, Model, OptimizationStatus, minimize, xsum
from src.instance import VrpInstance
from src.solution import VrpSolution


class VrpSolver:
    """
    A Mixed-Integer Programming (MIP) model for the Vehicle Routing Problem (VRP).

    This class formulates the VRP as a MIP model and uses an iterative approach
    to handle subtour elimination constraints. The model aims to find the set of
    routes with the minimum total distance for a fleet of vehicles to service a
    set of locations, starting and ending at a depot.
    """

    def __init__(
        self,
        instance: VrpInstance,
    ):
        """Initializes the VrpSolver.

        Args:
            instance: The VrpInstance object containing the problem data.
        """
        self.m = Model(solver_name="HIGHS")
        self.instance = instance

        # For now, we assume a single depot. This can be extended later.
        if len(instance.depots) != 1:
            raise NotImplementedError("Solver currently supports only one depot.")
        self.depot = instance.depots[0]
        self.depot_index = instance.distance_matrix.get_index(self.depot)

        # Locations to visit
        num_locations = len(instance.distance_matrix.locations)
        self.V = [i for i in range(num_locations)]
        self.V_excl_depot = [
            v for v in self.V if v != self.depot_index
        ]

    def build(self):
        """Builds the VRP model by defining variables, constraints, and the objective.

        This method sets up the initial MIP formulation without subtour elimination
        constraints. These are added lazily during the solving process.

        Variables:
            - x[i, j]: A binary variable that is 1 if a vehicle travels from location
              `i` to `j`, and 0 otherwise.

        Constraints:
            - Each location (except the depot) must be entered exactly once.
            - Each location (except the depot) must be exited exactly once.
            - The number of vehicles leaving the depot equals `num_vehicles`.
            - The number of vehicles returning to the depot equals `num_vehicles`.
            - No self-cycles (e.g., traveling from a location to itself).

        Objective:
            - Minimize the total distance traveled by all vehicles.
        """

        ####################
        # Define variables #
        ####################

        self.x = {
            i: {j: self.m.add_var(f"x({i}, {j})", var_type=BINARY) for j in self.V}
            for i in self.V
        }

        ###################
        # Add constraints #
        ###################

        for j in self.V_excl_depot:
            self.m.add_constr(xsum(self.x[i][j] for i in self.V) == 1)

        for i in self.V_excl_depot:
            self.m.add_constr(xsum(self.x[i][j] for j in self.V) == 1)

        self.m.add_constr(
            xsum(self.x[self.depot_index][j] for j in self.V_excl_depot)
            == self.instance.num_vehicles
        )

        self.m.add_constr(
            xsum(self.x[i][self.depot_index] for i in self.V_excl_depot)
            == self.instance.num_vehicles
        )

        # Prevent self-cycles e.g. [(1, 1)]
        for i in self.V:
            self.m.add_constr(self.x[i][i] == 0)

        #################
        # Add objective #
        #################

        self.m.objective = minimize(
            xsum(
                self.instance.distance_matrix.matrix[i][j] * self.x[i][j]
                for i in self.V
                for j in self.V
            )
        )

    @staticmethod
    def _extract_tours(adj_matrix: List[List[int]]) -> List[List[int]]:
        """
        Converts an adjacency matrix representation of a solution into a list of tours.

        For example, an adjacency matrix like:
            [[0, 1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]
        is transformed into [[0, 1, 0], [2, 3, 2]].

        Args:
            adj_matrix: A square adjacency matrix where `adj_matrix[i][j] == 1`
                if a vehicle travels from location `i` to `j`.

        Returns:
            A list of tours, where each tour is a list of location indices
            representing the path taken (e.g., [depot, loc1, loc2, depot]).
        """
        n = len(adj_matrix)
        if n == 0:
            return []

        used = set()  # Edges (i, j) that have already been used
        tours = []

        def next_unused_edge_from(u):
            for v, val in enumerate(adj_matrix[u]):
                if val == 1 and (u, v) not in used:
                    return v
            return None

        for start in range(n):
            while True:
                v0 = None
                for v, val in enumerate(adj_matrix[start]):
                    if val == 1 and (start, v) not in used:
                        v0 = v
                        break
                if v0 is None:
                    break
                tour = [start]
                current = start

                while True:
                    next_node = next_unused_edge_from(current)
                    if next_node is None:
                        break
                    used.add((current, next_node))
                    current = next_node
                    tour.append(current)
                    if current == start:  # closed cycle
                        break

                if len(tour) > 1 and tour[0] == tour[-1]:
                    tours.append(tour)

        return tours

    def _add_subtour_elimination_constr(self, subtours: List[List[int]]) -> None:
        """Adds subtour elimination constraints to the model for a given list of subtours.

        For each subtour, this method adds a constraint that ensures the number of
        active edges within the subtour's set of nodes is less than the number of
        nodes, effectively breaking the cycle.

        Args:
            subtours: A list of subtours to be eliminated. Each subtour is a list
                of location indices.
        """
        print(f"Found {len(subtours)} subtours. Adding constraints to eliminate them.")
        for subtour in subtours:
            nodes_in_subtour = list(set(subtour))
            if len(nodes_in_subtour) > 1:
                self.m.add_constr(
                    xsum(
                        self.x[i][j] for i in nodes_in_subtour for j in nodes_in_subtour
                    )
                    <= len(nodes_in_subtour) - 1
                )

    def solve(self, time_limit_secs: int = 60) -> VrpSolution | None:
        """Solves the VRP model and returns the optimal tours.

        This method uses an iterative approach to find a solution without subtours:
        1. Solve the current model.
        2. Check the solution for any subtours (tours not including the depot).
        3. If subtours exist, add constraints to eliminate them and re-solve.
        4. Repeat until a valid solution is found or the time limit is reached.

        Args:
            time_limit_secs: The maximum time in seconds allowed for the solver.

        Returns:
            A VrpSolution object representing the optimal solution if one is found
            within the time limit; otherwise, `None`.
        """
        start_time = time.time()

        iterations = 0
        while time.time() - start_time < time_limit_secs:
            remaining_time_secs = time_limit_secs - (time.time() - start_time)
            if remaining_time_secs <= 0:
                print("\nTime limit reached before finding a valid solution.")
                break

            iterations += 1

            self.m.max_seconds = remaining_time_secs
            status = self.m.optimize()

            if status == OptimizationStatus.INFEASIBLE:
                print("\nModel is infeasible!")
                return None

            # We need a solution to check for subtours
            if self.m.num_solutions == 0:
                print(f"\nNo solution found. Status: {status}")
                return None

            adj_matrix = [
                [1 if self.x[i][j].x > 0.001 else 0 for j in self.V] for i in self.V
            ]
            all_tours = self._extract_tours(adj_matrix)
            subtours = [
                tour for tour in all_tours if self.depot_index not in tour
            ]

            if not subtours:
                total_time_secs = time.time() - start_time
                print(
                    f"\nFound valid solution in {total_time_secs:.2f} secs (with {iterations} iterations) with cost: {self.m.objective_value}"
                )
                solution = VrpSolution(
                    tours=all_tours,
                    objective_value=self.m.objective_value,
                    solve_time_secs=total_time_secs,
                    instance=self.instance,
                )
                print(f"Solution: {solution.tours}")
                return solution

            self._add_subtour_elimination_constr(subtours)

        print("\nCould not find a solution without subtours within the time limit.")
        return None
