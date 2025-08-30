import argparse
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
from collections import defaultdict

from mip import BINARY, Model, OptimizationStatus, minimize, xsum

from src.cli_utils import print_solution_summary, save_solution_plots
from src.instance import Vehicle, VrpInstance
from src.solution import Tour, VrpSolution


class MipSolver:
    """
    A Mixed-Integer Programming (MIP) model for the Vehicle Routing Problem (VRP).

    This class formulates the VRP as a MIP model and uses an iterative approach
    to handle subtour elimination constraints. The model aims to find the set of
    routes with the minimum total distance for a fleet of vehicles to service a set
    of locations, starting and ending at one or more depots.
    """

    def __init__(
        self,
        instance: VrpInstance,
        use_symmetry_breaking: bool = True,
    ):
        """Initializes the MipSolver.

        Args:
            instance: The VrpInstance object containing the problem data.
            use_symmetry_breaking: Whether to add symmetry-breaking constraints.
        """
        self.m = Model(solver_name="HIGHS")
        self.instance = instance

        # Create a list of all vehicles and map them to their home depots.
        self.vehicles: List[Vehicle] = [v for d in instance.depots for v in d.fleet]
        self.num_vehicles = len(self.vehicles)
        self.vehicle_depot_map = {}  # vehicle_idx -> depot_idx
        k = 0
        for depot in self.instance.depots:
            depot_idx = self.instance.distance_matrix.get_index(depot)
            for _ in depot.fleet:
                self.vehicle_depot_map[k] = depot_idx
                k += 1
        self.use_symmetry_breaking = use_symmetry_breaking

        self.depot_idxs = {
            instance.distance_matrix.get_index(depot) for depot in instance.depots
        }
        # Locations to visit
        num_locations = len(instance.distance_matrix.locations)
        self.V = [i for i in range(num_locations)]
        self.V_customers = [v for v in self.V if v not in self.depot_idxs]

        # Create a mapping from customer index to demand
        self.demands = {
            self.instance.distance_matrix.get_index(c): c.demand
            for c in self.instance.customers
        }

        # Build the MIP model
        self._build()

    def _build(self):
        """Builds the VRP model by defining variables, constraints, and the objective.

        This method sets up the initial MIP formulation without subtour elimination
        constraints. These are added lazily during the solving process.

        Variables:
            - x[i, j]: A binary variable that is 1 if a vehicle travels from location
              `i` to `j`, and 0 otherwise.

        Constraints:
            - Each customer must be entered and exited exactly once.
            - The total number of vehicles leaving all depots must equal the total
              number of available vehicles.
            - For each individual depot, the number of vehicles that leave must equal
              the number that return.
            - No self-cycles are allowed (e.g., traveling from a location to itself).

        Objective:
            - Minimize the total distance traveled by all vehicles.
        """

        ####################
        # Define variables #
        ####################

        K = range(self.num_vehicles)
        self.x = {
            (i, j, k): self.m.add_var(var_type=BINARY, name=f"x({i},{j},{k})")
            for i in self.V
            for j in self.V
            for k in K
        }

        ###################
        # Add constraints #
        ###################

        # Add capacity constraints for each vehicle
        for k in K:
            capacity_k = self.vehicles[k].capacity
            if capacity_k > 0:
                # The total demand of customers visited by vehicle k cannot exceed its capacity.
                self.m.add_constr(
                    xsum(
                        self.demands[i] * xsum(self.x[j, i, k] for j in self.V)
                        for i in self.V_customers
                    )
                    <= capacity_k
                )

        if self.use_symmetry_breaking:
            # Group vehicle indices by their properties (depot, capacity, range).
            identical_vehicle_groups = defaultdict(list)
            for k, vehicle in enumerate(self.vehicles):
                depot_k = self.vehicle_depot_map[k]
                key = (depot_k, vehicle.capacity, vehicle.range_kms, vehicle.fixed_cost)
                identical_vehicle_groups[key].append(k)

            # Add symmetry-breaking constraints for identical vehicles from the same depot.
            # When a fleet has multiple identical vehicles, the solver can find many
            # equivalent solutions by simply swapping the tours between these vehicles.
            # These constraints prevent this by imposing an arbitrary order, which can
            # significantly speed up the search for a solution.
            num_symmetry_constrs = 0
            # For each group of identical vehicles, add ordering constraints.
            for group in identical_vehicle_groups.values():
                if len(group) > 1:
                    # Sort vehicle indices to ensure a consistent order for the constraints.
                    sorted_group = sorted(group)
                    depot_idx = self.vehicle_depot_map[sorted_group[0]]
                    for i in range(len(sorted_group) - 1):
                        k1 = sorted_group[i]
                        k2 = sorted_group[i + 1]
                        # Enforce an ordering on vehicle usage:
                        # if vehicle k2 is used, vehicle k1 must also be used.
                        self.m.add_constr(
                            xsum(self.x[depot_idx, j, k2] for j in self.V_customers)
                            <= xsum(self.x[depot_idx, j, k1] for j in self.V_customers)
                        )
                        num_symmetry_constrs += 1

            if num_symmetry_constrs > 0:
                print(
                    f"Added {num_symmetry_constrs} symmetry-breaking constraints for identical vehicles."
                )

        # Each customer must be visited exactly once by some vehicle.
        for j in self.V_customers:
            self.m.add_constr(xsum(self.x[i, j, k] for i in self.V for k in K) == 1)

        # For each vehicle, it must leave a node it enters.
        for k in K:
            # Each vehicle starts and ends at its assigned depot and has one tour at most.
            depot_k = self.vehicle_depot_map[k]
            self.m.add_constr(
                xsum(self.x[depot_k, j, k] for j in self.V_customers) <= 1
            )
            self.m.add_constr(
                xsum(self.x[depot_k, j, k] for j in self.V_customers)
                == xsum(self.x[j, depot_k, k] for j in self.V_customers)
            )
            for i in self.V_customers:
                self.m.add_constr(
                    xsum(self.x[j, i, k] for j in self.V)
                    == xsum(self.x[i, l, k] for l in self.V)
                )

        # Prevent self-cycles e.g. [(1, 1)]
        for i in self.V:
            for k in K:
                self.m.add_constr(self.x[i, i, k] == 0)

        #################
        # Add objective #
        #################

        total_travel_distance = xsum(
            self.instance.distance_matrix.matrix[i][j] * self.x[i, j, k]
            for i in self.V
            for j in self.V
            for k in K
        )

        # Add fixed cost for each vehicle that is used (i.e., leaves the depot)
        total_fixed_cost = xsum(
            self.vehicles[k].fixed_cost
            * xsum(self.x[self.vehicle_depot_map[k], j, k] for j in self.V_customers)
            for k in K
        )

        self.m.objective = minimize(total_travel_distance + total_fixed_cost)

    @staticmethod
    def _extract_tours_from_adj_matrix(adj_matrix: List[List[int]]) -> List[List[int]]:
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
        # Create a successor mapping from the adjacency matrix.
        # The model constraints ensure each node has at most one successor.
        successors = {
            i: j
            for i, row in enumerate(adj_matrix)
            for j, val in enumerate(row)
            if val == 1
        }
        if not successors:
            return []

        visited_nodes = set()
        tours = []
        for i in range(len(adj_matrix)):
            if i in visited_nodes:
                continue

            # Start tracing a tour from an unvisited node
            tour = []
            curr = i
            while curr not in visited_nodes and curr is not None:
                visited_nodes.add(curr)
                tour.append(curr)
                curr = successors.get(curr)

            # If we found a cycle, add it to the list of tours
            if tour and curr == tour[0]:
                tour.append(curr)  # Close the loop
                tours.append(tour)

        return tours

    def _get_tour_length(self, tour: List[int]) -> float:
        """Calculates the total distance of a tour."""
        length = 0
        for i in range(len(tour) - 1):
            u = tour[i]
            v = tour[i + 1]
            length += self.instance.distance_matrix.matrix[u][v]
        return length

    def _add_generalized_subtour_break_constrs(self, subtours: List[List[int]]) -> None:
        """Adds generalized subtour elimination constraints to the model.

        For each subtour, this method adds a constraint that ensures the number of
        active edges within the subtour's set of nodes is less than the number of
        nodes, summed across ALL vehicles. This prevents any vehicle from forming
        the subtour in future iterations.

        Args:
            subtours: A list of subtours to be eliminated. Each subtour is a list
                of customer location indices.
        """
        K = range(self.num_vehicles)
        for subtour in subtours:
            nodes_in_subtour = list(set(subtour))
            if len(nodes_in_subtour) > 1:
                self.m.add_constr(
                    xsum(
                        self.x[i, j, k]
                        for i in nodes_in_subtour
                        for j in nodes_in_subtour
                        for k in K
                    )
                    <= len(nodes_in_subtour) - 1
                )

    def _add_vehicle_specific_tour_break_constrs(
        self, invalid_tours: List[tuple[List[int], int]]
    ) -> None:
        """Adds vehicle-specific tour-breaking constraints to the model.

        This is used for constraints that only apply to a specific vehicle, such
        as a tour that exceeds a vehicle's specific range.

        Args:
            invalid_tours: A list of tours to be eliminated. Each element is a
                tuple containing the tour (a list of location indices) and the
                vehicle index `k` that performed it.
        """
        for tour, k in invalid_tours:
            nodes_in_tour = list(set(tour))
            if len(nodes_in_tour) > 1:
                self.m.add_constr(
                    xsum(self.x[i, j, k] for i in nodes_in_tour for j in nodes_in_tour)
                    <= len(nodes_in_tour) - 1
                )

    def solve(
        self, time_limit_secs: int = 60, verbose: bool = False
    ) -> VrpSolution | None:
        """Solves the VRP model and returns the optimal tours.

        This method uses an iterative approach to find a solution without subtours:
        1. Solve the current model.
        2. Check the solution for any invalid tours (subtours or long tours).
        3. If invalid tours exist, add constraints to eliminate them and re-solve.
        4. Repeat until a valid solution is found or the time limit is reached.

        Args:
            time_limit_secs: The maximum time in seconds allowed for the solver.
            verbose: If `False`, silences the solver's console output.

        Returns:
            A VrpSolution object representing the optimal solution if one is found
            within the time limit; otherwise, `None`.
        """
        start_time = time.time()

        if not verbose:
            self.m.verbose = 0

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

            K = range(self.num_vehicles)
            all_tours_with_vehicle = []
            for k in K:
                # For each vehicle, get its tour from the solution
                adj_matrix_k = [
                    [1 if self.x[i, j, k].x > 0.001 else 0 for j in self.V]
                    for i in self.V
                ]
                tours_k = self._extract_tours_from_adj_matrix(adj_matrix_k)
                for tour in tours_k:
                    all_tours_with_vehicle.append((tour, k))

            # A tour is a subtour if it does not contain any depot.
            subtours_with_vehicle = [
                (tour, k)
                for tour, k in all_tours_with_vehicle
                if not any(depot_idx in tour for depot_idx in self.depot_idxs)
            ]

            if subtours_with_vehicle:
                # We only need the unique subtours, not which vehicle performed them.
                # A generalized constraint will prevent any vehicle from forming it.
                unique_subtours = list(
                    {tuple(sorted(tour)) for tour, k in subtours_with_vehicle}
                )
                print(
                    f"{len(unique_subtours)} unique subtours found. Adding generalized subtour elimination constraints."
                )
                self._add_generalized_subtour_break_constrs(
                    [list(t) for t in unique_subtours]
                )
                continue

            # Check for tour length violations for each vehicle's specific range
            long_tours_with_vehicle = []
            for tour, k in all_tours_with_vehicle:
                vehicle_range = self.vehicles[k].range_kms
                if vehicle_range > 0 and self._get_tour_length(tour) > vehicle_range:
                    long_tours_with_vehicle.append((tour, k))

            if long_tours_with_vehicle:
                print(
                    f"{len(long_tours_with_vehicle)} long tours found. Adding vehicle-specific constraints."
                )
                self._add_vehicle_specific_tour_break_constrs(long_tours_with_vehicle)
                continue

            # Convert index-based tours to Tour objects for the final solution
            solution_tours = []
            all_locations = self.instance.distance_matrix.locations
            for tour_indices, k in all_tours_with_vehicle:
                if len(tour_indices) > 2:  # Only include tours that visit customers
                    tour_locations = tuple(all_locations[i] for i in tour_indices)
                    tour_length = self._get_tour_length(tour_indices)
                    tour_demand = sum(self.demands.get(i, 0) for i in tour_indices)
                    solution_tours.append(
                        Tour(
                            locations=tour_locations,
                            length=tour_length,
                            demand=tour_demand,
                        )
                    )

            # If we reach here, the solution is valid (no subtours and all tours are within range).
            total_time_secs = time.time() - start_time
            print(
                f"\nFound valid solution in {total_time_secs:.2f} secs (with {iterations} iterations) with cost: {self.m.objective_value}"
            )
            solution = VrpSolution(
                tours=tuple(solution_tours),
                objective_value=self.m.objective_value,
                solve_time_secs=total_time_secs,
                instance=self.instance,
            )
            print(f"Solution: {solution.tours}")
            return solution

        print("\nCould not find a solution without subtours within the time limit.")
        return None


def main():
    """
    Main function to run the MIP VRP solver on an instance defined in a JSON file.
    Accepts command-line arguments for the instance file path and other options.
    """
    parser = argparse.ArgumentParser(
        description="Run the MIP VRP solver on a given instance."
    )
    parser.add_argument(
        "instance_path",
        type=str,
        help="Path to the VRP instance JSON file (e.g., 'data/google_vrp_example.json').",
    )
    parser.add_argument(
        "--no-sb",
        action="store_false",
        dest="use_symmetry_breaking",
        help="Disable symmetry-breaking constraints.",
    )
    parser.add_argument(
        "--v",
        action="store_true",
        dest="verbose",
        help="Print the MIP solver's output.",
    )
    args = parser.parse_args()

    # 1. Load the problem instance from file
    data_filepath = args.instance_path
    print(f"Loading instance from: {data_filepath}")
    instance = VrpInstance.from_json_file(args.instance_path)

    # 2. Create and run the solver
    print("\nSolving with MIP exact solver...")
    solver = MipSolver(instance, use_symmetry_breaking=args.use_symmetry_breaking)
    solution = solver.solve(verbose=args.verbose)

    # 3. Process and plot the solution
    if solution:
        print_solution_summary(solution, "mip")
        save_solution_plots(solution, args.instance_path, "mip")
    else:
        print("\nNo solution found by MIP solver.")


if __name__ == "__main__":
    main()
