import argparse
import time
from collections import defaultdict

import numpy as np
from pyvrp import Client as PyVrpClient
from pyvrp import Depot as PyVrpDepot
from pyvrp import Model as PyVrpModel
from pyvrp import ProblemData as PyVrpProblemData
from pyvrp import VehicleType as PyVrpVehicleType
from pyvrp.stop import MaxIterations

from src.cli_utils import print_solution_summary, save_solution_plots
from src.instance import Customer, VrpInstance
from src.solution import Tour, VrpSolution


class PyVrpSolver:
    """
    A solver for the Vehicle Routing Problem (VRP) that uses the `pyvrp`
    heuristic library.

    This class acts as an adapter, translating the project's `VrpInstance`
    data model into the format expected by `pyvrp`, solving the problem,
    and then translating the result back into a `VrpSolution` object.
    """

    def __init__(self, instance: VrpInstance):
        """Initializes the PyVrpSolver."""
        self.instance = instance
        self._data: PyVrpProblemData = self._create_problem_data()

    def _create_problem_data(self) -> PyVrpProblemData:
        """
        Creates a `pyvrp.ProblemData` object from the `VrpInstance`.
        """
        # 1. Create pyvrp locations (depots and clients)
        pyvrp_depots = [
            PyVrpDepot(x=d.coords[0], y=d.coords[1]) for d in self.instance.depots
        ]
        pyvrp_clients = [
            PyVrpClient(x=c.coords[0], y=c.coords[1], delivery=[c.demand])
            for c in self.instance.customers
        ]

        # 2. Create vehicle types. PyVRP groups identical vehicles into types.
        vehicle_types = []
        for depot_idx, depot in enumerate(self.instance.depots):
            # Group vehicles in this depot by their properties
            grouped_vehicles = defaultdict(int)
            for vehicle in depot.fleet:
                key = (vehicle.capacity, vehicle.range_kms, vehicle.fixed_cost)
                grouped_vehicles[key] += 1

            for (capacity, range_kms, fixed_cost), num in grouped_vehicles.items():
                vehicle_types.append(
                    PyVrpVehicleType(
                        num_available=num,
                        capacity=[capacity],
                        start_depot=depot_idx,
                        end_depot=depot_idx,
                        tw_early=0,
                        tw_late=np.iinfo(np.int64).max,
                        max_distance=range_kms if range_kms > 0 else 0,
                        fixed_cost=fixed_cost,
                    )
                )

        # 3. Create the distance matrix in the order pyvrp expects: depots, then clients.
        ordered_locs = list(self.instance.depots) + list(self.instance.customers)
        num_locs = len(ordered_locs)
        dist_matrix: np.ndarray = np.zeros((num_locs, num_locs), dtype=int)

        for i, loc1 in enumerate(ordered_locs):
            for j, loc2 in enumerate(ordered_locs):
                # Round to integer as pyvrp expects integer distances/durations
                dist_matrix[i, j] = round(
                    self.instance.distance_matrix.get_distance(loc1, loc2)
                )

        return PyVrpProblemData(
            clients=pyvrp_clients,
            depots=pyvrp_depots,
            vehicle_types=vehicle_types,
            distance_matrices=[dist_matrix],
            duration_matrices=[dist_matrix],
        )

    def solve(self, stop: MaxIterations = MaxIterations(5000)) -> VrpSolution | None:
        """
        Solves the VRP using pyvrp and translates the result.

        Args:
            stop: A stopping criterion for the pyvrp solver.

        Returns:
            A VrpSolution object if a feasible solution is found, otherwise None.
        """
        start_time = time.time()

        model = PyVrpModel.from_data(self._data)
        result = model.solve(stop)

        if not result.is_feasible():
            return None

        # Translate pyvrp result back to VrpSolution
        solution_tours = []
        # The canonical list of locations used to build the problem data
        all_locations = list(self.instance.depots) + list(self.instance.customers)

        for route in result.best.routes():
            if not route.visits():
                continue  # Skip empty routes

            # pyvrp route indices refer to the combined list of [depots] + [clients]
            tour_locations = tuple(all_locations[i] for i in route.visits())
            depot_idx = route.start_depot()  # Assumed that start_depot == end_depot
            depot_loc = self.instance.depots[depot_idx]

            # Reconstruct the full tour path including the depot
            full_tour_locs = (depot_loc,) + tour_locations + (depot_loc,)

            tour_demand = sum(
                loc.demand for loc in tour_locations if isinstance(loc, Customer)
            )

            solution_tours.append(
                Tour(
                    locations=full_tour_locs,
                    length=route.distance(),
                    demand=tour_demand,
                )
            )

        total_time_secs = time.time() - start_time

        return VrpSolution(
            tours=tuple(solution_tours),
            objective_value=result.cost(),
            solve_time_secs=total_time_secs,
            instance=self.instance,
        )


def main():
    """
    Main function to run the heuristic VRP solver on an instance defined in a JSON file.
    """
    parser = argparse.ArgumentParser(
        description="Run the pyvrp-based heuristic VRP solver on a given instance."
    )
    parser.add_argument(
        "instance_path",
        type=str,
        help="Path to the VRP instance JSON file (e.g., 'data/google_vrp_example.json').",
    )
    args = parser.parse_args()

    # 1. Load the problem instance from file
    data_filepath = args.instance_path
    print(f"Loading instance from: {data_filepath}")
    instance = VrpInstance.from_json_file(data_filepath)

    # 2. Create and run the solver
    print("\nSolving with pyvrp heuristic solver...")
    solver = PyVrpSolver(instance)
    solution = solver.solve()

    # 3. Process and plot the solution
    if solution:
        print_solution_summary(solution, "PyVRP")
        save_solution_plots(solution, data_filepath, "PyVRP")
    else:
        print("\nNo solution found by pyvrp.")


if __name__ == "__main__":
    main()
