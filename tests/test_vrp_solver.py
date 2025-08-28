import json
import os

from matplotlib import pyplot as plt

from src.instance import Customer, Depot, DistanceMatrix, VrpInstance
from src.model import VrpSolver
from tests.conftest import params


class TestVrpSolver:
    @params(
        {
            "1 depot, 1 customer": {
                "depots": (Depot(name="D1", coords=(0, 0), num_vehicles=1),),
                "customers": (Customer(name="C1", coords=(1, 1), demand=1),),
                "matrix": ((0, 10), (10, 0)),
                "expected_tours": [[0, 1, 0]],
            },
            "2 depots, 2 customers": {
                "depots": (
                    Depot(name="D1", coords=(0, 0), num_vehicles=1),
                    Depot(name="D2", coords=(10, 10), num_vehicles=1),
                ),
                "customers": (
                    Customer(name="C1", coords=(0, 5), demand=1),
                    Customer(name="C2", coords=(10, 5), demand=1),
                ),
                "matrix": (
                    (0, 14, 5, 100),  # D1 -> (D1, D2, C1, C2)
                    (14, 0, 100, 5),  # D2 -> (D1, D2, C1, C2)
                    (5, 100, 0, 10),  # C1 -> (D1, D2, C1, C2)
                    (100, 5, 10, 0),  # C2 -> (D1, D2, C1, C2)
                ),
                "expected_tours": [[0, 2, 0], [1, 3, 1]],
            },
        }
    )
    def test_basic_cases(self, depots, customers, matrix, expected_tours):
        # The order of locations for the distance matrix is depots then customers
        all_locations = depots + customers
        distance_matrix = DistanceMatrix(locations=all_locations, matrix=matrix)

        instance = VrpInstance(
            depots=depots, customers=customers, distance_matrix=distance_matrix
        )
        solver = VrpSolver(instance)
        solver.build()
        solution = solver.solve()
        assert solution is not None
        # Compare tours as a set of tuples to ignore order
        assert set(map(tuple, solution.tours)) == set(map(tuple, expected_tours))
        assert solution.instance == instance

    def test_with_google_vrp_example(self):
        """Test solver on instance taken from https://developers.google.com/optimization/routing/vrp."""

        instance = VrpInstance.from_json_file("data/google_vrp_example.json")
        solver = VrpSolver(instance)
        solver.build()
        solution = solver.solve()

        assert solution is not None, "Solver failed to find a solution"

        # Create a directory for plots if it doesn't exist
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, "google_vrp_solution.png")

        # Generate the plot and save it
        fig = solution.plot()
        fig.savefig(save_path)

        # Close the figure to free up memory and prevent it from being displayed
        plt.close(fig)

        assert os.path.exists(save_path)

    @params(
        {
            # Empty matrix -> no tours
            "Test 1": {"adjacency_matrix": [], "expected_result": []},
            # Single node with no edges -> no tours
            "Test 2": {"adjacency_matrix": [[0]], "expected_result": []},
            # Simple closed loop between two nodes
            "Test 3": {
                "adjacency_matrix": [[0, 1], [1, 0]],
                "expected_result": [[0, 1, 0]],
            },
            # Depot with two independent tours
            "Test 4": {
                "adjacency_matrix": [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
                "expected_result": [[0, 1, 0], [0, 2, 0]],
            },
            # Self-loops (should include them as tours if treated as valid tours)
            "Test 5": {
                "adjacency_matrix": [[1, 0], [0, 0]],
                "expected_result": [[0, 0]],
            },
            # Larger cycle 0→1→2→0
            "Test 6": {
                "adjacency_matrix": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                "expected_result": [[0, 1, 2, 0]],
            },
            # Multiple disjoint cycles (0→1→0 and 2→3→2)
            "Test 7": {
                "adjacency_matrix": [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                "expected_result": [[0, 1, 0], [2, 3, 2]],
            },
        }
    )
    def test_extract_tours(self, adjacency_matrix, expected_result):
        tours = VrpSolver._extract_tours(adjacency_matrix)
        assert tours == expected_result
