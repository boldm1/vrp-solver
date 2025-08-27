import json
import os

from matplotlib import pyplot as plt

from src.instance import VrpInstance
from src.model import VrpSolver
from tests.conftest import params


class TestVrpSolver:
    @params(
        {
            # 0 vehicles
            "Test 1": {
                "distance_matrix": [[0, 10], [10, 0]],
                "num_vehicles": 1,
                "depot_index": 0,
                "expected_tours": [[0, 1, 0]],
            },
        }
    )
    def test_basic_cases(
        self, distance_matrix, num_vehicles, depot_index, expected_tours
    ):
        instance = VrpInstance(
            distance_matrix=distance_matrix,
            num_vehicles=num_vehicles,
            depot_index=depot_index,
        )
        solver = VrpSolver(instance)
        solver.build()
        solution = solver.solve()
        assert solution is not None
        assert solution.tours == expected_tours
        assert solution.instance.distance_matrix == distance_matrix
        assert solution.instance.location_coords is None

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
