from src.instance import Customer, Depot, DistanceMatrix, Vehicle, VrpInstance
from src.mip_solver import MipSolver
from vrp_tests.conftest import params


class TestVrpSolver:
    @params(
        {
            "1 depot, 1 customer": {
                "depots": (
                    Depot(
                        name="D1",
                        coords=(0, 0),
                        fleet=(Vehicle(capacity=0, range_kms=0),),
                    ),
                ),
                "customers": (Customer(name="C1", coords=(1, 1), demand=1),),
                "matrix": ((0, 10), (10, 0)),
                "expected_tours": [[0, 1, 0]],
            },
            "2 depots, 2 customers": {
                "depots": (
                    Depot(
                        name="D1",
                        coords=(0, 0),
                        fleet=(Vehicle(capacity=0, range_kms=0),),
                    ),
                    Depot(
                        name="D2",
                        coords=(10, 10),
                        fleet=(Vehicle(capacity=0, range_kms=0),),
                    ),
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
        solver = MipSolver(instance)
        solution = solver.solve()
        assert solution is not None

        # Convert solution tours back to lists of indices for comparison
        solved_tour_indices = [
            [solution.instance.distance_matrix.get_index(loc) for loc in tour.locations]
            for tour in solution.tours
        ]
        # Compare tours as a set of tuples to ignore order
        assert set(map(tuple, solved_tour_indices)) == set(map(tuple, expected_tours))
        assert solution.instance == instance

    def test_max_tour_length_constraint(self):
        """
        Tests that the max_tour_length constraint forces the solver to use more
        vehicles if the shortest path exceeds the limit.
        """
        # With no constraint, the optimal solution is one tour: D->C1->C2->D, with a
        # length of 10 + 14 + 10 = 34.
        # By setting max_tour_length=30, we force the solver to use two separate tours:
        # D->C1->D (length 20) and D->C2->D (length 20).
        # The total cost becomes 40.
        depots = (
            Depot(
                name="D1",
                coords=(0, 0),
                fleet=(
                    Vehicle(capacity=0, range_kms=30),
                    Vehicle(capacity=0, range_kms=30),
                ),
            ),
        )
        customers = (
            Customer(name="C1", coords=(10, 0), demand=1),
            Customer(name="C2", coords=(0, 10), demand=1),
        )
        all_locations = depots + customers
        matrix = (
            (0, 10, 10),  # D1 -> (D1, C1, C2)
            (10, 0, 14),  # C1 -> (D1, C1, C2)
            (10, 14, 0),  # C2 -> (D1, C1, C2)
        )
        distance_matrix = DistanceMatrix(locations=all_locations, matrix=matrix)

        instance = VrpInstance(
            depots=depots, customers=customers, distance_matrix=distance_matrix
        )
        solver = MipSolver(instance)
        solution = solver.solve()

        assert solution is not None
        assert solution.objective_value == 40

    def test_capacity_constraint(self):
        """
        Tests that the capacity constraint forces the solver to use more
        vehicles if the total demand on a tour exceeds capacity.
        """
        # With no capacity constraint, the optimal solution is one tour visiting both customers.
        # Total demand is 8 + 8 = 16.
        # By setting vehicle capacity to 10, we force two separate tours.
        # The total cost becomes (10+10) + (10+10) = 40.
        depots = (
            Depot(
                name="D1",
                coords=(0, 0),
                fleet=(
                    Vehicle(capacity=10, range_kms=0),
                    Vehicle(capacity=10, range_kms=0),
                ),
            ),
        )
        customers = (
            Customer(name="C1", coords=(10, 0), demand=8),
            Customer(name="C2", coords=(0, 10), demand=8),
        )
        all_locations = depots + customers
        matrix = (
            (0, 10, 10),  # D1 -> (D1, C1, C2)
            (10, 0, 14),  # C1 -> (D1, C1, C2)
            (10, 14, 0),  # C2 -> (D1, C1, C2)
        )
        distance_matrix = DistanceMatrix(locations=all_locations, matrix=matrix)

        instance = VrpInstance(
            depots=depots, customers=customers, distance_matrix=distance_matrix
        )
        solver = MipSolver(instance)
        solution = solver.solve()

        assert solution is not None
        assert solution.objective_value == 40
        assert len(solution.tours) == 2

    def test_fixed_cost(self):
        """
        Tests that the fixed_cost of vehicles is correctly included in the objective.
        """
        # This problem requires two vehicles due to capacity constraints.
        # Total demand is 16, vehicle capacity is 10.
        # Optimal distance is D->C1->D (20) and D->C2->D (20), total = 40.
        # With a fixed cost of 100 per vehicle, the total objective should be:
        # 40 (distance) + 100 (vehicle 1) + 100 (vehicle 2) = 240.
        depots = (
            Depot(
                name="D1",
                coords=(0, 0),
                fleet=(
                    Vehicle(capacity=10, range_kms=0, fixed_cost=100),
                    Vehicle(capacity=10, range_kms=0, fixed_cost=100),
                ),
            ),
        )
        customers = (
            Customer(name="C1", coords=(10, 0), demand=8),
            Customer(name="C2", coords=(0, 10), demand=8),
        )
        all_locations = depots + customers
        matrix = (
            (0, 10, 10),  # D1 -> (D1, C1, C2)
            (10, 0, 14),  # C1 -> (D1, C1, C2)
            (10, 14, 0),  # C2 -> (D1, C1, C2)
        )
        distance_matrix = DistanceMatrix(locations=all_locations, matrix=matrix)

        instance = VrpInstance(
            depots=depots, customers=customers, distance_matrix=distance_matrix
        )
        solver = MipSolver(instance)
        solution = solver.solve()

        assert solution is not None
        assert solution.objective_value == 240
        assert len(solution.tours) == 2

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
            "Test 4": {
                "adjacency_matrix": [[1, 0], [0, 0]],
                "expected_result": [[0, 0]],
            },
            # Larger cycle 0→1→2→0
            "Test 5": {
                "adjacency_matrix": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                "expected_result": [[0, 1, 2, 0]],
            },
            # Multiple disjoint cycles (0→1→0 and 2→3→2)
            "Test 6": {
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
        tours = MipSolver._extract_tours_from_adj_matrix(adjacency_matrix)
        # Compare tours as a set of tuples to ignore order
        assert set(map(tuple, tours)) == set(map(tuple, expected_result))
