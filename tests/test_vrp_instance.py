import pytest

from src.instance import Customer, Depot, DistanceMatrix, VrpInstance
from tests.conftest import params


class TestVrpInstance:
    @params(
        {
            "mismatched locations and matrix": {
                "locations": (Depot("D1", (0, 0), 1),),
                "matrix": ((0, 1), (1, 0)),
                "error_msg": "Number of locations must match the dimension of the distance matrix.",
            },
        }
    )
    def test_distance_matrix_errors(self, locations, matrix, error_msg):
        """Test that DistanceMatrix raises errors on invalid input."""
        with pytest.raises(ValueError, match=error_msg):
            DistanceMatrix(locations=locations, matrix=matrix)

    @params(
        {
            "depot not in matrix": {
                "depots": (Depot("D1", (0, 0), 1),),
                "customers": (),
                "dm_locations": (Customer("C1", (1, 1), 1),),
                "error_msg": "Depot 'D1' is not in the distance matrix.",
            },
            "customer not in matrix": {
                "depots": (Depot("D1", (0, 0), 1),),
                "customers": (Customer("C1", (1, 1), 1),),
                "dm_locations": (Depot("D1", (0, 0), 1),),
                "error_msg": "Customer 'C1' is not in the distance matrix.",
            },
        }
    )
    def test_instance_validation_errors(
        self, depots, customers, dm_locations, error_msg
    ):
        """Test that VrpInstance raises errors on data integrity issues."""
        # A valid distance matrix is required for the test
        matrix_dim = len(dm_locations)
        matrix = tuple([tuple([0] * matrix_dim)] * matrix_dim)
        distance_matrix = DistanceMatrix(locations=dm_locations, matrix=matrix)

        with pytest.raises(ValueError, match=error_msg):
            VrpInstance(
                depots=depots,
                customers=customers,
                distance_matrix=distance_matrix,
            )
