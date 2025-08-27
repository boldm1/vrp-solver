import pytest

from src.instance import VrpInstance
from tests.conftest import params


class TestVrpInstance:
    @params(
        {
            "non-square matrix": {
                "distance_matrix": [[0, 1], [1]],
                "num_vehicles": 1,
                "depot_index": 0,
                "error_msg": "Distance matrix must be square.",
            },
            "zero vehicles": {
                "distance_matrix": [[0, 1], [1, 0]],
                "num_vehicles": 0,
                "depot_index": 0,
                "error_msg": "At least 1 vehicle is required!",
            },
        }
    )
    def test_init_errors(self, distance_matrix, num_vehicles, depot_index, error_msg):
        """Test that VrpInstance raises errors on invalid input."""
        with pytest.raises(ValueError, match=error_msg):
            VrpInstance(
                distance_matrix=distance_matrix,
                num_vehicles=num_vehicles,
                depot_index=depot_index,
            )
