import json
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class VrpInstance:
    """
    Represents a Vehicle Routing Problem instance.

    This object is an immutable data container for the problem definition.

    Attributes:
        distance_matrix: A square matrix where `distance_matrix[i][j]` is the
            distance between location `i` and location `j`.
        num_vehicles: The number of vehicles available to service the locations.
        depot_index: The index of the depot in the distance matrix.
        location_coords: Optional list of (x, y) coordinates for each location,
            used for plotting the solution.
    """

    distance_matrix: List[List[float]]
    num_vehicles: int
    depot_index: int = 0
    location_coords: Optional[List[Tuple[int, int]]] = None

    def __post_init__(self):
        """Perform validation after initialization."""
        if any(len(row) != self.num_locations for row in self.distance_matrix):
            raise ValueError("Distance matrix must be square.")
        if self.num_vehicles < 1:
            raise ValueError("At least 1 vehicle is required!")

    @property
    def num_locations(self) -> int:
        """Returns the total number of locations, including the depot."""
        return len(self.distance_matrix)

    @classmethod
    def from_json_file(cls, filepath: str) -> "VrpInstance":
        """Loads a VRP instance from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # JSON loads tuples as lists, so convert location_coords back to tuples
        if "location_coords" in data and data["location_coords"] is not None:
            data["location_coords"] = [
                tuple(coord) for coord in data["location_coords"]
            ]

        return cls(
            distance_matrix=data["distance_matrix"],
            num_vehicles=data["num_vehicles"],
            depot_index=data.get("depot_index", 0),
            location_coords=data.get("location_coords"),
        )
