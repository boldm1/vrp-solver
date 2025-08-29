import abc
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Location(abc.ABC):
    """
    Represents an abstract base class for a location in the VRP problem.

    Attributes:
        name: The name of the location.
        coords: A tuple representing the (x, y) coordinates of the location.
    """

    name: str
    coords: Tuple[int, int]


@dataclass(frozen=True)
class Customer(Location):
    """
    Extends the Location class to represent a customer with a demand.
    """

    demand: int


@dataclass(frozen=True)
class Vehicle:
    """
    Represents a vehicle in the VRP problem.

    Attributes:
        capacity: The capacity of the vehicle.
        range_kms: The range of the vehicle in kilometers.
        fixed_cost: The fixed cost of using the vehicle.
    """

    capacity: int
    range_kms: int
    fixed_cost: float = 0.0


@dataclass(frozen=True)
class Depot(Location):
    """
    Extends the Location class to represent a depot with an associated fleet of vehicles.
    """

    fleet: Tuple[Vehicle, ...]

    @property
    def num_vehicles(self) -> int:
        """Returns the number of vehicles in the fleet."""
        return len(self.fleet)



@dataclass(frozen=True)
class DistanceMatrix:
    """
    A class to manage the distance matrix and the mapping between locations and indices.

    This class ensures that the distance matrix and the locations it corresponds to
    are always kept in sync.
    """

    locations: Tuple[Location, ...]
    matrix: Tuple[Tuple[float, ...], ...]

    def __post_init__(self):
        """Validates input and creates the location-to-index mapping."""
        if len(self.locations) != len(self.matrix):
            raise ValueError(
                "Number of locations must match the dimension of the distance matrix."
            )

    def get_distance(self, loc1: Location, loc2: Location) -> float:
        """Returns the distance between two locations using the location objects."""
        idx1 = self.locations.index(loc1)
        idx2 = self.locations.index(loc2)
        return self.matrix[idx1][idx2]

    def get_index(self, location: Location) -> int:
        """Returns the matrix index for a given location object."""
        return self.locations.index(location)


@dataclass(frozen=True)
class VrpInstance:
    """
    Represents a Vehicle Routing Problem instance.

    This object is an immutable data container for the problem definition.

    Attributes:
        depots: A tuple of Depot objects.
        customers: A tuple of Customer objects.
        distance_matrix: The DistanceMatrix object for the instance.
    """

    depots: Tuple[Depot, ...]
    customers: Tuple[Customer, ...]
    distance_matrix: DistanceMatrix

    @property
    def num_vehicles(self) -> int:
        """Returns the total number of vehicles across all depots."""
        return sum(d.num_vehicles for d in self.depots)

    def __post_init__(self):
        """Validates that all depots and customers are in the distance matrix."""
        dm_locations = set(self.distance_matrix.locations)

        for depot in self.depots:
            if depot not in dm_locations:
                raise ValueError(f"Depot '{depot.name}' is not in the distance matrix.")

        for customer in self.customers:
            if customer not in dm_locations:
                raise ValueError(
                    f"Customer '{customer.name}' is not in the distance matrix."
                )

    @classmethod
    def from_json_file(cls, filepath: str) -> "VrpInstance":
        """Loads a VRP instance from a JSON file."""

        with open(filepath, "r") as f:
            data = json.load(f)

        # Create a map of depot-specific data for easy lookup
        depot_info_map = {d["name"]: d for d in data["depots"]}

        # 1. Create a single, canonical object for each location.
        # The order must match the distance matrix.
        all_locations = []
        for loc_data in data["locations"]:
            loc_name = loc_data["name"]
            if loc_name in depot_info_map:
                # This is a depot, create a Depot object
                depot_spec = depot_info_map[loc_name]
                fleet = []
                for vehicle_spec in depot_spec["fleet"]:
                    fleet.append(
                        Vehicle(
                            capacity=vehicle_spec.get("capacity", 0),
                            range_kms=vehicle_spec.get("range_kms", 0),
                            fixed_cost=vehicle_spec.get("fixed_cost", 0.0),
                        )
                    )
                all_locations.append(
                    Depot(
                        name=loc_name,
                        coords=tuple(loc_data["coords"]),
                        fleet=tuple(fleet),
                    )
                )
            else:
                # This is a customer, create a Customer object
                all_locations.append(
                    Customer(
                        name=loc_name,
                        coords=tuple(loc_data["coords"]),
                        demand=loc_data.get("demand", 0),
                    )
                )

        # 2. Create different "views" (depots, customers) from the canonical list.
        # These are tuples of references, not new objects.
        depots = tuple(loc for loc in all_locations if isinstance(loc, Depot))
        customers = tuple(loc for loc in all_locations if isinstance(loc, Customer))

        # 3. Create the DistanceMatrix using the canonical list of locations.
        dist_matrix = DistanceMatrix(
            locations=tuple(all_locations),
            matrix=tuple(map(tuple, data["distance_matrix"])),
        )

        # 4. Create and return the VrpInstance.
        return cls(
            depots=depots,
            customers=customers,
            distance_matrix=dist_matrix,
        )
