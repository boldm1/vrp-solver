from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from src.instance import VrpInstance, Location

@dataclass(frozen=True)
class Tour:
    """Represents a single tour/route for a vehicle."""

    locations: Tuple[Location, ...]
    length: float
    demand: int

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the tour."""
        location_names = " -> ".join(loc.name for loc in self.locations)
        return f"Tour(length={self.length:.2f}, demand={self.demand}, path=[{location_names}])"


@dataclass
class VrpSolution:
    """
    Represents a solution to a Vehicle Routing Problem.

    Attributes:
        tours: A tuple of Tour objects, each representing a vehicle's route.
        objective_value: The total cost (e.g., distance) of the solution.
        solve_time_secs: The time taken to find the solution in seconds.
        instance: The VrpInstance object that this solution solves.
    """

    tours: Tuple[Tour, ...]
    objective_value: float
    solve_time_secs: Optional[float]
    instance: VrpInstance

    def plot(
        self,
        figsize: Tuple[int, int] = (8, 8),
    ):
        """
        Create a plot of the VRP solution.

        The caller is responsible for showing or saving the plot, e.g., by calling
        `plt.show()` or `fig.savefig('plot.png')`.

        Args:
            figsize (tuple): size of the figure.

        Returns:
            The matplotlib Figure object.

        Raises:
            ValueError: If location_coords are not available to plot.
        """
        all_locations = self.instance.distance_matrix.locations
        if not all([loc.coords for loc in all_locations]):
            raise ValueError(
                "Cannot plot solution without coordinates for all locations."
            )

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Plot locations
        location_coords = [loc.coords for loc in all_locations]
        x = [coord[0] for coord in location_coords]
        y = [coord[1] for coord in location_coords]
        ax.scatter(x, y, c="red", s=100, zorder=2, label="Locations")

        # Highlight depot
        for depot in self.instance.depots:
            ax.scatter(
                depot.coords[0],
                depot.coords[1],
                c="indigo",
                s=200,
                zorder=3,
                marker="s",
                label=f"Depot: {depot.name}",
            )

        # Draw each tour
        if self.tours:
            for tour in self.tours:
                tour_coords = [loc.coords for loc in tour.locations]
                tour_x = [coord[0] for coord in tour_coords]
                tour_y = [coord[1] for coord in tour_coords]
                ax.plot(tour_x, tour_y, "b-", zorder=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Route / Solution Plot")
        ax.grid(True)
        ax.legend()

        return fig
