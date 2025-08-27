from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.instance import VrpInstance

import matplotlib.pyplot as plt

@dataclass
class VrpSolution:
    """
    Represents a solution to a Vehicle Routing Problem.

    Attributes:
        tours: A list of tours, where each tour is a list of location indices.
        objective_value: The total cost (e.g., distance) of the solution.
        solve_time_secs: The time taken to find the solution in seconds.
        instance: The VrpInstance object that this solution solves.
    """

    tours: List[List[int]]
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
        if not self.instance.location_coords:
            raise ValueError("Cannot plot solution without location_coords.")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Plot locations
        x = [coord[0] for coord in self.instance.location_coords]
        y = [coord[1] for coord in self.instance.location_coords]
        ax.scatter(x, y, c="red", s=100, zorder=2, label="Locations")

        # Highlight depot
        if self.instance.location_coords:
            ax.scatter(
                x[self.instance.depot_index],
                y[self.instance.depot_index],
                c="indigo",
                s=200,
                zorder=3,
                marker="s",
                label="Depot",
            )

        # Draw each tour
        if self.tours:
            for tour in self.tours:
                tour_x = [x[i] for i in tour]
                tour_y = [y[i] for i in tour]
                ax.plot(tour_x, tour_y, "b-", zorder=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Route / Solution Plot")
        ax.grid(True)
        ax.legend()

        return fig
