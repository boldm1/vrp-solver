import os

import folium
import matplotlib.pyplot as plt
import numpy as np

from src.solution import VrpSolution


def print_solution_summary(solution: VrpSolution, solver_name: str):
    """Prints a summary of the solution to the console."""
    print(f"\n--- Solution Summary ({solver_name.upper()}) ---")
    print(f"  - Objective value: {solution.objective_value:.2f}")
    print(f"  - Solve time: {solution.solve_time_secs:.2f} seconds")
    print(f"  - Number of tours: {len(solution.tours)}")
    for i, tour in enumerate(solution.tours):
        tour_locs = " -> ".join([loc.name for loc in tour.locations])
        print(f"    - Tour {i+1}: {tour_locs}")
        print(f"      - Length: {tour.length:.2f}m, Demand: {tour.demand}")
    print("------------------------")


def save_solution_plots(solution: VrpSolution, instance_path: str, solver_name: str):
    """Saves both a static plot and an interactive map of the solution."""
    _save_static_plot(solution, instance_path, solver_name)
    _save_interactive_map(solution, instance_path, solver_name)


def _save_static_plot(solution: VrpSolution, instance_path: str, solver_name: str):
    """Saves a static plot of the solution."""
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{instance_name}_{solver_name}_plot.png")

    print(f"Saving static solution plot to: {plot_path}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"VRP Solution for {instance_name} ({solver_name})")

    # Plot locations
    for loc in solution.instance.distance_matrix.locations:
        is_depot = loc in solution.instance.depots
        ax.plot(
            loc.coords[0],
            loc.coords[1],
            "o",
            ms=10 if is_depot else 5,
            label="Depot" if is_depot else "Customer",
            color="red" if is_depot else "blue",
        )
        ax.text(loc.coords[0], loc.coords[1], f" {loc.name}", fontsize=9)

    # Plot tours
    for tour in solution.tours:
        tour_coords = np.array([loc.coords for loc in tour.locations])
        ax.plot(tour_coords[:, 0], tour_coords[:, 1], "-")

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)


def _save_interactive_map(solution: VrpSolution, instance_path: str, solver_name: str):
    """Saves the solution as an interactive HTML map using Folium."""
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, f"{instance_name}_{solver_name}_map.html")

    print(f"Saving interactive solution map to: {map_path}")

    # Calculate the center of the map. Folium expects (latitude, longitude).
    all_coords_latlon = [
        (loc.coords[1], loc.coords[0])
        for loc in solution.instance.distance_matrix.locations
    ]
    if not all_coords_latlon:
        return  # Cannot plot an empty solution

    center_lat, center_lon = np.mean(all_coords_latlon, axis=0)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Add markers for all locations
    for loc in solution.instance.distance_matrix.locations:
        lat, lon = loc.coords[1], loc.coords[0]
        is_depot = loc in solution.instance.depots
        popup_text = f"<b>{loc.name}</b>"
        icon_color = "red" if is_depot else "blue"
        icon_symbol = "home" if is_depot else "user"

        if not is_depot:
            popup_text += f"<br>Demand: {loc.demand}"

        folium.Marker(
            [lat, lon],
            popup=popup_text,
            icon=folium.Icon(color=icon_color, icon=icon_symbol, prefix="fa"),
        ).add_to(m)

    # Define a list of colors for the routes
    route_colors = [
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "lightred",
        "beige",
        "darkblue",
        "darkgreen",
        "cadetblue",
        "darkpurple",
        "pink",
        "lightblue",
        "lightgreen",
        "gray",
        "black",
        "lightgray",
    ]

    # Add polylines for each tour
    for i, tour in enumerate(solution.tours):
        tour_coords_latlon = [(loc.coords[1], loc.coords[0]) for loc in tour.locations]
        color = route_colors[i % len(route_colors)]
        folium.PolyLine(
            tour_coords_latlon,
            color=color,
            weight=3,
            opacity=0.8,
            popup=f"Tour {i+1}<br>Length: {tour.length:.2f}m",
        ).add_to(m)

    m.save(map_path)
