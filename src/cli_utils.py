import os

from matplotlib import pyplot as plt

from src.solution import VrpSolution


def print_solution_summary(solution: VrpSolution):
    """Prints a summary of the VRP solution to the console."""
    print("\n------------------------")
    print("--- Solution Summary ---")
    print("------------------------\n")
    print(f"Objective value: {solution.objective_value:.4f}")
    print(f"Solve time: {solution.solve_time_secs:.2f} secs")
    print("Tours:")
    for i, tour in enumerate(solution.tours):
        print(f"\tTour {i+1}: {tour}")


def save_solution_plot(
    solution: VrpSolution,
    data_filepath: str,
    solver_type: str,
):
    """Saves a plot of the VRP solution to a file."""
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(data_filepath))[0]
    plot_filename = f"{base_filename}_{solver_type}.png"
    save_path = os.path.join(plots_dir, plot_filename)

    print(f"\nSaving solution plot to: {save_path}")
    fig = solution.plot()
    fig.savefig(save_path)
    plt.close(fig)
