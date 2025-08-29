import argparse
import os

from matplotlib import pyplot as plt

from src.instance import VrpInstance
from src.model import VrpSolver


def main():
    """
    Main function to run the VRP solver on an instance defined in a JSON file.
    Accepts command-line arguments for the instance file path and other options.
    """
    parser = argparse.ArgumentParser(
        description="Run the VRP solver on a given instance."
    )
    parser.add_argument(
        "instance_path",
        type=str,
        help="Path to the VRP instance JSON file (e.g., 'data/google_vrp_example.json').",
    )
    parser.add_argument(
        "--no-sb",
        action="store_false",
        dest="use_symmetry_breaking",
        help="Disable symmetry-breaking constraints.",
    )
    parser.add_argument(
        "--v",
        action="store_true",
        dest="verbose",
        help="Print the MIP solver's output.",
    )
    args = parser.parse_args()

    # 1. Load the problem instance from file
    data_filepath = args.instance_path
    print(f"Loading instance from: {data_filepath}")
    instance = VrpInstance.from_json_file(data_filepath)

    # 2. Create and build the solver
    solver = VrpSolver(instance, use_symmetry_breaking=args.use_symmetry_breaking)
    solver.build()

    # 3. Solve the model
    solution = solver.solve(verbose=args.verbose)

    # 4. Process and plot the solution
    if solution:
        print("\n--- Solution Summary ---")
        print(f"Objective value: {solution.objective_value:.4f}")
        print(f"Solve time: {solution.solve_time_secs:.2f} seconds")
        print("Tours:")
        for i, tour in enumerate(solution.tours):
            print(f"  Vehicle {i+1}: {tour}")

        # Create a directory for plots if it doesn't exist
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        # Get the filename from the data filepath to create a unique plot name
        plot_filename = os.path.splitext(os.path.basename(data_filepath))[0] + ".png"
        save_path = os.path.join(plots_dir, plot_filename)

        # Generate the plot and save it
        print(f"\nSaving solution plot to: {save_path}")
        fig = solution.plot()
        fig.savefig(save_path)
        plt.close(fig)  # Close the figure to free up memory
    else:
        print("\nNo solution found.")


if __name__ == "__main__":
    main()
