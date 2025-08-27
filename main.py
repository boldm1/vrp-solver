import json
import os

from matplotlib import pyplot as plt

from src.instance import VrpInstance
from src.model import VrpSolver


def main():
    """Main function to run the VRP solver on the Google VRP example."""
    print("Running VRP solver on Google OR-Tools example...")

    # 1. Load the problem instance from file
    data_filepath = "data/google_vrp_example.json"
    instance = VrpInstance.from_json_file(data_filepath)

    # 2. Create and build the solver
    solver = VrpSolver(instance)
    solver.build()

    # 3. Solve the model
    solution = solver.solve()

    # 4. Process and plot the solution
    if solution:
        print("\n--- Solution Summary ---")
        print(f"Objective value: {solution.objective_value}")
        print(f"Solve time: {solution.solve_time_secs:.2f} seconds")
        print("Tours:")
        for i, tour in enumerate(solution.tours):
            print(f"  Vehicle {i+1}: {tour}")

        # Create a directory for plots if it doesn't exist
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, "google_vrp_solution.png")

        # Generate the plot and save it
        print(f"\nSaving solution plot to: {save_path}")
        fig = solution.plot()
        fig.savefig(save_path)
        plt.close(fig)  # Close the figure to free up memory
    else:
        print("\nNo solution found.")


if __name__ == "__main__":
    main()
