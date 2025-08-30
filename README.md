# VRP Solver 🚚

A Vehicle Routing Problem (VRP) solver that uses Mixed-Integer Programming (MIP) to solve the following types of VRPs:

- Vehicle capacities (capacitated VRP)
- Multiple depots
- Fleets of heterogeneous vehicles, i.e. with different capacities, ranges, costs.

## Overview

This project provides two Python-based solvers for the classic Vehicle Routing Problem: a custom MIP solver and a metaheuristic solver using the PyVRP library.

The **MIP solver** formulates the VRP as a Mixed-Integer Programming model and solves it using the [HiGHS](https://highs.dev/) solver, accessed via the `python-mip` library. A key feature is the iterative (lazy) addition of subtour elimination constraints. Instead of pre-generating all possible subtour constraints (which grow exponentially), the model is solved repeatedly. After each solve, the solution is checked for invalid tours. If any are found, constraints are added to break them, and the model is re-solved. This process continues until a valid, subtour-free solution is found.

The **PyVRP solver** interfaces with the [PyVRP](https://github.com/PyVRP/PyVRP) solver.

The project is structured into several main components for clarity and modularity:

-  `VrpInstance`: A data class to define the problem (distance matrix, number of vehicles, depot location, etc.).
-  `MipSolver`: The core class that builds the MIP model and implements the iterative solving logic.
-  `PyVrpSolver`: A wrapper to adapt the problem instance and run the PyVRP solver.
-  `VrpSolution`: A data class to hold and interact with the solution, including the final tours, objective cost, and a plotting method.

## 🛠️ Installation

This project uses Poetry for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mattbold/vrp-solver.git
    cd vrp-solver
    ```

2.  **Install dependencies:**
    Ensure you have Poetry installed. Then, run the following command in the project root. This will create a dedicated virtual environment and install all required packages from the `pyproject.toml` file.
    ```bash
    poetry install
    ```

## ✅ Running Tests

The test suite is built with `pytest`. To run all tests and verify the installation, use the following command:

```bash
poetry run pytest
```

## 🚀 Usage

The project provides two command-line entry points, one for each solver. Both accept a path to a problem instance JSON file, output the solution summary to the console, and save a plot of the solution to the `plots/` directory.

The solvers generate two types of plots for each solution:
- A static plot (`.png`) showing the routes on a 2D coordinate plane.
- An interactive map (`.html`) showing the routes on a real-world map using OpenStreetMap tiles. You can open this file in your browser to explore the solution.

### MIP Solver

This solver provides an exact solution using Mixed-Integer Programming. It is generally slower but guarantees optimality for smaller instances.

**Basic Usage:**
```bash
poetry run python src/mip_solver.py <path_to_instance_file>
```

The script will output the solution summary to the console and save a plot of the solution to the `plots/` directory.

**Options:**
- `--no-sb`: Disable symmetry-breaking constraints.
- `--v`: Print the MIP solver's output.

### PyVRP Solver

This solver uses the fast PyVRP metaheuristic to find a high-quality solution. It is generally faster and can handle larger instances.

**Basic Usage:**
```bash
poetry run python src/pyvrp_solver.py <path_to_instance_file>
```

![Example solution](assets/solution_example.png)
