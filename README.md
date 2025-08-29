# VRP Solver 🚚

A simple Vehicle Routing Problem (VRP) solver that uses Mixed-Integer Programming (MIP) with lazy subtour elimination.

## Overview

This project provides a Python-based solver for the classic Vehicle Routing Problem. It formulates the VRP as a MIP model and solves it using the HiGHS solver, accessed via the `python-mip` library.

A key feature is the iterative (lazy) addition of subtour elimination constraints. There are $2^n$ possible subtours for a set of $n$ locations, and hence the total number of possible subtour constraints grows exponentially with the number of locations. So instead of pre-generating all possible subtour constraints, the model is solved repeatedly, and after each solve the solution is checked for subtours. If any are found, constraints are added to break them, and the model is re-solved. This process continues until a valid, subtour-free solution is found.

The project is structured into three main components for clarity and modularity:

-  `VrpInstance`: A data class to define the problem (distance matrix, number of vehicles, depot location, etc.).
-  `VrpSolver`: The core class that builds the MIP model and implements the iterative solving logic.
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

The basic workflow involves defining a problem instance, passing it to the solver, and then using the returned solution object.

Here is a minimal example:
```python
from src.instance import Customer, Depot, DistanceMatrix, Vehicle, VrpInstance
from src.model import VrpSolver

# 1. Define locations and the distance matrix
depot = Depot(name="D0", coords=(0, 0), fleet=(Vehicle(capacity=0, range_kms=0),))
customer = Customer(name="C1", coords=(1, 1), demand=1)
all_locs = (depot, customer)
dist_matrix = DistanceMatrix(locations=all_locs, matrix=((0, 10), (10, 0)))

# 2. Define the problem instance
instance = VrpInstance(
    depots=(depot,), customers=(customer,), distance_matrix=dist_matrix
)

# 3. Create and build the solver
solver = VrpSolver(instance)
solver.build()

# 4. Solve the model
solution = solver.solve()

# 5. Use the solution
if solution:
    print(f"Tours: {solution.tours}")
    print(f"Cost: {solution.objective_value}")

    # If coordinates were provided in the instance, you can also plot the solution
    # fig = solution.plot()
    # fig.savefig("solution_plot.png")
```
