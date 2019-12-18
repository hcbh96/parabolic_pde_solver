# Project Parabolic Pde Solver
Parabolic PDE solver (the solver) is a solver for parabolic partial differential equa- tions over a 1d space (L) and a given time (T). The solver can be used to plot or analyse the results to the heat equation over time. The solver makes it easy to evaluate solutions with non-homogeneous Dirichlet boundary conditions, internal heat sources and/or non-constant diffusion coefficients.

The solver has been vectorised and uses sparse matrices to improve computa- tional efficiency.
Visualisation functionality has also been included with the solver able to produ- ce 3d surface plots of the temperature of the bar or of an internal heat source as a function of postion (x ∈ [0, L]) and time (t ∈ [0, T]).

The solver has been broken down into subroutines and internal function calls, some of which have been made ecternally available. These functions can be used separately to create and solve a range of Euler and Crank Nicholson matrices required by the heat equation i.e functions to create, compute and solve the matrices required by the separate methods used to solve the heat equation including Forward Euler (FE), Backward Euler (BE) and Crank-Nicholson (CN) respectively. Any external facing functions have been fully documented.


The solver was able to handle non-homogeneous Dirichlet boundary values. The user needed to pass in a function bcf(t) which was then used to calculate the boundary conditions for various values of t ∈ [0, T].

The solver was able to handle internal heat sources. An internal heat source could be specified using heat\_source(x,t). This meant that a heat source could change with both respect to position x ∈ [0, L] and time t ∈ [0, T].

Users could specify non-constant diffusion coefficients by specifying f\_kappa(x). This means bars with varying conductivity can be simulated by passing in a function that returns the diffusion coefficient along x ∈ [0, L].

# Project Numerical Continuation Code
General numerical continuation code for tracking limit cycles
as system parameters change

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

python3 [python3](https://www.python.org/downloads/)

SciPy  [Scipy](https://pypi.org/project/scipy/)

pytest [pytest](pip install pytest)

### Installing
A step by step series of examples that tell you how to get a development env running

Clone git directory

```
git clone https://github.com/hcbh96/parabolic_pde_solver.git
```

Navigate to the directory

```
cd parabolic_pde_solver
```

Visualse Solutions to the Heat equation

```
python3 parabolic_pde_solver.py
```

Visualise the Truncation Errors

```
python3 accuracy_explorer.py
```

## Running the tests

Tests can be run using the terminal command

```
py.test
```


## Authors

* **Harry Blakiston Houston** - *Initial work* - [hcbh96](https://github.com/hcbh96)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details



