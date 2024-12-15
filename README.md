# Code for the Qinnovision World Challenge 2025

This repository contains the code and resources used for the Qinnovision World Challenge 2025. Our project focuses on formulating and solving Vehicle Routing Problems (VRP) as Quadratic Unconstrained Binary Optimization (QUBO) problems, leveraging quantum and hybrid quantum-classical methods. The implementation uses CUDA-Q for solving Max-Cut instances derived from VRP formulations.

## Folder Structure

- **`TestSet/`**: Contains Q matrices representing VRP instances used for testing and benchmarking. Instances generated from https://github.com/smharwood/vrp-as-qubo
- **`QAOA_Cuda_Q.ipynb`**: Jupyter Notebook implementation of QAOA using CUDA-Q for solving Max-Cut problems using decomposition.
- **`QAOA_Cuda_Q.py`**: Python script to solve the QUBO instances derived from VRP using QAOA using decomposition.
- **`VRP_Challenge.ipynb`**: Jupyter Notebook implementation of QAOA using CUDA-Q for solving Max-Cut problems.
- **`VRP_Challenge.py`**: Python script to solve the QUBO instances derived from VRP using QAOA.
- **`results-tensolver.csv`**: Contains results of the tensor-based solver TenSolver.jl for QUBO instances.
- **`run.jl`**: Julia script for running instances and generating additional results.

## Time Windows for QUBO Generation

The following time windows were used to generate each QUBO instance:

```
16 21 26 31 36 41 46 51
```
