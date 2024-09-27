# Physics-informed neural networks

This directory contains implementations of the following algorithms:

- [Physics-informed neural networks (PINNs)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [Variational PINNs](https://arxiv.org/abs/1912.00873) (future release)

These algorithms are tested using the following datasets:

- Burgers' shock formation (from PINN paper)
- Nonlinear Schrodinger equation (from PINN paper)
- Incompressible Navier-Stokes equation (from PINN paper)
- Kuramoto-Sivashinsky equation (from [pysindy](https://pysindy.readthedocs.io/))

The algorithms and datasets will be expanded further as I have time. 
My goal is to build a sufficiently broad platform to compare the performance of each algorithm. 
Hopefully by testing on a broader swath of problems one will be able to understand 
the situations in which these algorithms both succeed and fail.