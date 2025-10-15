# Simple-mpc

**Simple-mpc** is a C++ implementation of multiple predictive control schemes for locomotion based on the Aligator optimization solver.

It can be used with quadrupeds and bipeds to generate whole-body walking motions based on a pre-defined contact plan.

## Features

The **Simple-mpc** library provides:

* an interface to generate different locomotion gaits in a MPC-like fashion
* Python bindings to enable fast prototyping
* three different kinds of locomotion dynamics (centroidal, kinodynamics and full dynamics)

## Installation

### Build from source (devel)
0. Install Pixi (from prefix.dev) : see https://pixi.sh/latest/installation/

1. Clone repo.
```bash
git clone git@github.com:Simple-Robotics/simple-mpc.git --recursive
cd simple-mpc
```

2. Build
```bash
pixi run build
```

#### Dependencies

* [Aligator](https://github.com/edantec/aligator) | [conda](https://anaconda.org/conda-forge/aligator)
* [proxsuite](https://github.com/Simple-Robotics/proxsuite.git) | [conda](https://anaconda.org/conda-forge/proxsuite)
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | [conda](https://anaconda.org/conda-forge/pinocchio)
* [hpp-fcl (renamed coal)](https://github.com/humanoid-path-planner/hpp-fcl) | [conda](https://anaconda.org/conda-forge/coal)
* [tsid](https://github.com/stack-of-tasks/tsid) >= 1.9.0 | [conda](https://anaconda.org/conda-forge/tsid)
* [ndcurves](https://github.com/loco-3d/ndcurves)
* [Eigen3](https://eigen.tuxfamily.org) >= 3.3.7
* [eigenpy](https://github.com/stack-of-tasks/eigenpy) >=3.9.0 (Python bindings)
* (optional) [example-robot-data](https://github.com/Gepetto/example-robot-data) (for tests, benchmarks and examples | [conda](https://anaconda.org/conda-forge/example-robot-data)
* (optional) [pybullet](https://github.com/bulletphysics/bullet3) (Simulation examples) | [conda](https://anaconda.org/conda-forge/pybullet)
