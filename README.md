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

1. Clone repo.
```bash
mkdir -p simple-mpc_ws/src
cd simple-mpc_ws/src
git clone git@github.com:Simple-Robotics/simple-mpc.git --recursive
```

2. Create conda environment.
(It is recommended to use `mamba` instead of `conda` for faster/better dependencies solving)
```bash
mamba env create -f simple-mpc/environment-devel.yaml
mamba activate simple-mpc-devel
```

3. Clone some dependencies
(Some dependencies are not available on conda, or not with adequate versions)
(vcs allow for cloning and managing multiple repo at once)
```bash
vcs import --recursive < simple-mpc/devel-git-deps.yaml
```

4. Build all dependencies of simple-mpc:
(Due to a renaming issue, you may need to rename hpp-fcl into coal in the package.xml file of the Pinocchio library)
```bash
export MAKEFLAGS="-j4" # It is recommended to reduce the number of jobs as you ram might get full easily with the default number.
cd ..
colcon build --event-handlers console_direct+ --packages-ignore simple-mpc --cmake-args \
-DCMAKE_BUILD_TYPE=Release             \
-DCMAKE_PREFIX_PATH=$CONDA_PREFIX      \
-DPYTHON_EXECUTABLE=$(which python)    \
-DCMAKE_CXX_COMPILER_LAUNCHER='ccache' \
-DBUILD_TESTING=OFF                    \
-DBUILD_DOCUMENTATION=OFF              \
-DBUILD_EXAMPLES=OFF                   \
-DBUILD_BENCHMARK=OFF                  \
-DBUILD_BENCHMARKS=OFF                 \
-DBUILD_WITH_COLLISION_SUPPORT=ON      \
-DGENERATE_PYTHON_STUBS=OFF            \
-DCOAL_BACKWARD_COMPATIBILITY_WITH_HPP_FCL=ON
```

5. Source the environment and install simple-mpc:
```bash
source install/setup.bash
colcon build --packages-select simple-mpc
```

6. Source the environment one more time
(This step needs to be repeated every time a new shell is opened. It can be put in your ~/.bashrc)
```bash
mamba activate simple-mpc-devel # If not already done
source install/setup.bash
```

#### Dependencies

* [proxsuite-nlp](https://github.com/Simple-Robotics/proxsuite-nlp.git)
* [proxsuite](https://github.com/Simple-Robotics/proxsuite.git)
* [Eigen3](https://eigen.tuxfamily.org) >= 3.3.7
* [Boost](https://www.boost.org) >= 1.71.0
* OpenMP
* [hpp-fcl](https://github.com/humanoid-path-planner/hpp-fcl)
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | [conda](https://anaconda.org/conda-forge/pinocchio)
* [Aligator](https://github.com/edantec/aligator) temporary_fix branch
* [example-robot-data](https://github.com/Gepetto/example-robot-data)
* [ndcurves](https://github.com/loco-3d/ndcurves)
* (optional) [eigenpy](https://github.com/stack-of-tasks/eigenpy)>=3.9.0 (Python bindings)
* (optional) [bullet](https://github.com/bulletphysics/bullet3) (Simulation examples)
