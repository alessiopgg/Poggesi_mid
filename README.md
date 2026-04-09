K-MEANS OPENMP PROJECT

This project implements K-Means clustering in C++ with:
- one sequential version
- one OpenMP version with AoS layout
- one OpenMP version with SoA layout

Project files:
- kmeans_seq.cpp / .h        -> sequential implementation
- kmeans_omp_aos.cpp / .h    -> OpenMP AoS version
- kmeans_omp_soa.cpp / .h    -> OpenMP SoA version
- dataset.cpp / .h           -> CSV loading and centroid initialization
- main.cpp                   -> benchmark program
- main_correctness.cpp       -> correctness test
- CMakeLists.txt             -> build configuration

Build:
- C++17
- OpenMP required
- Release flags include optimization and vectorization

Executables:
- Poggesi_mid
- Poggesi_mid_correctness

Benchmark settings:
- Datasets from 10k to 1M points
- K values: 6, 10, 20
- Threads: 1, 2, 4, 8, 16, 32
- Schedules: static, dynamic
- Chunks: 0, 1, 32, 128
- 5 runs with 1 warm-up run

Correctness:
The parallel versions are checked against the sequential one by comparing:
- final SSE
- number of iterations

Main result:
The best overall configuration is usually the SoA OpenMP version with static scheduling.
The best speedup reported in the project is about 4.6x on the largest dataset.

Notes:
- CSV loading and centroid initialization are not included in the measured runtime
- Synthetic 2D datasets are used for the experiments
