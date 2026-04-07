K-Means Clustering: Sequential and OpenMP Versions
This project implements and benchmarks a sequential C++ version of K-Means clustering and two parallel OpenMP variants based on different memory layouts:
Sequential baseline
OpenMP AoS (Array of Structures)
OpenMP SoA (Structure of Arrays)
The goal is to compare correctness and performance, with a particular focus on:
speedup with increasing thread counts
impact of memory layout (AoS vs SoA)
effect of OpenMP scheduling policy and chunk size
The implementation follows a shared abstract interface (`BaseKMeans`) and separates dataset loading, algorithm implementations, benchmarking, correctness validation, and a small scheduling demo trace.
Project structure
`base_kmeans.h` – common interface for all K-Means variants
`point.h` – definition of `Point` and `PointsSoA`
`dataset.h / dataset.cpp` – CSV loader and centroid initialization
`kmeans_seq.h / kmeans_seq.cpp` – sequential implementation
`kmeans_omp_aos.h / kmeans_omp_aos.cpp` – OpenMP implementation using AoS layout
`kmeans_omp_soa.h / kmeans_omp_soa.cpp` – OpenMP implementation using SoA layout
`main.cpp` – benchmark driver
`main_correctness.cpp` – correctness validation using SSE comparison
`CMakeLists.txt` – build configuration
Algorithm overview
Each implementation performs the standard K-Means loop:
Assignment step  
Each point is assigned to the nearest centroid using squared Euclidean distance.
Update step  
Each centroid is recomputed as the mean of the points assigned to its cluster.
Stopping criterion  
The algorithm stops when either:
the maximum centroid shift becomes smaller than `1e-6`, or
`600` iterations are reached.
Parallelization strategy
Both OpenMP versions parallelize the most expensive parts of K-Means:
Assignment phase: parallelized over points with `#pragma omp parallel for schedule(runtime)`
Centroid accumulation: implemented with thread-local partial sums and counts to avoid contention
Centroid finalization: parallelized over clusters
A direct OpenMP reduction is not used for centroid updates because cluster assignments cause indirect indexed updates. Instead, each thread accumulates local sums and counts and a merge step combines them afterward.
AoS vs SoA
AoS
Points are stored as:
```cpp
struct Point { double x; double y; };
std::vector<Point>
```
This layout is simple and intuitive, but memory accesses may be less efficient when only coordinates are needed.
SoA
Points are stored as two separate arrays:
```cpp
std::vector<double> x;
std::vector<double> y;
```
This improves memory locality and is generally more cache-friendly during distance computations. In the experiments, the SoA variant is the best-performing configuration.
Build requirements
C++17
CMake >= 3.23
OpenMP-enabled compiler
CMake configuration
The project defines three executables:
`Poggesi_mid` → benchmark program
`Poggesi_mid_correctness` → correctness checker
Important notes from `CMakeLists.txt`:
default build type: Debug
recommended build type for benchmarks: Release
on non-MSVC toolchains, Release uses:
```txt
-O3 -march=native -ftree-vectorize -fopenmp
```
Build instructions
Example:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
Dataset format
Datasets are 2D CSV files with one point per row:
```csv
x,y
12.4,8.9
10.1,7.2
...
```
The loader accepts both:
CSV files with header
CSV files without header
The code loads the same dataset into both:
AoS representation
SoA representation
This ensures a fair comparison between the two parallel variants.
Important runtime settings
The benchmark driver in `main.cpp` currently uses a hard-coded Windows path for the input datasets:
```cpp
std::string folder = "C:/Users/Alessio/Documents/Projects/Poggesi_mid/data/";
```
If you run the project on another machine, this path must be updated.
Benchmark parameters used in the code
schedules: `static`, `dynamic`
chunk sizes: `0`, `1`, `32`, `128`
thread counts: `1`, `2`, `4`, `8`, `16`, `32`
measured runs: `5`
warm-up runs discarded: `1`
Datasets included in the benchmark campaign
The benchmark is configured for the following input sizes and cluster counts:
`N = 10k, 50k, 100k, 300k, 500k, 1M`
`K = 6, 10, 20`
The filenames are expected in the form:
```txt
N10000_K6.csv
N500000_K10.csv
N1000000_K20.csv
```
How to run
1. Benchmark
Run the benchmark executable to generate the CSV with timing statistics:
```bash
./Poggesi_mid
```
Output file:
```txt
risultati_openmp_aos_vs_soa.csv
```
The CSV contains, for each configuration:
layout
number of points
number of clusters
threads
schedule
chunk
sequential timing statistics
OpenMP timing statistics
iteration statistics
2. Correctness validation
Run:
```bash
./Poggesi_mid_correctness
```
This program compares:
Sequential
OpenMP AoS
OpenMP SoA
Correctness is checked by comparing the final SSE against the sequential reference using a relative tolerance.

Key implementation choices
deterministic centroid initialization with a seed
same initialization shared between sequential and parallel runs for fair comparison
timing based on `omp_get_wtime()`
compute time measured only on the `fit()` phase
CSV loading and centroid initialization excluded from timed region
empty clusters handled by keeping the previous centroid
Main findings
According to the report and benchmark setup:
SoA is generally faster and more robust than AoS
static scheduling is consistently the best choice for this workload
dynamic scheduling, especially with very small chunk sizes, introduces large overhead
scaling improves with larger datasets
the best speedup reported is approximately 4.6x on the largest dataset using the SoA version with 8 threads
Limitations
benchmark path is hard-coded and machine-dependent
evaluation is limited to synthetic 2D datasets
experiments are performed on a single shared-memory CPU machine
SIMD is not explicitly forced for the distance loop; the implementation mainly relies on compiler optimization and data layout improvements
Suggested future improvements
remove hard-coded dataset paths and pass them as command-line arguments
support higher-dimensional datasets
improve centroid merge efficiency
investigate more aggressive SIMD/vectorization strategies
test on additional hardware and memory configurations
Author
Alessio Poggesi  
Project for the course Parallel Programming.
