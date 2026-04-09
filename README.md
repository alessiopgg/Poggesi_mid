# K-Means Clustering with OpenMP

This project implements and compares three versions of K-Means clustering in C++17:

- **Sequential baseline**
- **OpenMP AoS** (Array of Structures)
- **OpenMP SoA** (Structure of Arrays)

The repository includes:

- the full source code
- the synthetic CSV datasets used for the benchmark
- a benchmark executable
- a correctness-check executable

The goal of the project is to evaluate the impact of parallelization strategy and memory layout on K-Means performance in a shared-memory setting.

---

## Repository Structure

```text
.
├── CMakeLists.txt
├── README.md
├── base_kmeans.h
├── dataset.cpp
├── dataset.h
├── point.h
├── kmeans_seq.cpp
├── kmeans_seq.h
├── kmeans_omp_aos.cpp
├── kmeans_omp_aos.h
├── kmeans_omp_soa.cpp
├── kmeans_omp_soa.h
├── main.cpp
├── main_correctness.cpp
├── generate_blobs
└── data/
    ├── N10000_K6.csv
    ├── N10000_K10.csv
    ├── ...
    └── N1000000_K20.csv
````

---

## Build Instructions

### Requirements

* CMake >= 3.23
* A C++17 compiler
* OpenMP support enabled in the selected toolchain

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Depending on the generator, the executables may be placed in:

* `build/`
* `build/Release/`
* `cmake-build-release/`

---

## Run Instructions

### 1. Benchmark

Runs the full benchmark campaign and writes the output CSV with timing statistics.

```bash
./Poggesi_mid
```

Current default output file:

```text
risultati_openmp_aos_vs_soa.csv
```

### 2. Correctness Check

Runs the sequential, OpenMP AoS and OpenMP SoA implementations on the same dataset and compares the final SSE and convergence behavior.

```bash
./Poggesi_mid_correctness
```

---

## Important Note on Paths

The current source code uses a **hardcoded absolute path** for the dataset folder in the benchmark and correctness executables.

If you clone the repository to a different location, update the `folder` variable in:

* `main.cpp`
* `main_correctness.cpp`

to match your local machine.

Example currently used in the code:

```cpp
std::string folder = "C:/Users/Alessio/Documents/Projects/Poggesi_mid/data/";
```

---

## Parameter Table

### Benchmark (`main.cpp`)

| Parameter              | Value                                           |
| ---------------------- | ----------------------------------------------- |
| Memory layouts         | `AoS`, `SoA`                                    |
| Dataset sizes          | `10000, 50000, 100000, 300000, 500000, 1000000` |
| Cluster counts         | `6, 10, 20`                                     |
| Number of datasets     | `18`                                            |
| Thread counts          | `1, 2, 4, 8, 16, 32`                            |
| Scheduling policies    | `static`, `dynamic`                             |
| Chunk sizes            | `0, 1, 32, 128`                                 |
| Runs per configuration | `5`                                             |
| Warm-up runs           | `1`                                             |
| Output file            | `risultati_openmp_aos_vs_soa.csv`               |

### Correctness Check (`main_correctness.cpp`)

| Parameter          | Value               |
| ------------------ | ------------------- |
| Default dataset    | `N500000_K10.csv`   |
| `k`                | `10`                |
| Threads            | `8`                 |
| Warm-up runs       | `1`                 |
| Measured runs      | `5`                 |
| Relative tolerance | `1e-9`              |
| Runtime schedule   | `static`, chunk `0` |

---

## CSV Input Format

Datasets are stored as CSV files with one point per row:

```csv
x,y
12.3,4.5
10.1,7.2
...
```

A header row is optional.

---

## Dataset Citation

The datasets used in this project are **synthetic 2D datasets** generated specifically for benchmarking K-Means.
They are not taken from an external real-world repository.

Generation details:

* generated through a dedicated Python script
* based on a `make_blobs`-style generator
* 2D Gaussian clusters
* deterministic seeds for reproducibility
* final benchmark campaign uses only the fixed subset of 18 datasets listed above

The repository already includes the datasets required by the benchmark under the `data/` folder.

---

## Notes

* The benchmark measures only the **K-Means compute phase** (`fit()`), excluding CSV loading and centroid initialization.
* The benchmark compares sequential and parallel runs using the **same deterministic seed** for centroid initialization within each run.
* OpenMP scheduling is controlled at runtime through `schedule(runtime)` inside the kernels and configured from the benchmark driver.
* Release builds are recommended for performance evaluation.

---

## License / Academic Context

This project was developed for the course **Parallel Programming for Machine Learning** as an academic comparison between sequential and OpenMP implementations of K-Means clustering.
---
