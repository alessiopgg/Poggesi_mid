cmake_minimum_required(VERSION 3.23)
project(Poggesi_mid)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# Build type default: Debug
# =========================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# =========================
# Compiler flags
# =========================
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
    # Se vuoi ottimizzazioni anche su MSVC, puoi aggiungere:
    # set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ftree-vectorize -fopenmp")
endif()

# =========================
# OpenMP
# =========================
find_package(OpenMP REQUIRED)

# =========================
# Include headers
# =========================
include_directories(include)

# =========================
# Core sources (NO main)
# =========================
set(CORE_SOURCES
        dataset.cpp
        kmeans_seq.cpp
        kmeans_omp_aos.cpp
        kmeans_omp_soa.cpp
        )

# =========================
# Executable: benchmark
# =========================
add_executable(Poggesi_mid
        main.cpp
        ${CORE_SOURCES}
        )

# =========================
# Executable: correctness
# =========================
add_executable(Poggesi_mid_correctness
        main_correctness.cpp
        ${CORE_SOURCES}
        )

# =========================
# Link OpenMP
# =========================
if(OpenMP_CXX_FOUND)
    target_link_libraries(Poggesi_mid PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(Poggesi_mid_correctness PUBLIC OpenMP::OpenMP_CXX)
endif()
