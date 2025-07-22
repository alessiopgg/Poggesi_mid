cmake_minimum_required(VERSION 3.23)
project(Poggesi_mid)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Modalità di build Debug con simboli e senza ottimizzazioni (utile per VTune)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Flags specifici per MSVC e GCC/Clang (Windows/Linux compatibilità)
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()

# Flags per Release – ATTENZIONE: qui attiviamo ottimizzazioni e vettorizzazione
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ftree-vectorize -fopenmp")
endif()

# Cerca OpenMP
find_package(OpenMP REQUIRED)

# Includi la cartella "include" per i file header
include_directories(include)

# Lista dei sorgenti
set(SOURCES
        #main.cpp
        main_test.cpp
        base_kmeans.h
        kmeans_seq.cpp
        kmeans_seq.h
        kmeans_omp.cpp
        kmeans_omp.h
        dataset.cpp
        dataset.h)

# Crea l'eseguibile
add_executable(Poggesi_mid ${SOURCES})

# Collega OpenMP (se trovato)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Poggesi_mid PUBLIC OpenMP::OpenMP_CXX)
endif()
