cmake_minimum_required(VERSION 3.23)
project(Poggesi_mid)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Cerca OpenMP
find_package(OpenMP REQUIRED)

# Includi la cartella "include" per i file header
include_directories(include)

# Aggiungi tutti i file sorgenti
set(SOURCES
        main.cpp
        base_kmeans.h kmeans_seq.cpp kmeans_seq.h kmeans_omp.cpp kmeans_omp.h dataset.cpp dataset.h)

# Crea l'eseguibile
add_executable(Poggesi_mid ${SOURCES})

# Collega OpenMP (se trovato)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Poggesi_mid PUBLIC OpenMP::OpenMP_CXX)
endif()
