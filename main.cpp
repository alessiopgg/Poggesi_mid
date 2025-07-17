#include <iostream>
#include "dataset.h"
#include "kmeans_seq.h"
#include "kmeans_omp.h"
#include <sys/time.h>

double get_time_in_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}


int main() {
    Dataset ds;

    // Percorso del file CSV
    std::string filename = "C:/Users/Alessio/Documents/Projects/Poggesi_mid/data/N1000000_K12.csv";
    int k = 200;     // numero di centroidi

    // Carica i punti
    if (!ds.load_from_csv(filename)) {
        std::cerr << "Errore durante la lettura del file.\n";
        return 1;
    }

    // Inizializza i centroidi casualmente (usando il Dataset)
    ds.init_centroids(k);
    ds.print_centroids();


    // Costruisce l'oggetto KMeans usando i punti e i centroidi iniziali
    //KMeansSequential model(ds.get_points(), ds.get_centroids());

    KMeansOpenMP model(ds.get_points(), ds.get_centroids());

    double start_time = get_time_in_ms();
    model.fit(k);
    double end_time = get_time_in_ms();


    // Mostra i centroidi finali
    model.print_centroids();


    std::cout << "Tempo di esecuzione (wall-clock): "
              << (end_time - start_time) << " ms\n";

    return 0;
}
