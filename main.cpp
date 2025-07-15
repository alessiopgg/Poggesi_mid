#include <iostream>
#include "dataset.h"
#include "kmeans_seq.h"
#include "kmeans_omp.h"

int main() {
    Dataset ds;

    // Percorso del file CSV
    std::string filename = "C:/Users/Alessio/Documents/Projects/Poggesi_mid/data/insane_dataset.csv";
    int k = 10;     // numero di centroidi

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

    // Avvia l'algoritmo di clustering
    model.fit(k);

    // Mostra i centroidi finali
    model.print_centroids();

    return 0;
}
