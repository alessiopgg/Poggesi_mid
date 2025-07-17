#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include "dataset.h"
#include "kmeans_seq.h"
#include "kmeans_omp.h"

double get_time_in_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// 👇 Qui metti la lista dei file da processare
std::vector<std::string> filenames = {
        "N10000_K2.csv", "N10000_K2_moons.csv", "N10000_K2_circles.csv", "N10000_K4.csv", "N10000_K6.csv", "N10000_K8.csv", "N10000_K10.csv", "N10000_K12.csv",
        "N50000_K2.csv", "N50000_K2_moons.csv", "N50000_K2_circles.csv", "N50000_K4.csv", "N50000_K6.csv", "N50000_K8.csv", "N50000_K10.csv", "N50000_K12.csv",
        "N100000_K2.csv", "N100000_K2_moons.csv", "N100000_K2_circles.csv", "N100000_K4.csv", "N100000_K6.csv", "N100000_K8.csv", "N100000_K10.csv", "N100000_K12.csv",
        "N300000_K2.csv", "N300000_K2_moons.csv", "N300000_K2_circles.csv", "N300000_K4.csv", "N300000_K6.csv", "N300000_K8.csv", "N300000_K10.csv", "N300000_K12.csv",
        "N1000000_K2.csv", "N1000000_K2_moons.csv", "N1000000_K2_circles.csv", "N1000000_K4.csv", "N1000000_K6.csv", "N1000000_K8.csv", "N1000000_K10.csv", "N1000000_K12.csv"
};

int main() {
    std::string folder = "C:/Users/Alessio/Documents/Projects/Poggesi_mid/data/";
    std::string output_file = "risultati_tempi.csv";

    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Errore nell'apertura del file di output.\n";
        return 1;
    }

    out << "n_points,k,seq_time_ms,omp_time_ms\n";

    for (const auto& file : filenames) {
        std::cout << "Elaborazione file: " << file << "\n";

        // Estrai n e k dal nome file
        int n_points, k;
        if (sscanf(file.c_str(), "N%d_K%d.csv", &n_points, &k) != 2) {
            std::cerr << "Formato nome file non riconosciuto: " << file << "\n";
            continue;
        }

        Dataset ds;
        std::string full_path = folder + file;
        if (!ds.load_from_csv(full_path)) {
            std::cerr << "Errore nel caricamento di " << file << "\n";
            continue;
        }

        // Sequenziale
        ds.init_centroids(k);
        KMeansSequential model_seq(ds.get_points(), ds.get_centroids());
        double start_seq = get_time_in_ms();
        model_seq.fit(k);
        double end_seq = get_time_in_ms();

        // Parallelo
        ds.init_centroids(k);
        KMeansOpenMP model_omp(ds.get_points(), ds.get_centroids());
        double start_omp = get_time_in_ms();
        model_omp.fit(k);
        double end_omp = get_time_in_ms();

        out << n_points << "," << k << ","
            << (end_seq - start_seq) << ","
            << (end_omp - start_omp) << "\n";
    }

    out.close();
    std::cout << "\n✅ File risultati scritto in: " << output_file << "\n";
    return 0;
}
