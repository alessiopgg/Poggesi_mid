#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <numeric> // per std::accumulate
#include "dataset.h"
#include "kmeans_seq.h"
#include "kmeans_omp.h"

double get_time_in_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

std::vector<std::string> filenames = {
        "N10000_K2.csv", "N10000_K6.csv", "N10000_K10.csv", "N10000_K20.csv", "N10000_K50.csv", "N10000_K100.csv",
        "N50000_K2.csv", "N50000_K6.csv", "N50000_K10.csv", "N50000_K20.csv", "N50000_K50.csv", "N50000_K100.csv",
        "N100000_K2.csv", "N100000_K6.csv", "N100000_K10.csv", "N100000_K20.csv", "N100000_K50.csv", "N100000_K100.csv",
        "N300000_K2.csv", "N300000_K6.csv", "N300000_K10.csv", "N300000_K20.csv", "N300000_K50.csv", "N300000_K100.csv",
        "N500000_K2.csv", "N500000_K6.csv", "N500000_K10.csv", "N500000_K20.csv", "N500000_K50.csv", "N500000_K100.csv",
        "N1000000_K2.csv", "N1000000_K6.csv", "N1000000_K10.csv", "N1000000_K20.csv", "N1000000_K50.csv", "N1000000_K100.csv"
};

const int NUM_RUNS = 10;

int main() {
    std::string folder = "C:/Users/Alessio/Documents/Projects/Poggesi_mid/data/";
    std::string output_file = "risultati_tempi.csv";

    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Errore nell'apertura del file di output.\n";
        return 1;
    }

    out << "n_points,k,avg_seq_time_ms,avg_omp_time_ms\n";

    for (const auto& file : filenames) {
        std::cout << "Elaborazione file: " << file << "\n";

        int n_points, k;
        if (sscanf(file.c_str(), "N%d_K%d.csv", &n_points, &k) != 2) {
            std::cerr << "Formato nome file non riconosciuto: " << file << "\n";
            continue;
        }

        std::string full_path = folder + file;

        std::vector<double> seq_times, omp_times;

        for (int run = 0; run < NUM_RUNS; ++run) {
            Dataset ds_seq, ds_omp;

            if (!ds_seq.load_from_csv(full_path) || !ds_omp.load_from_csv(full_path)) {
                std::cerr << "Errore nel caricamento di " << file << "\n";
                break;
            }

            // Sequenziale
            ds_seq.init_centroids(k);
            KMeansSequential model_seq(ds_seq.get_points(), ds_seq.get_centroids());
            double start_seq = get_time_in_ms();
            model_seq.fit(k);
            double end_seq = get_time_in_ms();
            seq_times.push_back(end_seq - start_seq);

            // Parallelo
            ds_omp.init_centroids(k);
            KMeansOpenMP model_omp(ds_omp.get_points(), ds_omp.get_centroids());
            double start_omp = get_time_in_ms();
            model_omp.fit(k);
            double end_omp = get_time_in_ms();
            omp_times.push_back(end_omp - start_omp);
        }

        if (seq_times.size() == NUM_RUNS && omp_times.size() == NUM_RUNS) {
            double avg_seq = std::accumulate(seq_times.begin(), seq_times.end(), 0.0) / NUM_RUNS;
            double avg_omp = std::accumulate(omp_times.begin(), omp_times.end(), 0.0) / NUM_RUNS;

            out << n_points << "," << k << "," << avg_seq << "," << avg_omp << "\n";
        } else {
            std::cerr << "Numero di run incompleto per " << file << "\n";
        }
    }

    out.close();
    std::cout << "\n✅ File risultati scritto in: " << output_file << "\n";
    return 0;
}
