#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <omp.h>

#include "dataset.h"
#include "kmeans_seq.h"
#include "kmeans_omp_aos.h"
#include "kmeans_omp_soa.h"

// =========================
// HIGH-RES OPENMP TIMER
// =========================
double get_time_in_ms() {
    return omp_get_wtime() * 1000.0;
}

// =========================
// STATISTICS STRUCT
// =========================
struct Stats {
    double mean;
    double stddev;
    double min;
    double max;
};

Stats compute_stats(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x;
    double mean = sum / v.size();

    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    var /= v.size();
    double stddev = std::sqrt(var);

    double minv = *std::min_element(v.begin(), v.end());
    double maxv = *std::max_element(v.begin(), v.end());

    return {mean, stddev, minv, maxv};
}

// =========================
// PARALLELIZATION PARAMETERS
// =========================
std::vector<std::string> SCHEDULES = {"static", "dynamic"};
std::vector<int> CHUNKS = {0, 32, 128};
std::vector<int> THREAD_LIST = {1, 2, 4, 8, 16, 32};

std::vector<std::string> filenames = {
        "N10000_K6.csv", "N10000_K10.csv", "N10000_K20.csv",
        "N50000_K6.csv", "N50000_K10.csv", "N50000_K20.csv",
        "N100000_K6.csv", "N100000_K10.csv", "N100000_K20.csv",
        "N300000_K6.csv", "N300000_K10.csv", "N300000_K20.csv",
        "N500000_K6.csv", "N500000_K10.csv", "N500000_K20.csv",
        "N1000000_K6.csv", "N1000000_K10.csv", "N1000000_K20.csv"
};

const int NUM_RUNS = 5;
const int WARMUP_RUNS = 1;

// =========================
// MAIN FUNCTION
// =========================
int main() {
    std::string folder = "C:/Users/Alessio/Documents/Projects/Poggesi_mid/data/";
    std::string output_file = "risultati_openmp_aos_vs_soa.csv";

    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Errore apertura file output\n";
        return 1;
    }

    // CSV HEADER
    out << "layout,n_points,k,threads,schedule,chunk,"
           "seq_mean,seq_std,seq_min,seq_max,"
           "omp_mean,omp_std,omp_min,omp_max\n";

    // MEMORY LAYOUTS TO TEST
    std::vector<std::string> LAYOUTS = {"AoS", "SoA"};

    for (const auto& file : filenames) {
        std::cout << "\n=== FILE: " << file << " ===\n";

        int n_points, k;
        if (sscanf(file.c_str(), "N%d_K%d.csv", &n_points, &k) != 2) {
            std::cerr << "Errore parsing nome file!\n";
            continue;
        }

        std::string full_path = folder + file;

        for (const auto& layout : LAYOUTS) {
            for (int threads : THREAD_LIST) {
                omp_set_num_threads(threads);

                for (const auto& sched : SCHEDULES) {
                    for (int chunk : CHUNKS) {

                        std::cout << " -> layout=" << layout
                                  << ", threads=" << threads
                                  << ", sched=" << sched
                                  << ", chunk=" << chunk << "\n";

                        std::vector<double> seq_times;
                        std::vector<double> omp_times;

                        omp_sched_t kind = omp_sched_static;
                        if (sched == "dynamic") kind = omp_sched_dynamic;
                        // se poi aggiungi "guided": kind = omp_sched_guided;

                        omp_set_schedule(kind, chunk); // chunk può essere 0: significa "default" per quel kind


                        // MULTIPLE RUNS (WITH WARM-UP)
                        for (int run = 0; run < NUM_RUNS; ++run) {

                            Dataset ds_seq, ds_omp;
                            if (!ds_seq.load_from_csv(full_path) ||
                                !ds_omp.load_from_csv(full_path)) {
                                std::cerr << "Errore caricamento dataset\n";
                                break;
                            }

                            // SEQUENTIAL
                            ds_seq.init_centroids(k);
                            KMeansSequential model_seq(
                                    ds_seq.get_points(),
                                    ds_seq.get_centroids()
                            );

                            double t0 = get_time_in_ms();
                            model_seq.fit(k);
                            double t1 = get_time_in_ms();

                            // PARALLEL
                            ds_omp.init_centroids(k);
                            double p0, p1;

                            if (layout == "AoS") {
                                KMeansOpenMP_AoS model_omp(
                                        ds_omp.get_points(),
                                        ds_omp.get_centroids()
                                );
                                p0 = get_time_in_ms();
                                model_omp.fit(k);
                                p1 = get_time_in_ms();
                            } else { // SoA
                                KMeansOpenMP_SoA model_omp(
                                        ds_omp.get_points(),
                                        ds_omp.get_centroids()
                                );
                                p0 = get_time_in_ms();
                                model_omp.fit(k);
                                p1 = get_time_in_ms();
                            }

                            if (run >= WARMUP_RUNS) {
                                seq_times.push_back(t1 - t0);
                                omp_times.push_back(p1 - p0);
                            }
                        }

                        // WRITE RESULTS
                        if (seq_times.size() == NUM_RUNS - WARMUP_RUNS &&
                            omp_times.size() == NUM_RUNS - WARMUP_RUNS) {

                            Stats seq_s = compute_stats(seq_times);
                            Stats omp_s = compute_stats(omp_times);

                            out << layout << ","
                                << n_points << ","
                                << k << ","
                                << threads << ","
                                << sched << ","
                                << chunk << ","
                                << seq_s.mean << ","
                                << seq_s.stddev << ","
                                << seq_s.min << ","
                                << seq_s.max << ","
                                << omp_s.mean << ","
                                << omp_s.stddev << ","
                                << omp_s.min << ","
                                << omp_s.max << "\n";
                        }
                    }
                }
            }
        }
    }

    out.close();
    std::cout << "\n✅ File scritto: " << output_file << "\n";
    return 0;
}
