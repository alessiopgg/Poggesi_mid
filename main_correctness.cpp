#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <omp.h>

#include "dataset.h"
#include "kmeans_seq.h"
#include "kmeans_omp_aos.h"
#include "kmeans_omp_soa.h"

struct Stats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
};

static Stats stats_of(const std::vector<double>& v) {
    Stats s;
    if (v.empty()) return s;
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();

    double var = 0.0;
    for (double x : v) var += (x - s.mean) * (x - s.mean);
    var /= (double)v.size();
    s.stddev = std::sqrt(var);
    return s;
}

static double sse_aos(const std::vector<Point>& pts,
                      const std::vector<Point>& cents,
                      const std::vector<int>& lab) {
    double sse = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        int c = lab[i];
        double dx = pts[i].x - cents[c].x;
        double dy = pts[i].y - cents[c].y;
        sse += dx * dx + dy * dy;
    }
    return sse;
}

static bool rel_close(double ref, double val, double rel_eps) {
    double denom = std::max(1.0, std::abs(ref));
    double rel = std::abs(ref - val) / denom;
    return rel <= rel_eps;
}

static double now_ms() {
    return omp_get_wtime() * 1000.0;
}

int main() {
    // ====== CONFIG ======
    std::string folder = "C:/Users/Alessio/Documents/Projects/Poggesi_mid/data/";
    std::string file   = "N500000_K10.csv";
    int k = 10;

    int threads = 8;                 // scegli tu (o metti 0 per non forzare)
    int warmup_runs = 1;
    int runs = 5;                    // runs misurati
    double rel_eps = 1e-9;           // tolleranza (1e-9 è già larghissima rispetto a 1e-16)

    // =====================
    Dataset ds;
    if (!ds.load_from_csv(folder + file)) {
        std::cerr << "ERROR: cannot load dataset: " << (folder + file) << "\n";
        return 1;
    }

    ds.init_centroids(k, 123); // init IDENTICO per tutti
    const auto& initC = ds.get_centroids();

    if (threads > 0) omp_set_num_threads(threads);

    // ====== WARMUP (non misurato) ======
    for (int w = 0; w < warmup_runs; ++w) {
        KMeansSequential tmp(ds.get_points(), initC);
        tmp.fit(k);
        KMeansOpenMP_AoS tmp2(ds.get_points(), initC);
        tmp2.fit(k);
        KMeansOpenMP_SoA tmp3(ds.get_points_soa(), initC);
        tmp3.fit(k);
    }

    // ====== MEASURED RUNS ======
    std::vector<double> t_seq, t_aos, t_soa;
    std::vector<int> it_seq, it_aos, it_soa;
    omp_set_schedule(omp_sched_static, 0);


    for (int r = 0; r < runs; ++r) {
        // --- SEQ
        double t0 = now_ms();
        KMeansSequential seq(ds.get_points(), initC);
        seq.fit(k);
        double t1 = now_ms();
        double sse_seq = sse_aos(ds.get_points(), seq.get_centroids(), seq.get_labels());

        t_seq.push_back(t1 - t0);
        it_seq.push_back(seq.get_last_iterations());

        // --- OMP AoS
        t0 = now_ms();
        KMeansOpenMP_AoS omp_aos(ds.get_points(), initC);
        omp_aos.fit(k);
        t1 = now_ms();
        double sse_aos_v = sse_aos(ds.get_points(), omp_aos.get_centroids(), omp_aos.get_labels());

        t_aos.push_back(t1 - t0);
        it_aos.push_back(omp_aos.get_last_iterations());

        // --- OMP SoA
        t0 = now_ms();
        KMeansOpenMP_SoA omp_soa(ds.get_points_soa(), initC);
        omp_soa.fit(k);
        t1 = now_ms();
        double sse_soa_v = sse_aos(ds.get_points(), omp_soa.get_centroids(), omp_soa.get_labels());

        t_soa.push_back(t1 - t0);
        it_soa.push_back(omp_soa.get_last_iterations());

        // ===== correctness check (per run) =====
        bool ok_aos = rel_close(sse_seq, sse_aos_v, rel_eps);
        bool ok_soa = rel_close(sse_seq, sse_soa_v, rel_eps);

        std::cout << "Run " << (r+1) << "/" << runs
                  << " | it: SEQ=" << it_seq.back()
                  << " OMP_AoS=" << it_aos.back()
                  << " OMP_SoA=" << it_soa.back()
                  << " | SSE: SEQ=" << sse_seq
                  << " AOS=" << sse_aos_v << (ok_aos ? " OK" : " FAIL")
                  << " SOA=" << sse_soa_v << (ok_soa ? " OK" : " FAIL")
                  << "\n";

        if (!ok_aos || !ok_soa) {
            std::cout << "NOTE: SSE mismatch beyond tolerance rel_eps=" << rel_eps << "\n";
        }
    }

    // ====== SUMMARY ======
    auto st_seq = stats_of(t_seq);
    auto st_aos = stats_of(t_aos);
    auto st_soa = stats_of(t_soa);

    std::cout << "\n=== Timing summary (ms) ===\n";
    std::cout << "Threads: " << (threads > 0 ? std::to_string(threads) : std::string("default")) << "\n";
    std::cout << "Dataset: " << file << " | K=" << k << " | runs=" << runs << "\n\n";

    auto print_stats = [](const char* name, const Stats& s) {
        std::cout << name
                  << " mean=" << s.mean
                  << " std=" << s.stddev
                  << " min=" << s.min
                  << " max=" << s.max
                  << "\n";
    };

    print_stats("SEQ    ", st_seq);
    print_stats("OMP_AoS", st_aos);
    print_stats("OMP_SoA", st_soa);

    // Speedup (media)
    if (st_aos.mean > 0 && st_soa.mean > 0) {
        std::cout << "\n=== Speedup (mean) ===\n";
        std::cout << "SEQ / OMP_AoS = " << (st_seq.mean / st_aos.mean) << "x\n";
        std::cout << "SEQ / OMP_SoA = " << (st_seq.mean / st_soa.mean) << "x\n";
    }

    return 0;
}
