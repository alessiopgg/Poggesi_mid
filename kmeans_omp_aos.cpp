#include "kmeans_omp_aos.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <omp.h>

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------
KMeansOpenMP_AoS::KMeansOpenMP_AoS(const std::vector<Point>& input_points,
                                   const std::vector<Point>& initial_centroids)
        : points(input_points),
          centroids(initial_centroids),
          k(static_cast<int>(initial_centroids.size())),
          labels(input_points.size(), -1) {}


// ------------------------------------------------------------
// Phase 1: Assignment step (parallel over points)
// ------------------------------------------------------------
void KMeansOpenMP_AoS::assign_clusters() {
#pragma omp parallel for schedule(runtime) default(none) shared(points, centroids, labels, k)
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const Point& p = points[i];

        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = -1;

        for (int j = 0; j < k; ++j) {
            double dx = p.x - centroids[j].x;
            double dy = p.y - centroids[j].y;
            double dist = dx * dx + dy * dy;
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels[i] = best_cluster;
    }
}


// ------------------------------------------------------------
// Phase 2: Centroid update (thread-local sums + merge)
// ------------------------------------------------------------
void KMeansOpenMP_AoS::update_centroids() {
    // Determine real number of threads used
    int num_threads = 1;
#pragma omp parallel default(none) shared(num_threads)
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }

    // Thread-local accumulators
    std::vector<std::vector<double>> local_sum_x(num_threads, std::vector<double>(k, 0.0));
    std::vector<std::vector<double>> local_sum_y(num_threads, std::vector<double>(k, 0.0));
    std::vector<std::vector<int>>    local_count(num_threads, std::vector<int>(k, 0));

#pragma omp parallel default(none) shared(local_sum_x, local_sum_y, local_count, points, labels, k)
    {
        int tid = omp_get_thread_num();
        auto& sx = local_sum_x[tid];
        auto& sy = local_sum_y[tid];
        auto& ct = local_count[tid];

#pragma omp for schedule(runtime)
        for (size_t i = 0; i < points.size(); ++i) {
            int c = labels[i];
            sx[c] += points[i].x;
            sy[c] += points[i].y;
            ct[c] += 1;
        }
    }

    // Merge thread-local results
    std::vector<double> sum_x(k, 0.0), sum_y(k, 0.0);
    std::vector<int> count(k, 0);

    for (int t = 0; t < num_threads; ++t) {
        for (int j = 0; j < k; ++j) {
            sum_x[j] += local_sum_x[t][j];
            sum_y[j] += local_sum_y[t][j];
            count[j] += local_count[t][j];
        }
    }

    // Update centroids in-place (no full vector assignment)
#pragma omp parallel for schedule(runtime) default(none) shared(centroids, sum_x, sum_y, count, k)
    for (int j = 0; j < k; ++j) {
        if (count[j] > 0) {
            centroids[j].x = sum_x[j] / count[j];
            centroids[j].y = sum_y[j] / count[j];
        }
        // else: cluster empty => keep previous centroid (common choice)
    }
}


// ------------------------------------------------------------
// Fit with max iterations + epsilon (stable for benchmarks)
// ------------------------------------------------------------
void KMeansOpenMP_AoS::fit(int k_) {
    k = k_;

    const int max_iters = 600;   // <-- alzato
    const double eps = 1e-6;
    const double eps2 = eps * eps;

    last_iters_ = 0;

    for (int it = 0; it < max_iters; ++it) {
        assign_clusters();

        std::vector<Point> old = centroids;
        update_centroids();

        double max_shift = 0.0;
        for (int j = 0; j < k; ++j) {
            double dx = centroids[j].x - old[j].x;
            double dy = centroids[j].y - old[j].y;
            double shift = dx * dx + dy * dy;
            if (shift > max_shift) max_shift = shift;
        }

        last_iters_ = it + 1;          // <-- quante iterazioni ho fatto davvero
        if (max_shift < eps2) break;
    }
}



// ------------------------------------------------------------
// Print
// ------------------------------------------------------------
void KMeansOpenMP_AoS::print_centroids() const {
    std::cout << "Centroidi finali (OpenMP AoS):\n";
    for (int i = 0; i < k; ++i) {
        std::cout << " C" << i << " = (" << centroids[i].x << ", " << centroids[i].y << ")\n";
    }
}
