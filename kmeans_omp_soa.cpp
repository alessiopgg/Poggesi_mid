#include "kmeans_omp_soa.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <omp.h>

KMeansOpenMP_SoA::KMeansOpenMP_SoA(const PointsSoA& input_points,
                                   const std::vector<Point>& initial_centroids)
        : points(input_points),
          centroids(initial_centroids),
          centroid_x(initial_centroids.size()),
          centroid_y(initial_centroids.size()),
          k(static_cast<int>(initial_centroids.size())),
          labels(input_points.size(), -1)
{
    for (int j = 0; j < k; ++j) {
        centroid_x[j] = centroids[j].x;
        centroid_y[j] = centroids[j].y;
    }
}


// FASE 1 – Parallelizzata con OpenMP + Vettorizzata (SIMD)
void KMeansOpenMP_SoA::assign_clusters() {
#pragma omp parallel for schedule(runtime) default(none) shared(labels) shared(points, centroid_x, centroid_y, k)
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        double px = points.x[i];
        double py = points.y[i];

        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = -1;

        // Nota: il pragma omp simd con "min_dist/best_cluster" è delicato.
        // Per ora lasciamo il loop normale (corretto e stabile).
        for (int j = 0; j < k; ++j) {
            double dx = px - centroid_x[j];
            double dy = py - centroid_y[j];
            double dist = dx * dx + dy * dy;

            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }

        labels[i] = best_cluster;
    }
}


// FASE 2 – Parallelizzazione multithread, non SIMD (per via di accessi indiretti)
void KMeansOpenMP_SoA::update_centroids() {
    int num_threads = omp_get_max_threads();

    std::vector<std::vector<double>> local_sum_x(num_threads, std::vector<double>(k, 0.0));
    std::vector<std::vector<double>> local_sum_y(num_threads, std::vector<double>(k, 0.0));
    std::vector<std::vector<int>>    local_count(num_threads, std::vector<int>(k, 0));

#pragma omp parallel default(none) shared(local_sum_x, local_sum_y, local_count, labels, points, k)
    {
        int tid = omp_get_thread_num();
        auto& sx = local_sum_x[tid];
        auto& sy = local_sum_y[tid];
        auto& ct = local_count[tid];

#pragma omp for schedule(runtime)
        for (size_t i = 0; i < points.size(); ++i) {
            int c = labels[i];
            sx[c] += points.x[i];
            sy[c] += points.y[i];
            ct[c] += 1;
        }
    }

    std::vector<double> sum_x(k, 0.0), sum_y(k, 0.0);
    std::vector<int> count(k, 0);

    for (int t = 0; t < num_threads; ++t) {
        for (int j = 0; j < k; ++j) {
            sum_x[j] += local_sum_x[t][j];
            sum_y[j] += local_sum_y[t][j];
            count[j] += local_count[t][j];
        }
    }

#pragma omp parallel for schedule(runtime) default(none) shared(sum_x, sum_y, count, k, centroid_x, centroid_y)
    for (int j = 0; j < k; ++j) {
        if (count[j] > 0) {
            centroid_x[j] = sum_x[j] / count[j];
            centroid_y[j] = sum_y[j] / count[j];
        }
    }

    // aggiorno anche la versione AoS per print/convergenza
    for (int j = 0; j < k; ++j) {
        centroids[j].x = centroid_x[j];
        centroids[j].y = centroid_y[j];
    }
}


void KMeansOpenMP_SoA::fit(int k_) {
    k = k_;

    const int max_iters = 600;   // <-- alzato
    const double eps = 1e-6;
    const double eps2 = eps * eps;

    last_iters_ = 0;

    for (int it = 0; it < max_iters; ++it) {
        assign_clusters();

        std::vector<double> old_x = centroid_x;
        std::vector<double> old_y = centroid_y;

        update_centroids();

        double max_shift = 0.0;
        for (int j = 0; j < k; ++j) {
            double dx = centroid_x[j] - old_x[j];
            double dy = centroid_y[j] - old_y[j];
            double shift = dx * dx + dy * dy;
            if (shift > max_shift) max_shift = shift;
        }

        last_iters_ = it + 1;          // <-- quante iterazioni ho fatto davvero
        if (max_shift < eps2) break;
    }
}




void KMeansOpenMP_SoA::print_centroids() const {
    std::cout << "Centroidi finali (OpenMP):\n";
    for (int i = 0; i < k; ++i) {
        std::cout << " C" << i << " = (" << centroids[i].x << ", " << centroids[i].y << ")\n";
    }
}
