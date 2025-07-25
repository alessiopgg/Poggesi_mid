#include "kmeans_omp.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <omp.h>

KMeansOpenMP::KMeansOpenMP(const std::vector<Point>& input_points,
                           const std::vector<Point>& initial_centroids)
        : points(input_points),
          centroids(initial_centroids),
          k(static_cast<int>(initial_centroids.size())),
          labels(input_points.size(), -1) {}

// FASE 1 – Parallelizzata con OpenMP + Vettorizzata (SIMD)
void KMeansOpenMP::assign_clusters() {
    std::vector<double> centroid_x(k);
    std::vector<double> centroid_y(k);

    // Copia i centroidi separando x e y (Structure of Arrays)
    for (int j = 0; j < k; ++j) {
        centroid_x[j] = centroids[j].x;
        centroid_y[j] = centroids[j].y;
    }

#pragma omp parallel for default(none) shared(points, centroid_x, centroid_y, labels, k)
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = -1;
        double px = points[i].x;
        double py = points[i].y;

        // Vettorizzazione esplicita del ciclo su k (centroidi)
#pragma omp simd
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
void KMeansOpenMP::update_centroids() {
    std::vector<std::vector<Point>> local_sums;
    std::vector<std::vector<int>> local_counts;
    int num_threads = omp_get_max_threads();

    local_sums.resize(num_threads, std::vector<Point>(k, {0.0, 0.0}));
    local_counts.resize(num_threads, std::vector<int>(k, 0));

    int tid;

#pragma omp parallel default(none) shared(points, labels, local_sums, local_counts, k) private(tid)
    {
        tid = omp_get_thread_num();
        std::vector<Point>& sums = local_sums[tid];
        std::vector<int>& counts = local_counts[tid];

#pragma omp for
        for (size_t i = 0; i < points.size(); ++i) {
            int cluster = labels[i];
            sums[cluster].x += points[i].x;
            sums[cluster].y += points[i].y;
            counts[cluster]++;
        }
    }

    std::vector<Point> new_centroids(k, {0.0, 0.0});
    std::vector<int> count(k, 0);

    for (int t = 0; t < num_threads; ++t) {
        for (int j = 0; j < k; ++j) {
            new_centroids[j].x += local_sums[t][j].x;
            new_centroids[j].y += local_sums[t][j].y;
            count[j] += local_counts[t][j];
        }
    }

    // (Facoltativo) vettorizzazione del calcolo delle medie finali
#pragma omp parallel for simd default(none) shared(count, new_centroids, k)
    for (int j = 0; j < k; ++j) {
        if (count[j] > 0) {
            new_centroids[j].x /= count[j];
            new_centroids[j].y /= count[j];
        }
    }

    centroids = new_centroids;
}

void KMeansOpenMP::fit(int k_) {
    k = k_;
    bool converged = false;

    while (!converged) {
        assign_clusters();

        std::vector<Point> old_centroids = centroids;
        update_centroids();

        converged = true;
        for (int i = 0; i < k; ++i) {
            if (centroids[i].x != old_centroids[i].x ||
                centroids[i].y != old_centroids[i].y) {
                converged = false;
                break;
            }
        }
    }
}

void KMeansOpenMP::print_centroids() const {
    std::cout << "Centroidi finali (OpenMP):\n";
    for (int i = 0; i < k; ++i) {
        std::cout << " C" << i << " = (" << centroids[i].x << ", " << centroids[i].y << ")\n";
    }
}
