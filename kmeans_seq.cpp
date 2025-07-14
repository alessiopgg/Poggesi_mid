#include "kmeans_seq.h"
#include <random>
#include <iostream>
#include <limits>
#include <cmath>

KMeansSequential::KMeansSequential(const std::vector<Point>& input_points, const std::vector<Point>& initial_centroids)
        : points(input_points),
          centroids(initial_centroids),
          k(static_cast<int>(initial_centroids.size())),
          labels(input_points.size(), -1) {}


// Assegna ogni punto al centroide più vicino
void KMeansSequential::assign_clusters() {
    for (size_t i = 0; i < points.size(); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = -1;

        for (int j = 0; j < k; ++j) {
            double dx = points[i].x - centroids[j].x;
            double dy = points[i].y - centroids[j].y;
            double dist = dx * dx + dy * dy; // distanza al quadrato

            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }

        labels[i] = best_cluster;
    }
}

// Ricalcola i centroidi come media dei punti assegnati
void KMeansSequential::update_centroids() {
    std::vector<Point> new_centroids(k, {0.0, 0.0});
    std::vector<int> count(k, 0);

    for (size_t i = 0; i < points.size(); ++i) {
        int cluster = labels[i];
        new_centroids[cluster].x += points[i].x;
        new_centroids[cluster].y += points[i].y;
        count[cluster]++;
    }

    for (int j = 0; j < k; ++j) {
        if (count[j] > 0) {
            new_centroids[j].x /= count[j];
            new_centroids[j].y /= count[j];
        }
    }

    centroids = new_centroids;
}

// Esegue tutto il ciclo K-means
void KMeansSequential::fit(int k, int max_iters) {
    for (int iter = 0; iter < max_iters; ++iter) {
        assign_clusters();
        update_centroids();
    }
}

// Stampa a schermo i centroidi finali
void KMeansSequential::print_centroids() const {
    std::cout << "Centroidi finali:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << " C" << i << " = (" << centroids[i].x << ", " << centroids[i].y << ")\n";
    }
}
