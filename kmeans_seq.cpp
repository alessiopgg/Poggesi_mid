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
    for (size_t i = 0; i < points.size(); ++i) {//per ogni punto del dataset
        double min_dist = std::numeric_limits<double>::max();//inizializzo variabile per tenere conto della distanza punto-centroide
        int best_cluster = -1;//mi tengo l'indice del centroide

        for (int j = 0; j < k; ++j) {//confronto la distanza dal punto ai centroidi
            double dx = points[i].x - centroids[j].x;
            double dy = points[i].y - centroids[j].y;
            double dist = dx * dx + dy * dy; // distanza al quadrato

            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;//ottengo l'indice del centroide piu vicino
            }
        }

        labels[i] = best_cluster;//assegno al punto un centroide
    }
}

// Ricalcola i centroidi come media dei punti assegnati
void KMeansSequential::update_centroids() {
    std::vector<Point> new_centroids(k, {0.0, 0.0});//vettore per tenere i centroidi
    std::vector<int> count(k, 0);// tiene quanti punti sono stati assegnati al centroide k

    for (size_t i = 0; i < points.size(); ++i) {//per tutti i punti del dataset
        int cluster = labels[i]; // a quale cluster appartiene il punto i
        new_centroids[cluster].x += points[i].x;//sommo le componenti
        new_centroids[cluster].y += points[i].y;
        count[cluster]++;//incremento il numero di punti associati al centroide
    }

    for (int j = 0; j < k; ++j) {//per ogni centroide calcolo la media
        if (count[j] > 0) {
            new_centroids[j].x /= count[j];
            new_centroids[j].y /= count[j];
        }
    }

    centroids = new_centroids;//assegno il nuovo centroide
}

// Esegue tutto il ciclo K-means
void KMeansSequential::fit(int k_) {
    k = k_;

    bool converged = false;

    while (!converged) {
        assign_clusters();

        std::vector<Point> old_centroids = centroids;

        update_centroids();
        std::cout << "iter\n";

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

// Stampa a schermo i centroidi finali
void KMeansSequential::print_centroids() const {
    std::cout << "Centroidi finali:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << " C" << i << " = (" << centroids[i].x << ", " << centroids[i].y << ")\n";
    }
}
