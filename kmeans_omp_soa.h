#pragma once

#include "dataset.h"   // per PointsSoA e Point
#include "base_kmeans.h"
#include <vector>

class KMeansOpenMP_SoA : public BaseKMeans {
private:
    const PointsSoA& points;          // <-- PUNTI in SoA (vero)
    std::vector<Point> centroids;     // ok tenerli anche AoS per stampa/compatibilità
    std::vector<double> centroid_x;   // <-- centroidi in SoA
    std::vector<double> centroid_y;

    int k;
    std::vector<int> labels;
    int last_iters_ = 0;


public:
    KMeansOpenMP_SoA(const PointsSoA& input_points,
                     const std::vector<Point>& initial_centroids);

    void assign_clusters() override;
    void update_centroids() override;
    void fit(int k_) override;
    void print_centroids() const override;
    int get_last_iterations() const { return last_iters_; }
    const std::vector<Point>& get_centroids() const { return centroids; }
    const std::vector<int>&   get_labels()   const { return labels; }

};
