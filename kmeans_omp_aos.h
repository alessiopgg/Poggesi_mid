#pragma once
#include "base_kmeans.h"
#include <vector>
#include "point.h"

class KMeansOpenMP_AoS : public BaseKMeans {
public:
    KMeansOpenMP_AoS(const std::vector<Point>& input_points,
                     const std::vector<Point>& initial_centroids);

    void assign_clusters() override;
    void update_centroids() override;
    void fit(int k) override;
    void print_centroids() const override;
    int get_last_iterations() const { return last_iters_; }


private:
    std::vector<Point> points;
    std::vector<Point> centroids;
    std::vector<int> labels;
    int k;
    int last_iters_ = 0;

};
