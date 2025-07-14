#pragma once
#include "base_kmeans.h"

class KMeansSequential : public BaseKMeans {
public:
    KMeansSequential(const std::vector<Point>& input_points, const std::vector<Point>& initial_centroids);
    void assign_clusters() override;
    void update_centroids() override;
    void fit(int k, int max_iters) override;
    void print_centroids() const override;

private:
    std::vector<Point> points;
    std::vector<Point> centroids;
    std::vector<int> labels;
    int k;
};
