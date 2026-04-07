#pragma once
#include "base_kmeans.h"
#include "point.h"

class KMeansSequential : public BaseKMeans {
public:
    KMeansSequential(const std::vector<Point>& input_points, const std::vector<Point>& initial_centroids);
    void assign_clusters() override;
    void update_centroids() override;
    void fit(int k);
    void print_centroids() const override;
    int get_last_iterations() const { return last_iters_; }
    const std::vector<Point>& get_centroids() const { return centroids; }
    const std::vector<int>&   get_labels()   const { return labels; }


private:
    std::vector<Point> points;
    std::vector<Point> centroids;
    std::vector<int> labels;
    int k;
    int last_iters_ = 0;

};
