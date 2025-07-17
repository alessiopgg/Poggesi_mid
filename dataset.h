#pragma once
#include <vector>
#include <string>
#include "base_kmeans.h"

class Dataset {
public:
    bool load_from_csv(const std::string& filename);
    void init_centroids(int k, int seed = 24);
    void print_centroids() const;

    const std::vector<Point>& get_points() const { return points; }
    const std::vector<Point>& get_centroids() const { return centroids; }

private:
    std::vector<Point> points;
    std::vector<Point> centroids;
};
