#pragma once
#include <vector>
#include <string>
#include "point.h"

class Dataset {
private:
    std::vector<Point> points;
    PointsSoA points_soa;
    std::vector<Point> centroids;

public:
    bool load_from_csv(const std::string& filename);
    void init_centroids(int k, int seed);
    void print_centroids() const;

    const std::vector<Point>& get_points() const { return points; }
    const PointsSoA& get_points_soa() const { return points_soa; }
    const std::vector<Point>& get_centroids() const { return centroids; }
};
