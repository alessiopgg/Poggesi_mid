#pragma once
#include <vector>
#include <string>

struct Point {
    double x, y;
};

class BaseKMeans {
public:
    virtual void assign_clusters() = 0;
    virtual void update_centroids() = 0;
    virtual void fit(int k, int max_iters) = 0;
    virtual void print_centroids() const = 0;

    virtual ~BaseKMeans() = default;
};
