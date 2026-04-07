#pragma once
#include <vector>
#include <cstddef>

struct Point {
    double x;
    double y;
};

struct PointsSoA {
    std::vector<double> x;
    std::vector<double> y;

    void clear() { x.clear(); y.clear(); }
    size_t size() const { return x.size(); }
};
