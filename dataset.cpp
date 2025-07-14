#include "dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <ctime>

bool Dataset::load_from_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "❌ Errore apertura file: " << filename << "\n";
        return false;
    }

    std::string line;
    std::getline(file, line); // salta intestazione

    points.clear();
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string x_str, y_str;
        std::getline(ss, x_str, ',');
        std::getline(ss, y_str, ',');

        Point p{std::stod(x_str), std::stod(y_str)};
        points.push_back(p);
    }

    return true;
}

void Dataset::init_centroids(int k, int seed) {
    std::mt19937 rng(seed);
    //std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<size_t> dist(0, points.size() - 1);

    centroids.clear();
    for (int i = 0; i < k; ++i) {
        centroids.push_back(points[dist(rng)]);
    }
}

void Dataset::print_centroids() const {
    std::cout << "Centroidi iniziali:\n";
    for (size_t i = 0; i < centroids.size(); ++i) {
        std::cout << " C" << i << " = (" << centroids[i].x << ", " << centroids[i].y << ")\n";
    }
}
