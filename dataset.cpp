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

    points.clear();
    points_soa.clear();

    std::string line;

    auto try_parse_xy = [&](const std::string& s, double& x, double& y) -> bool {
        std::stringstream ss(s);
        std::string x_str, y_str;

        if (!std::getline(ss, x_str, ',')) return false;
        if (!std::getline(ss, y_str, ',')) return false;

        try {
            x = std::stod(x_str);
            y = std::stod(y_str);
            return true;
        } catch (...) {
            return false;
        }
    };

    // Leggi la prima riga: può essere header oppure dati
    if (std::getline(file, line)) {
        double x, y;
        if (try_parse_xy(line, x, y)) {
            // Prima riga era già un punto (niente header)
            points.push_back(Point{x, y});
            points_soa.x.push_back(x);
            points_soa.y.push_back(y);
        }
        // altrimenti era header: ignora e continua
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        double x, y;
        if (!try_parse_xy(line, x, y)) continue; // salta righe malformate

        points.push_back(Point{x, y});
        points_soa.x.push_back(x);
        points_soa.y.push_back(y);
    }

    return !points.empty();
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
