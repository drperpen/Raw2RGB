//
// Created by perpen on 3/1/2025.
//

#pragma once

#include "common_types.h"
#include <fstream>

class ColorLUT {
public:
    static constexpr int SIZE = 128;
    static constexpr int SIZE2 = SIZE * SIZE;
    static constexpr int SIZE3 = SIZE * SIZE * SIZE;

    bool load(const std::string& path);
    cv::Mat apply(const cv::Mat &input) const;
    std::vector<cv::Vec3f> lut;

// private:

};