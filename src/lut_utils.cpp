//
// Created by perpen on 3/1/2025.
//

#include "lut_utils.h"

bool ColorLUT::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open LUT file: " << path << std::endl;
        return false;
    }

    lut.resize(SIZE3);
    std::string line;
    bool dataSection = false;
    int index = 0;
    while (std::getline(file, line)) {
        // Always check for "LUT data points" first:
        if (line.find("LUT data points") != std::string::npos) {
            dataSection = true;
            continue;
        }

        if (line.empty() || line[0] == '#') {
            continue;
        }

        if (dataSection && index < SIZE3) {
            float r, g, b;
            std::istringstream iss(line);
            if (iss >> r >> g >> b) {
                lut[index++] = cv::Vec3f(r, g, b);
            }
        }
    }

    return index == SIZE3;
}


cv::Mat ColorLUT::apply(const cv::Mat& src) const {
    cv::Mat dst;
    // const std::vector<cv::Vec3f>& lut,
    int cubeSize = 128;
    float scale = 65535.0f;

    // Verify input image type
    CV_Assert(src.type() == CV_16UC3);

    // Prepare output image
    dst.create(src.size(), src.type());

    // Calculate indices and weights for trilinear interpolation
    auto interpolateLUT = [&](const cv::Vec3f& color) -> cv::Vec3f {
        // Normalize color values to [0, 1] range
        float r = color[0] / scale;
        float g = color[1] / scale;
        float b = color[2] / scale;

        // Scale to LUT indices
        float r_scaled = r * (cubeSize - 1);
        float g_scaled = g * (cubeSize - 1);
        float b_scaled = b * (cubeSize - 1);

        // Get integer indices and fractional parts for interpolation
        int r_idx = std::min(static_cast<int>(r_scaled), cubeSize - 2);
        int g_idx = std::min(static_cast<int>(g_scaled), cubeSize - 2);
        int b_idx = std::min(static_cast<int>(b_scaled), cubeSize - 2);

        float r_frac = r_scaled - r_idx;
        float g_frac = g_scaled - g_idx;
        float b_frac = b_scaled - b_idx;

        // Calculate the 8 neighboring indices in the 3D LUT
        int idx000 = (r_idx * cubeSize * cubeSize) + (g_idx * cubeSize) + b_idx;
        int idx001 = idx000 + 1;
        int idx010 = idx000 + cubeSize;
        int idx011 = idx010 + 1;
        int idx100 = idx000 + cubeSize * cubeSize;
        int idx101 = idx100 + 1;
        int idx110 = idx100 + cubeSize;
        int idx111 = idx110 + 1;

        // Retrieve values from LUT
        cv::Vec3f c000 = lut[idx000];
        cv::Vec3f c001 = lut[idx001];
        cv::Vec3f c010 = lut[idx010];
        cv::Vec3f c011 = lut[idx011];
        cv::Vec3f c100 = lut[idx100];
        cv::Vec3f c101 = lut[idx101];
        cv::Vec3f c110 = lut[idx110];
        cv::Vec3f c111 = lut[idx111];

        // Trilinear interpolation
        cv::Vec3f c00 = c000 * (1.0f - r_frac) + c100 * r_frac;
        cv::Vec3f c01 = c001 * (1.0f - r_frac) + c101 * r_frac;
        cv::Vec3f c10 = c010 * (1.0f - r_frac) + c110 * r_frac;
        cv::Vec3f c11 = c011 * (1.0f - r_frac) + c111 * r_frac;

        cv::Vec3f c0 = c00 * (1.0f - g_frac) + c10 * g_frac;
        cv::Vec3f c1 = c01 * (1.0f - g_frac) + c11 * g_frac;

        cv::Vec3f result = c0 * (1.0f - b_frac) + c1 * b_frac;

        // Scale back to original range
        result[0] *= scale;
        result[1] *= scale;
        result[2] *= scale;

        return result;
    };

    // Process each pixel
#pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Vec3w srcPixel = src.at<cv::Vec3w>(y, x);
            cv::Vec3f floatPixel(srcPixel[0], srcPixel[1], srcPixel[2]);

            // Apply LUT with interpolation
            cv::Vec3f result = interpolateLUT(floatPixel);

            // Clamp and convert back to 16-bit
            cv::Vec3w dstPixel;
            dstPixel[2] = static_cast<ushort>(std::min(std::max(result[0], 0.0f), scale));
            dstPixel[1] = static_cast<ushort>(std::min(std::max(result[1], 0.0f), scale));
            dstPixel[0] = static_cast<ushort>(std::min(std::max(result[2], 0.0f), scale));

            dst.at<cv::Vec3w>(y, x) = dstPixel;
        }
    }

    return dst;
}

