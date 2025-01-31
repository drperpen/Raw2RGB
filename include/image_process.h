#pragma once

#include <iostream>
#include <unordered_map>

#include "common_types.h"

cv::Mat loadImage(const std::string &imagePath);
void saveImage(std::string imagePath, const cv::Mat &image);
cv::Mat processImage(const cv::Mat &image, const int demosaicMode);
std::unordered_map<int, Image> loadAndProcessImages(std::string path, int frame, int startCam, int endCam, std::map<int, cameraStrct> camData);