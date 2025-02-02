#pragma once

#include <iostream>
#include <unordered_map>
#include "common_types.h"

cv::Mat loadImage(const std::string &imagePath);
void printImageInfo(const cv::Mat& image, const cv::Point& pixel);
void correctDeadPixels(cv::Mat& rawImage, float brightFactor = 2.5, float darkFactor = 0.2);
std::vector<Image> getRawFiles(const std::string& folderPath, const std::string& outputFolder);
void saveImage(const std::string &imagePath, const cv::Mat &image);
void processImages(const std::vector<Image> &images, int demosaicMode, double gamma);
std::unordered_map<int, Image> loadAndProcessImages(std::string path, int frame, int startCam, int endCam, std::map<int, cameraStrct> camData);