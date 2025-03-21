#pragma once

#include <iostream>
#include <unordered_map>
#include "common_types.h"
#include <immintrin.h>
#include "lut_utils.h"

cv::Mat loadImage(const std::string &imagePath, const std::string &logFilePath);
void printImageInfo(const cv::Mat& image, const cv::Point& pixel);
void shiftBlackPixels(cv::Mat& image, ushort neutralValue = 2048);
void correctDeadPixels(cv::Mat& rawImage, float brightFactor = 2.5, float darkFactor = 0.2);
cv::Mat createBayerMask(const cv::Size& size, int by, int bx);
std::vector<Image> getRawFiles(const std::string& folderPath, const std::string& outputFolder);
Image getRawFile(const std::string& inputPath, const std::string& outputPath);
void saveImage(const std::string &imagePath, const cv::Mat &image);
void processImages(const std::vector<Image> &images, int demosaicMode, double gamma, const std::string &vignettePath, const std::string& lutPath);
void generate16bit(const Image &image, int demosaicMode);

VignetteModel estimateVignetting(const cv::Mat& rawBayer, const std::string &outPath, const std::string& bayerPattern = "RGGB", int polyDegree = 6);
cv::Mat applyVignetteCorrection(const cv::Mat& rawBayer, const VignetteModel& model);
