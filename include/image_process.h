#pragma once

#include <iostream>
#include <unordered_map>

#include "common_types.h"

std::unordered_map<int, Image> loadImages(std::string path, int frame, int startCam, int endCam, std::map<int, cameraStrct> camData);