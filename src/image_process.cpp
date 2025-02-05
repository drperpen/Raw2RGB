#include "image_process.h"

cv::Mat loadImage(const std::string &imagePath) {
    cv::Mat mat = cv::imread(imagePath, cv::IMREAD_ANYDEPTH);
    return mat;
}

void printImageInfo(const cv::Mat& image, const cv::Point& pixel) {
    // Get type
    int type = image.type();
    int depth = type & CV_MAT_DEPTH_MASK;  // Extract the depth part
    int channels = 1 + (type >> CV_CN_SHIFT);  // Extract the channel count

    // std::cout << "Image Information:\n";
    // std::cout << " - Dimensions: " << image.cols << " x " << image.rows << "\n";
    // std::cout << " - Channels: " << channels << "\n";
    // std::cout << " - Data depth: ";
    switch (depth) {
        case CV_8U:  std::cout << "8-bit unsigned\n"; break;
        case CV_8S:  std::cout << "8-bit signed\n"; break;
        case CV_16U: std::cout << "16-bit unsigned\n"; break;
        case CV_16S: std::cout << "16-bit signed\n"; break;
        case CV_32S: std::cout << "32-bit signed\n"; break;
        case CV_32F: std::cout << "32-bit float\n"; break;
        case CV_64F: std::cout << "64-bit float\n"; break;
        default:     std::cout << "Unknown depth\n"; break;
    }

    // Check and print the pixel value
    if (pixel.x < image.cols && pixel.y < image.rows) {
        std::cout << "Pixel Value at (" << pixel.x << ", " << pixel.y << "):\n";
        if (depth == CV_8U)
            std::cout << " - Value: " << image.at<uchar>(pixel) << "\n";
        else if (depth == CV_16U)
            std::cout << " - Value: " << image.at<ushort>(pixel) << "\n";
        else if (depth == CV_32F)
            std::cout << " - Value: " << image.at<float>(pixel) << "\n";
        else if (depth == CV_64F)
            std::cout << " - Value: " << image.at<double>(pixel) << "\n";
        else
            std::cout << " - Unsupported data type for pixel value\n";
    } else {
        std::cout << "Pixel (" << pixel.x << ", " << pixel.y << ") is out of bounds!\n";
    }
}

void correctDeadPixels(cv::Mat& rawImage, cv::Mat &mask, float brightFactor, float darkFactor) {
    CV_Assert(rawImage.rows % 2 == 0 && rawImage.cols % 2 == 0);
    CV_Assert(rawImage.type() == CV_16U);
    CV_Assert(mask.type() == CV_8U);
    CV_Assert(rawImage.size() == mask.size());


    // Determine SIMD width based on available instruction sets
    #ifdef __AVX512F__
        const int chunkSize = 32;  // AVX-512: 512 bits = 32 x 16-bit integers
    #elif defined(__AVX__)
        const int chunkSize = 16;  // AVX: 256 bits = 16 x 16-bit integers
    #else
        const int chunkSize = 8;   // SSE: 128 bits = 8 x 16-bit integers
    #endif

    // Pre-allocate all matrices at once
    const cv::Size halfSize(rawImage.cols / 2, rawImage.rows / 2);
    std::array<cv::Mat, 4> subImages, subMasks;
    std::array<cv::Mat, 4> medianSubsampled;

    // Create lookup tables for factors - using SIMD
    alignas(64) uint16_t brightLUT[65536];  // Aligned for AVX-512
    alignas(64) uint16_t darkLUT[65536];

    #pragma omp simd aligned(brightLUT, darkLUT: 64)
    for (int i = 0; i < 65536; ++i) {
        brightLUT[i] = static_cast<uint16_t>(std::min(65535.0f, i * brightFactor));
        darkLUT[i] = static_cast<uint16_t>(i * darkFactor);
    }


    for (int i = 0; i < 4; ++i) {
        subImages[i] = cv::Mat(halfSize, CV_16U);
        subMasks[i] = cv::Mat(halfSize, CV_8U);
        medianSubsampled[i] = cv::Mat(halfSize, CV_16U);
    }

    // Process each Bayer channel
    for (int bayerIdx = 0; bayerIdx < 4; ++bayerIdx) {
        const int by = bayerIdx / 2;
        const int bx = bayerIdx % 2;

        auto* srcPtr = rawImage.ptr<uint16_t>();
        auto* maskPtr = mask.ptr<uint8_t>();
        auto* dstPtr = subImages[bayerIdx].ptr<uint16_t>();
        auto* dstMaskPtr = subMasks[bayerIdx].ptr<uint8_t>();
        const int stride = rawImage.cols;

        // SIMD-optimized subsampling with mask
        for (int y = 0; y < halfSize.height; ++y) {
            const uint16_t* srcRow = srcPtr + (y * 2 + by) * stride + bx;
            const uint8_t* maskRow = maskPtr + (y * 2 + by) * stride + bx;
            uint16_t* dstRow = dstPtr + y * halfSize.width;
            uint8_t* dstMaskRow = dstMaskPtr + y * halfSize.width;

            #pragma omp simd aligned(srcRow, maskRow, dstRow, dstMaskRow: 64)
            for (int x = 0; x < halfSize.width; ++x) {
                dstRow[x] = srcRow[x * 2];
                dstMaskRow[x] = maskRow[x * 2];
            }
        }

        // Apply median blur
        cv::medianBlur(subImages[bayerIdx], medianSubsampled[bayerIdx], 3);
    }

    // Process pixels in SIMD chunks
    const int totalPixels = rawImage.total();
    auto* imgPtr = rawImage.ptr<uint16_t>();
    auto* maskPtr = mask.ptr<uint8_t>();
    const int alignedSize = totalPixels & ~(chunkSize - 1);

    for (int i = 0; i < alignedSize; i += chunkSize) {
        alignas(64) uint16_t pixelVals[chunkSize];
        alignas(64) uint16_t medianVals[chunkSize];
        alignas(64) uint8_t maskVals[chunkSize];
        alignas(64) int bayerIndices[chunkSize];
        alignas(64) int yCoords[chunkSize];
        alignas(64) int xCoords[chunkSize];

        // Load values and prepare coordinates
        #pragma omp simd aligned(pixelVals, maskVals, yCoords, xCoords, bayerIndices: 64)
        for (int j = 0; j < chunkSize; ++j) {
            const int idx = i + j;
            yCoords[j] = idx / rawImage.cols;
            xCoords[j] = idx % rawImage.cols;
            bayerIndices[j] = (yCoords[j] % 2) * 2 + (xCoords[j] % 2);
            pixelVals[j] = imgPtr[idx];
            maskVals[j] = maskPtr[idx];
        }

        // Get median values
        #pragma omp simd aligned(medianVals, yCoords, xCoords, bayerIndices: 64)
        for (int j = 0; j < chunkSize; ++j) {
            medianVals[j] = medianSubsampled[bayerIndices[j]].at<uint16_t>(yCoords[j]/2, xCoords[j]/2);
        }

        // Compare and correct only masked pixels
        #pragma omp simd aligned(pixelVals, medianVals, maskVals: 64)
        for (int j = 0; j < chunkSize; ++j) {
            const int idx = i + j;
            if (maskVals[j] && (pixelVals[j] > brightLUT[medianVals[j]] ||
                               pixelVals[j] < darkLUT[medianVals[j]])) {
                imgPtr[idx] = medianVals[j];
            }
        }
    }

    // Handle remaining pixels with mask check
    for (int i = alignedSize; i < totalPixels; ++i) {
        if (!maskPtr[i]) continue;

        const int y = i / rawImage.cols;
        const int x = i % rawImage.cols;
        const int bayerIdx = (y % 2) * 2 + (x % 2);

        const uint16_t pixelVal = imgPtr[i];
        const uint16_t medianVal = medianSubsampled[bayerIdx].at<uint16_t>(y/2, x/2);

        if (pixelVal > brightLUT[medianVal] || pixelVal < darkLUT[medianVal]) {
            imgPtr[i] = medianVal;
        }
    }
}



void saveImage(const std::string &imagePath, const cv::Mat &image) {
    cv::imwrite(imagePath, image);
}

std::vector<Image> getRawFiles(const std::string& folderPath, const std::string& outputFolder) {
    std::vector<Image> images;
    int frameCounter = 0;

    // Iterate over the contents of the folder
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        // Check if the entry is a file
        if (entry.is_regular_file() &&
            (entry.path().extension() == ".tiff" || entry.path().extension() == ".TIFF")) {

            Image imgStruct;

            // Set the readPath as the current file path
            imgStruct.readPath = entry.path().string();

            // Create the writePath by replacing the extension with ".png"
            std::string outputFileName = entry.path().stem().string() + ".png";
            imgStruct.writePath = (std::filesystem::path(outputFolder) / outputFileName).string();

            // Assign the frame number and other image fields
            imgStruct.frame = frameCounter++;
            // imgStruct.img = {}; // Placeholder for cv::Mat. Actual image loading can be deferred.

            // Add the Image struct to the vector
            images.push_back(imgStruct);
            }
    }
    return images;
}
//         cv::demosaicing(image, processed_image, demosaicMode); //https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

void processImages(const std::vector<Image> &images, const int demosaicMode, const double gamma, const std::string vignettePath) {

    const int ACTUAL_MAX_VALUE = 4095;
    const int ACTUAL_MAX_BLACKLEVEL = 64;

    std::filesystem::path filePath(images[0].writePath);
    std::filesystem::path folderPath = filePath.parent_path();

    // Check if the folder exists
    if (!std::filesystem::exists(folderPath)) {
        // If it doesn't exist, create the directory
        try {
            std::filesystem::create_directories(folderPath);
            std::cout << "Created directory: " << folderPath << std::endl;
            std::cout << "Saving images to : " << folderPath.string()<< std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            // return; // Exit if directory creation fails
        }
    } else {
        std::cout << "Saving images to : " << folderPath.string() << std::endl;
    }

    // Load vignette map
    // Load
    cv::Mat image_ref = loadImage(images[0].readPath);
    cv::Mat loadedMap(image_ref.rows, image_ref.cols, CV_64F);
    FILE* fp = fopen(vignettePath.c_str(), "rb");
    fread(loadedMap.data, sizeof(double), image_ref.rows*image_ref.cols, fp);
    fclose(fp);
    VignetteModel model;
    model.correctionMap = loadedMap;

    cv::Mat blackLevelMat = cv::Mat::ones(2, 2, CV_32F) * ACTUAL_MAX_BLACKLEVEL;



#pragma omp parallel for
    for (int idx = 0; idx < images.size(); idx++) {

        cv::Mat image_ = loadImage(images[idx].readPath);

        // Create mask for non-zero pixels (valid image data)
        cv::Mat mask = (image_ != 0);

        correctDeadPixels(image_, mask, 1.5f, 0.75f);
        cv::Mat image = applyVignetteCorrection(image_, model);

        // Convert to float for processing
        image.convertTo(image, CV_32F);
        // Black level correction
        cv::subtract(image, ACTUAL_MAX_BLACKLEVEL, image, mask);
        cv::max(image, 0, image);
        // Normalize to full 16-bit range
        image.convertTo(image, CV_32F, 1.0 / (ACTUAL_MAX_VALUE - ACTUAL_MAX_BLACKLEVEL));

        // Convert to 16-bit
        image.convertTo(image, CV_16U, 65535.0);

        cv::Mat processed_image;
        cv::demosaicing(image, processed_image, demosaicMode); //https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

        cv::Mat rgbMask;
        cv::cvtColor(mask, rgbMask, cv::COLOR_GRAY2BGR);

        //
        // // After demosaicing (processed_image is still 16-bit)
        // double max_value;
        // cv::minMaxLoc(processed_image, nullptr, &max_value);

        // Apply gamma correction only to masked pixels
        if (gamma != 1.0) {
            cv::Mat gamma_input;
            processed_image.convertTo(gamma_input, CV_32F, 1.0/65535.0);

            // Create output matrix
            cv::Mat gamma_corrected = gamma_input.clone();

            // Apply gamma correction only where mask is non-zero
            cv::pow(gamma_input, gamma, gamma_corrected);
            gamma_corrected.copyTo(processed_image, rgbMask);

            processed_image.convertTo(processed_image, CV_16U, 65535.0);
        }


        // Save image
        saveImage(images[idx].writePath, processed_image);
        }

    std::cout << "Finished Saving images to : " << folderPath.string() <<  std::endl;
}


float calculateEntropy(float a, float b, float c, const cv::Mat& img) {
    const int rows = img.rows;
    const int cols = img.cols;

    float centerX = cols/2.0f, centerY = rows/2.0f;
    const float maxRadius = std::sqrt(centerX*centerX + centerY*centerY);

    // Create floating point image for intermediate calculations
    cv::Mat floatImg(img.size(), CV_32F);

    // Apply vignette correction formula
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {
            float dx = x - centerX;
            float dy = y - centerY;
            float r = std::sqrt(dx*dx + dy*dy) / maxRadius;
            float r2 = r*r;
            float r4 = r2*r2;
            float r6 = r2*r4;
            float g = 1 + a*r2 + b*r4 + c*r6;
            floatImg.at<float>(y, x) = img.at<uint16_t>(y, x) * g;
        }
    }

    // Calculate log image
    cv::Mat logImg;
    floatImg.convertTo(logImg, CV_32F);
    cv::log(1.0f + floatImg, logImg);
    logImg = 255.0f * logImg / 8.0f;


    // Calculate histogram
    float histogram[256] = {0};
    for(int y = 0; y < rows; y++) {
        for(int x = 0; x < cols; x++) {
            float val = logImg.at<float>(y, x);
            int k_down = std::floor(val);
            int k_up = std::ceil(val);
            if(k_down >= 0 && k_down < 256)
                histogram[k_down] += (1 + k_down - val);
            if(k_up >= 0 && k_up < 256)
                histogram[k_up] += (val - k_up);
        }
    }

    // Smooth histogram
    float tempHist[264];
    for(int i = 0; i < 4; i++) tempHist[i] = histogram[4-i];
    std::memcpy(tempHist + 4, histogram, 256 * sizeof(float));
    for(int i = 0; i < 4; i++) tempHist[260+i] = histogram[251+i];

    for(int i = 0; i < 256; i++) {
        histogram[i] = (tempHist[i] + 2*tempHist[i+1] + 3*tempHist[i+2] +
                       4*tempHist[i+3] + 5*tempHist[i+4] + 4*tempHist[i+5] +
                       3*tempHist[i+6] + 2*tempHist[i+7] + tempHist[i+8]) / 25.0f;
    }

    // Calculate entropy
    float sum = 0;
    for(int i = 0; i < 256; i++) sum += histogram[i];

    float entropy = 0;
    for(int i = 0; i < 256; i++) {
        float p = histogram[i] / sum;
        if(p > 0) entropy -= p * std::log(p);
    }

    return entropy;
}

bool checkCoefficients(float a, float b, float c) {
    if ((a > 0) && (b == 0) && (c == 0)) return true;
    if (a >= 0 && b > 0 && c == 0) return true;
    if (c == 0 && b < 0 && -a <= 2*b) return true;
    if (c > 0 && b*b < 3*a*c) return true;
    if (c > 0 && b*b == 3*a*c && b >= 0) return true;
    if (c > 0 && b*b == 3*a*c && -b >= 3*c) return true;
    if (c > 0 && b*b > 3*a*c) {
        float q_p = (-2*b + std::sqrt(4*b*b - 12*a*c))/(6*c);
        float q_d = (-2*b - std::sqrt(4*b*b - 12*a*c))/(6*c);
        if (q_p <= 0) return true;
        if (q_d >= 1) return true;
    }
    if (c < 0 && b*b > 3*a*c) {
        float q_p = (-2*b + std::sqrt(4*b*b - 12*a*c))/(6*c);
        float q_d = (-2*b - std::sqrt(4*b*b - 12*a*c))/(6*c);
        if (q_p >= 1 && q_d <= 0) return true;
    }
    return false;
}

VignetteModel estimateVignetting(const cv::Mat& rawBayer, const std::string &outPath, const std::string& bayerPattern, int polyDegree) {
    cv::Mat grayImg;
    cv::cvtColor(rawBayer, grayImg, cv::COLOR_BayerRG2GRAY);

    float a = 0, b = 0, c = 0;
    float a_min = 0, b_min = 0, c_min = 0;
    float delta = 8;
    float minEntropy = calculateEntropy(a, b, c, grayImg);

    // Optimize coefficients
    while(delta > 1.0f/256) {
        for(float* param : {&a, &b, &c}) {
            float orig = *param;

            // Try increasing
            *param = orig + delta;
            if(checkCoefficients(a, b, c)) {
                float entropy = calculateEntropy(a, b, c, grayImg);
                if(entropy < minEntropy) {
                    a_min = a; b_min = b; c_min = c;
                    minEntropy = entropy;
                }
            }

            // Try decreasing
            *param = orig - delta;
            if(checkCoefficients(a, b, c)) {
                float entropy = calculateEntropy(a, b, c, grayImg);
                if(entropy < minEntropy) {
                    a_min = a; b_min = b; c_min = c;
                    minEntropy = entropy;
                }
            }

            *param = orig; // Reset for next iteration
        }
        delta /= 2.0f;
    }

    // Create correction map
    cv::Mat correctionMap(rawBayer.size(), CV_64F);
    const float centerX = rawBayer.cols/2.0f;
    const float centerY = rawBayer.rows/2.0f;
    const float maxRadius = std::sqrt(centerX*centerX + centerY*centerY);

    for(int y = 0; y < rawBayer.rows; y++) {
        for(int x = 0; x < rawBayer.cols; x++) {
            float dx = x - centerX;
            float dy = y - centerY;
            float r = std::sqrt(dx*dx + dy*dy) / maxRadius;
            float r2 = r*r;
            float r4 = r2*r2;
            float r6 = r2*r4;
            correctionMap.at<double>(y, x) = 1 + a_min*r2 + b_min*r4 + c_min*r6;
        }
    }

    VignetteModel model;
    model.correctionMap = correctionMap;
    model.a = a_min;
    model.b = b_min;
    model.c = c_min;

    // Save correction map
    if(!outPath.empty()) {
        FILE* fp = fopen(outPath.c_str(), "wb");
        if(fp) {
            fwrite(correctionMap.data, sizeof(double), rawBayer.rows*rawBayer.cols, fp);
            fclose(fp);
        }
    }

    return model;
}

cv::Mat applyVignetteCorrection(const cv::Mat& rawBayer, const VignetteModel& model) {
    cv::Mat result;
    rawBayer.convertTo(result, CV_64F);
    result = result.mul(model.correctionMap);

    // Clip values to valid range
    cv::min(result, 65535.0, result);
    cv::max(result, 0.0, result);

    cv::Mat output;
    result.convertTo(output, CV_16U);
    return output;
}


std::unordered_map<int, Image> loadAndProcessImages(const std::string &path, int frame, int startCam, int endCam, std::map<int, cameraStrct> camData)
    {

        auto start = std::chrono::system_clock::now();

        std::unordered_map<int, Image> images;
        std::mutex mtx;

#pragma omp parallel for
    for (int idx = startCam; idx <= endCam; idx++)
    {
        std::string pathIter = path;

        // // kochika
        // size_t pad2 = 2;
        // auto frameStr = std::string(size_t(2) - std::min(size_t(2), std::to_string(idx).length()), '0') + std::to_string(idx);
        // pathIter += "capture_focus_" + frameStr + ".jpg";

        // dome
        size_t pad2 = 3;
        constexpr size_t paddingWidth = 3;
        auto frameStr = std::string(paddingWidth - std::min(paddingWidth, std::to_string(idx).length()), '0') + std::to_string(idx);
        pathIter += "C" + frameStr + "F0002000.png";

        // std::cout << "frameStr: " << frameStr << std::endl;

        if (std::filesystem::exists(pathIter))
        {
            cv::Mat img = cv::imread(pathIter);

            //             #### Demosaicing

            // img_d = cv.demosaicing(img, cv.COLOR_BayerRG2BGR)
            // ratio = np.amax(img_d) / 255
            // img_d = (img_d / ratio).astype('uint8')
            // gamma = 1.75
            // invGamma = 1.0 / gamma
            // table = np.array([((i / 255.0) ** invGamma) * 255
            //     for i in np.arange(0, 256)]).astype("uint8")
            // img_d = cv.LUT(img_d, table)

            // cv::Mat distortions = (cv::Mat_<double>(5, 1) << camData[idx].k1,
            //                        camData[idx].k2,
            //                        camData[idx].p2,
            //                        camData[idx].p1,
            //                        camData[idx].k3);

            cv::Mat distortions = (cv::Mat_<double>(5, 1) << camData[idx].k1,
                                   camData[idx].k2,
                                   camData[idx].p2,
                                   camData[idx].p1,
                                   camData[idx].k3);

            cv::Mat camMatrix = (cv::Mat_<double>(3, 3) << camData[idx].f, 0.0, (camData[idx].resolution.height / 2.0) + camData[idx].cx,
                                 0.0, camData[idx].f, (camData[idx].resolution.width / 2.0) + camData[idx].cy,
                                 0.0, 0.0, 1.0);

            cv::Mat offsetCamMatrix;
            camMatrix.copyTo(offsetCamMatrix);
            offsetCamMatrix.at<double>(0, 2) -= camData[idx].cx;
            offsetCamMatrix.at<double>(1, 2) -= camData[idx].cy;

            cv::Mat map1, map2;
            cv::initUndistortRectifyMap(camMatrix,
                                        distortions,
                                        cv::Mat::eye(3, 3, CV_64F),
                                        offsetCamMatrix,
                                        cv::Size(camData[idx].resolution.width, camData[idx].resolution.height),
                                        CV_32FC1,
                                        map1, map2);

            cv::Mat undistortedImg;
            cv::remap(img, undistortedImg, map1, map2, cv::INTER_LINEAR);

            // imagesStrct imgStrct = imagesStrct(idx, undistortedImg, pathIter);

            // // Undistort

            // mtx.lock();
            // imgs.insert(std::make_pair(idx, imgStrct));
            // mtx.unlock();
        }
        else
        {
            std::cout << "Could not find image: " << pathIter << std::endl;
        }
    }
    // Images::images = imgs;
    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elap = end - start;
    // std::cout << imgs.size() << " images loaded in " << elap.count() << " seconds" << std::endl;
    return images;
}