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

void correctDeadPixels(cv::Mat& rawImage, float brightFactor, float darkFactor) {
    CV_Assert(rawImage.rows % 2 == 0 && rawImage.cols % 2 == 0);
    CV_Assert(rawImage.type() == CV_16U);

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
    std::array<cv::Mat, 4> subImages;
    std::array<cv::Mat, 4> medianSubsampled;

    // Create lookup tables for factors - using SIMD
    alignas(64) uint16_t brightLUT[65536];  // Aligned for AVX-512
    alignas(64) uint16_t darkLUT[65536];

    #pragma omp simd aligned(brightLUT, darkLUT: 64)
    for (int i = 0; i < 65536; ++i) {
        brightLUT[i] = static_cast<uint16_t>(std::min(65535.0f, i * brightFactor));
        darkLUT[i] = static_cast<uint16_t>(i * darkFactor);
    }

    // Pre-allocate subimages
    for (int i = 0; i < 4; ++i) {
        subImages[i] = cv::Mat(halfSize, CV_16U);
        medianSubsampled[i] = cv::Mat(halfSize, CV_16U);
    }

    // Process each Bayer channel
    for (int bayerIdx = 0; bayerIdx < 4; ++bayerIdx) {
        const int by = bayerIdx / 2;
        const int bx = bayerIdx % 2;

        // Direct pointer access for faster subsampling
        auto* srcPtr = rawImage.ptr<uint16_t>();
        auto* dstPtr = subImages[bayerIdx].ptr<uint16_t>();
        const int stride = rawImage.cols;

        // SIMD-optimized subsampling
        for (int y = 0; y < halfSize.height; ++y) {
            const uint16_t* srcRow = srcPtr + (y * 2 + by) * stride + bx;
            uint16_t* dstRow = dstPtr + y * halfSize.width;

            #pragma omp simd aligned(srcRow, dstRow: 64)
            for (int x = 0; x < halfSize.width; ++x) {
                dstRow[x] = srcRow[x * 2];
            }
        }

        cv::medianBlur(subImages[bayerIdx], medianSubsampled[bayerIdx], 3);
    }

    // Process pixels and apply correction using SIMD
    const int totalPixels = rawImage.total();
    auto* imgPtr = rawImage.ptr<uint16_t>();

    // Process pixels in SIMD chunks
    const int alignedSize = totalPixels & ~(chunkSize - 1);

    for (int i = 0; i < alignedSize; i += chunkSize) {
        alignas(64) uint16_t pixelVals[chunkSize];
        alignas(64) uint16_t medianVals[chunkSize];
        alignas(64) int bayerIndices[chunkSize];
        alignas(64) int yCoords[chunkSize];
        alignas(64) int xCoords[chunkSize];

        // Prepare coordinates and bayer indices
        #pragma omp simd aligned(yCoords, xCoords, bayerIndices: 64)
        for (int j = 0; j < chunkSize; ++j) {
            const int idx = i + j;
            yCoords[j] = idx / rawImage.cols;
            xCoords[j] = idx % rawImage.cols;
            bayerIndices[j] = (yCoords[j] % 2) * 2 + (xCoords[j] % 2);
            pixelVals[j] = imgPtr[idx];
        }

        // Get median values
        #pragma omp simd aligned(medianVals, yCoords, xCoords, bayerIndices: 64)
        for (int j = 0; j < chunkSize; ++j) {
            medianVals[j] = medianSubsampled[bayerIndices[j]].at<uint16_t>(yCoords[j]/2, xCoords[j]/2);
        }

        // Compare and correct
        #pragma omp simd aligned(pixelVals, medianVals: 64)
        for (int j = 0; j < chunkSize; ++j) {
            const int idx = i + j;
            if (pixelVals[j] > brightLUT[medianVals[j]] || pixelVals[j] < darkLUT[medianVals[j]]) {
                imgPtr[idx] = medianVals[j];
            }
        }
    }

    // Handle remaining pixels
    for (int i = alignedSize; i < totalPixels; ++i) {
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

    const int ACTUAL_MAX_VALUE = 1225;
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

        // auto start_time = std::chrono::high_resolution_clock::now();
        cv::Mat image_ = loadImage(images[idx].readPath);
        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << "Total processing time loadImage: " << duration << " s" << std::endl;


        // Create mask for non-zero pixels (valid image data)
        cv::Mat mask = (image_ != 0); // This creates a CV_8U mask (values 0 or 255)
        // mask.convertTo(mask, CV_8U); // Ensure it's explicitly unsigned char
        // mask = mask / 255;          // Scale to binary (0 or 1)

        // start_time = std::chrono::high_resolution_clock::now();
        correctDeadPixels(image_, 1.5f, 0.75f);
        // end_time = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << "Total processing time correctDeadPixels: " << duration << " s" << std::endl;

        // start_time = std::chrono::high_resolution_clock::now();
        cv::Mat image = applyVignetteCorrection(image_, model);
        // end_time = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << "Total processing time applyVignetteCorrection: " << duration << " s" << std::endl;

        // start_time = std::chrono::high_resolution_clock::now();
        // Convert to float for processing
        image.convertTo(image, CV_32F);
        double minVal, maxVal;
        cv::minMaxLoc(image, &minVal, &maxVal, nullptr, nullptr, mask);
        cv::subtract(image, ACTUAL_MAX_BLACKLEVEL, image, mask);
        cv::max(image, 0, image);

        // Normalize to full 16-bit range
        image.convertTo(image, CV_32F, 1.0 / (maxVal - ACTUAL_MAX_BLACKLEVEL));
        // end_time = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << "Total processing time BLACKLEVEL: " << duration << " s" << std::endl;

        // Apply gamma correction
        // start_time = std::chrono::high_resolution_clock::now();
        if (gamma != 1.0) {
            cv::pow(image, gamma, image);
        }
        // end_time = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << "Total processing time gamma: " << duration << " s" << std::endl;

        // Convert to 16-bit
        image.convertTo(image, CV_16U, 65535.0);

        cv::Mat processed_image;
        // start_time = std::chrono::high_resolution_clock::now();
        cv::demosaicing(image, processed_image, demosaicMode); //https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
        // end_time = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << "Total processing time demosaicing: " << duration << " s" << std::endl;

        cv::Mat outputImage;
        processed_image.convertTo(outputImage, CV_8U, 1.0 / 256.0);

        // Save image
        // start_time = std::chrono::high_resolution_clock::now();
        saveImage(images[idx].writePath, outputImage);
        // end_time = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << "Total processing time saveImage: " << duration << " s" << std::endl;
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