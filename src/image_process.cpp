#include "image_process.h"

void shiftBlackPixels(cv::Mat& image, ushort neutralValue) {
    // Process all pixels in parallel using OpenCV's optimized operations
    cv::Mat zeroMask = (image == 0);
    image.setTo(neutralValue, zeroMask);
}


cv::Mat loadImage(const std::string &imagePath, const std::string &logFilePath) {
    // Attempt to load the image
    cv::Mat mat = cv::imread(imagePath, cv::IMREAD_UNCHANGED);

    // Check if the image was successfully loaded
    if (mat.empty()) {
        // Log the issue to a file
        std::ofstream logFile(logFilePath, std::ios_base::app); // Open in append mode
        if (logFile.is_open()) {
            logFile << "Error! Skipping image: " << imagePath << std::endl;
            logFile.close();
        } else {
            std::cerr << "Failed to open log file at " << logFilePath << std::endl;
        }

        // Return an empty matrix to indicate failure
        return cv::Mat();
    }

    if (mat.channels() == 3) {
        std::vector<cv::Mat> channels;
        cv::split(mat, channels);
        mat = channels[2];
    }

    return mat;
}


void printImageInfo(const cv::Mat& image, const cv::Point& pixel) {
    // Get type
    int type = image.type();
    int depth = type & CV_MAT_DEPTH_MASK;  // Extract the depth part
    int channels = 1 + (type >> CV_CN_SHIFT);  // Extract the channel count

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

Image getRawFile(const std::string& inputPath, const std::string& outputPath) {
    Image imgStruct;

    // Create filesystem paths
    std::filesystem::path inPath(inputPath);
    std::filesystem::path outPath(outputPath);

    // Verify input file exists and has correct extension
    if (!std::filesystem::exists(inPath) ||
        (inPath.extension() != ".tiff" && inPath.extension() != ".TIFF")) {
        throw std::runtime_error("Invalid input file or unsupported format");
        }

    // Set the readPath
    imgStruct.readPath = inputPath;

    // Create the writePath
    if (std::filesystem::is_directory(outPath)) {
        // If output is a directory, create filename inside it
        std::string outputFileName = inPath.stem().string() + ".png";
        imgStruct.writePath = (outPath / outputFileName).string();
    } else {
        // Use output path as is
        imgStruct.writePath = outputPath;
    }

    // Set frame to 0 since it's a single file
    imgStruct.frame = 0;

    return imgStruct;
}

void processImages(
    const std::vector<Image> &images,
    const int demosaicMode,
    const double gamma,
    const std::string &vignettePath,
    const std::string& lutPath) {

    // const int ACTUAL_MAX_VALUE = 1225;
    const int ACTUAL_MAX_BLACKLEVEL = 64;
    const double ASSUMED_MAX_VALUE = 4095.0; // Assuming 10-bit raw data

    std::filesystem::path filePath(images[0].writePath);
    std::filesystem::path folderPath = filePath.parent_path();
    std::string logPath = folderPath.string() + "/log.txt";

    // Check if the folder exists
    if (!std::filesystem::exists(folderPath)) {
        // If it doesn't exist, create the directory
        try {
            std::filesystem::create_directories(folderPath);
            std::cout << "Created directory: " << folderPath << std::endl;
            std::cout << "Saving images to : " << folderPath.string()<< std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
        }
    } else {
        std::cout << "Saving images to : " << folderPath.string() << std::endl;
    }

    // Load vignette map
    cv::Mat image_ref = loadImage(images[0].readPath, logPath);
    cv::Mat loadedMap(image_ref.rows, image_ref.cols, CV_64F);
    VignetteModel model;
    bool useVignetteCorrection = false; // Flag to track if vignette correction should be applied

    if (!vignettePath.empty()) {
        FILE* fp = fopen(vignettePath.c_str(), "rb");
        if (fp) {
            // File exists, proceed to load
            fread(loadedMap.data, sizeof(double), image_ref.rows * image_ref.cols, fp);
            fclose(fp);
            model.correctionMap = loadedMap;
            useVignetteCorrection = true; // Enable vignette correction
        } else {
            // File does not exist, log a warning and proceed without vignette correction
            std::cerr << "Warning: Vignette map file '" << vignettePath << "' does not exist. Proceeding without vignette correction." << std::endl;
        }
    }

    // Load LUT once for all images or create black level
    ColorLUT colorLUT;
    bool useLUT = false;
    if (!lutPath.empty()) {
        useLUT = colorLUT.load(lutPath);
        useLUT = true;
        std::cout << "Loaded color LUT from: " << lutPath << std::endl;
    } else {
        cv::Mat blackLevelMat = cv::Mat::ones(2, 2, CV_32F) * ACTUAL_MAX_BLACKLEVEL;
    }


#pragma omp parallel for
    for (int idx = 0; idx < images.size(); idx++) {
        cv::Mat image_ = loadImage(images[idx].readPath, logPath);

        // Create mask for non-zero pixels (valid image data)
        cv::Mat mask = (image_ != 0); // This creates a CV_8U mask (values 0 or 255)
        correctDeadPixels(image_, 1.5f, 0.75f);
        cv::Mat image = image_;
        if (useVignetteCorrection) {
            image = applyVignetteCorrection(image_, model);
        }

        if (!useLUT) {
            // Convert to float for processing
            image.convertTo(image, CV_32F);

            // Apply black level subtraction to all valid pixels
            cv::subtract(image, ACTUAL_MAX_BLACKLEVEL, image, mask);
            cv::max(image, 0, image);

            // Use a fixed normalization factor instead of dynamic min/max
            // This ensures consistent gamma correction regardless of image content
            image.convertTo(image, CV_32F, 1.0 / (ASSUMED_MAX_VALUE - ACTUAL_MAX_BLACKLEVEL));

            // Apply gamma correction
            if (gamma != 1.0) {
                // Only apply gamma to non-zero pixels to maintain black levels
                cv::pow(image, gamma, image);
            }

            // Convert to 16-bit
            image.convertTo(image, CV_16U, 65535.0);
        }

        cv::Mat processed_image;
        cv::demosaicing(image, processed_image, demosaicMode); //https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

        if (useLUT) {
            processed_image = colorLUT.apply(processed_image);
        }

        cv::Mat outputImage;
        processed_image.convertTo(outputImage, CV_8U, 1.0 / 256.0);

        // Save image
        saveImage(images[idx].writePath, outputImage);

        }

    std::cout << "Finished Saving images to : " << folderPath.string() <<  std::endl;
}


void generate16bit(
    const Image &image,
    const int demosaicMode) {

    // const int ACTUAL_MAX_VALUE = 1225;
    const int ACTUAL_MAX_BLACKLEVEL = 64;

    std::filesystem::path filePath(image.writePath);
    std::filesystem::path folderPath = filePath.parent_path();
    std::string logPath = folderPath.string() + "/log.txt";

    // Check if the folder exists
    if (!std::filesystem::exists(folderPath)) {
        // If it doesn't exist, create the directory
        try {
            std::filesystem::create_directories(folderPath);
            std::cout << "Created directory: " << folderPath << std::endl;
            std::cout << "Saving 16 bit image to : " << folderPath.string()<< std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            // return; // Exit if directory creation fails
        }
    } else {
        std::cout << "Saving 16 bit image to : " << folderPath.string() << std::endl;
    }

    // Load vignette map
    // Load
    cv::Mat image_ref = loadImage(image.readPath, logPath);
    cv::Mat loadedMap(image_ref.rows, image_ref.cols, CV_64F);

    cv::Mat image_ = loadImage(image.readPath, logPath);

    // Create mask for non-zero pixels (valid image data)
    cv::Mat mask = (image_ != 0); // This creates a CV_8U mask (values 0 or 255)
    correctDeadPixels(image_, 1.5f, 0.75f);
    cv::Mat image_temp = image_;

    cv::Mat processed_image;
    cv::demosaicing(image_temp, processed_image, demosaicMode); //https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html

    cv::Mat outputImage;
    processed_image.convertTo(outputImage, CV_16U);

    // Save image
    saveImage(image.writePath, outputImage);

    std::cout << "Finished Saving 16 bit image to : " << folderPath.string() <<  std::endl;
}



float calculateEntropy(const float a, const float b, const float c, const cv::Mat& img) {
    const int rows = img.rows;
    const int cols = img.cols;

    const float centerX = static_cast<float>(cols)/2.0f, centerY = static_cast<float>(rows)/2.0f;
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
    const float centerX = static_cast<float>(rawBayer.cols)/2.0f;
    const float centerY = static_cast<float>(rawBayer.rows)/2.0f;
    const float maxRadius = std::sqrt(centerX*centerX + centerY*centerY);

    for(int y = 0; y < rawBayer.rows; y++) {
        for(int x = 0; x < rawBayer.cols; x++) {
            float dx = static_cast<float>(x) - centerX;
            float dy = static_cast<float>(y) - centerY;
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
