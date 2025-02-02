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
    // Create separate masks for each Bayer color
    cv::Mat masks[4];    // For RGGB pattern
    cv::Mat medians[4];  // Median values for each color

    for(int by = 0; by < 2; by++) {
        for(int bx = 0; bx < 2; bx++) {
            // Create downsampled image for this Bayer position
            cv::Mat subImage;
            cv::Mat subsampledImg;

            // Extract pixels for this Bayer position
            subsampledImg.create(rawImage.rows/2, rawImage.cols/2, rawImage.type());
            for(int y = by; y < rawImage.rows-1; y += 2) {
                for(int x = bx; x < rawImage.cols-1; x += 2) {
                    subsampledImg.at<uint16_t>(y/2, x/2) = rawImage.at<uint16_t>(y, x);
                }
            }

            // Calculate median for this color channel
            cv::medianBlur(subsampledImg, subImage, 3);

            // Upsample back to full resolution
            cv::Mat upsampled;
            cv::resize(subImage, upsampled, rawImage.size(), 0, 0, cv::INTER_NEAREST);

            // Create mask for this Bayer position
            cv::Mat colorMask = cv::Mat::zeros(rawImage.size(), CV_8U);
            for(int y = by; y < rawImage.rows; y += 2) {
                for(int x = bx; x < rawImage.cols; x += 2) {
                    colorMask.at<uchar>(y, x) = 1;
                }
            }

            // Store median values and create mask for outliers
            int idx = by * 2 + bx;
            medians[idx] = upsampled.clone();
            masks[idx] = (rawImage > upsampled * brightFactor |
                         rawImage < upsampled * darkFactor) & colorMask;
        }
    }

    // Combine all masks
    cv::Mat finalMask = masks[0] | masks[1] | masks[2] | masks[3];

    // Find dead pixel locations
    std::vector<cv::Point> deadPixels;
    cv::findNonZero(finalMask, deadPixels);

    // Only process the dead pixels
    for(const auto& pt : deadPixels) {
        int bayerIdx = (pt.y % 2) * 2 + (pt.x % 2);
        rawImage.at<uint16_t>(pt.y, pt.x) = medians[bayerIdx].at<uint16_t>(pt.y, pt.x);
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


void processImages(const std::vector<Image> &images, const int demosaicMode, const double gamma) {

    // img_d = cv.demosaicing(image, cv.COLOR_BayerRG2BGR)
    // ratio = np.amax(img_d) / 255
    // img_d = (img_d / ratio).astype('uint8')
    // gamma = 1.75
    // invGamma = 1.0 / gamma
    // table = np.array([((i / 255.0) ** invGamma) * 255
    //     for i in np.arange(0, 256)]).astype("uint8")
    // img_d = cv.LUT(img_d, table)

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


#pragma omp parallel for
    for (int idx = 0; idx < images.size(); idx++) {

        cv::Mat image = loadImage(images[idx].readPath);

        // constexpr int offset = 2;
        // for (int i = -offset; i <= offset; i++) {
        //     printImageInfo(image, cv::Point(1087+i, 3077+i));
        // }

        correctDeadPixels(image, 1.5f, 0.75f);

        cv::Mat processed_image;
        cv::demosaicing(image, processed_image, demosaicMode); //https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html



        // After demosaicing (processed_image is still 16-bit)
        double max_value;
        cv::minMaxLoc(processed_image, nullptr, &max_value);
        const double ratio = max_value / 255.0;  // Scale to 8-bit range

        // Convert to 8-bit
        processed_image = processed_image / ratio;
        processed_image.convertTo(processed_image, CV_8U);  // This will scale to 0-255 range

        // Create gamma lookup table for 8-bit
        const double inv_gamma = 1.0 / gamma;
        cv::Mat lookupTable(1, 256, CV_8U);
        uchar* p = lookupTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, inv_gamma) * 255.0);
        }

        // Apply gamma correction using LUT
        cv::LUT(processed_image, lookupTable, processed_image);

        // Save image
        saveImage(images[idx].writePath, processed_image);
        }

    std::cout << "Finished Saving images to : " << folderPath.string() <<  std::endl;
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