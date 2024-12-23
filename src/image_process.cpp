#include "image_process.h"

std::unordered_map<int, Image> loadImages(std::string path, int frame, int startCam, int endCam, std::map<int, cameraStrct> camData)
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
        auto frameStr = std::string(size_t(3) - std::min(size_t(3), std::to_string(idx).length()), '0') + std::to_string(idx);
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

            imagesStrct imgStrct = imagesStrct(idx, undistortedImg, pathIter);

            // Undistort

            mtx.lock();
            imgs.insert(std::make_pair(idx, imgStrct));
            mtx.unlock();
        }
        else
        {
            std::cout << "Could not find image: " << pathIter << std::endl;
        }
    }
    Images::images = imgs;
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elap = end - start;
    std::cout << imgs.size() << " images loaded in " << elap.count() << " seconds" << std::endl;
    return images;
}