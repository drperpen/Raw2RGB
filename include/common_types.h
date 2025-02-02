#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <filesystem>

struct Image
{
    int frame;
    cv::Mat img;
    std::string readPath;
    std::string writePath;
};

struct cameraStrct
{
    // Camera pose
    int cam;
    int camId;
    int camSensorId;
    std::string label;
    std::vector<double> transforms;

    // Camera intrinsics
    cv::Size resolution;
    double f, cx, cy, k1, k2, k3, p1, p2;
    double pw;

    // Camera transforms
    Eigen::Matrix4d mToNDC;
    Eigen::Matrix4d mFromNDC;
    Eigen::Matrix4d W;
    Eigen::Vector3d pos;
    double farthestCam;
    Eigen::Matrix3d rotationMatrix;
    Eigen::Matrix<double, 3, 4> projectionMatrix;

    // Camera extras
    std::string imagePath;

    cameraStrct()
    {
    }

    cameraStrct(int _cam, int _camID, int _camSensorId, std::string _label,
                cv::Size _resolution, double _f, double _cx, double _cy, double _k1, double _k2, double _k3, double _p1, double _p2,
                double _pw,
                Eigen::Matrix4d _mToNDC,
                Eigen::Matrix4d _mFromNDC,
                Eigen::Matrix4d _W,
                Eigen::Vector3d _pos,
                Eigen::Matrix3d _rotationMatrix,
                Eigen::Matrix<double, 3, 4> _projectionMatrix,
                double _farthestCam,
                std::string _imagePath)
    {
        cam = _cam;
        camId = _camID;
        camSensorId = _camSensorId;
        label = _label;
        f = _f;
        cx = _cx;
        cy = _cy;
        k1 = _k1;
        k2 = _k2;
        k3 = _k3;
        p1 = _p1;
        p2 = _p2;

        pw = _pw;

        mToNDC = _mToNDC;
        mFromNDC = _mFromNDC;
        W = _W;
        pos = _pos;
        farthestCam = _farthestCam;
        rotationMatrix = _rotationMatrix;
        projectionMatrix = _projectionMatrix;
        imagePath = _imagePath;
    }
};