
#include <iostream>
#include "common_types.h"
#include "image_process.h"

#include <thread> // for std::this_thread::sleep_for
#include <chrono> // for std::chrono::seconds

int main(int argc, char** argv){

    if (argc < 3) {
        std::cerr << "Error with inputs!" << std::endl;
        std::cerr << "Usage: Raw2RGB.exe input_path output_path" << std::endl;
        std::exit(1);
    }
    std::string img_load_path = argv[1];
    std::string img_save_path = argv[2];
    int demosaic_mode = 48;
    if (argv[3]) {
        demosaic_mode = std::stoi(argv[3]);
    }

    cv::Mat img = loadImage(img_load_path);
    cv::Mat img_processed = processImage(img, demosaic_mode);

    std::this_thread::sleep_for(std::chrono::seconds(5));

    saveImage(img_save_path, img_processed);
    std::cout << "Saved image: " << img_save_path << std::endl;
}
