#include <iostream>
#include <chrono>
#include "common_types.h"
#include "image_process.h"

int main(const int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: Raw2RGB.exe input_path_folder output_path_folder gamma_value" << std::endl;
        return 1;
    }

    const std::string img_load_path = argv[1];
    const std::string img_save_path = argv[2];
    const double gamma = std::stof(argv[3]);

    // Set default demosaic mode if not in arguments
    int demosaic_mode = 48;
    if (argv[4]) {
        demosaic_mode = std::stoi(argv[4]);
    }

    // Get list of raw files
    const std::vector<Image> raw_files = getRawFiles(img_load_path, img_save_path);
    if (raw_files.empty()) {
        std::cerr << "No raw files found in " << img_load_path << std::endl;
        return 1;
    }

    // Start timing processImages
    auto start_time = std::chrono::high_resolution_clock::now();

    // Process the images
    processImages(raw_files, demosaic_mode, gamma);

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Total processing time: " << duration << " s" << std::endl;

    return 0;
}