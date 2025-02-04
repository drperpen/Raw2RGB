#include <iostream>
#include <chrono>
#include "common_types.h"
#include "image_process.h"

int main(const int argc, char** argv) {

    if (std::string(argv[1]) == "--generate_vignette_map") {
        std::cout << "Generating vignette map..." << std::endl;
        const std::string img_load_path = argv[2];
        const std::string map_save_path = argv[3];
        auto map = estimateVignetting(loadImage(img_load_path), map_save_path);

        return 0;

    } else {
        if (argc < 5) {
            std::cerr << "Usage: Raw2RGB.exe input_path_folder output_path_folder gamma_value" << std::endl;
            return 1;
        }


        const std::string img_load_path = argv[1];
        const std::string img_save_path = argv[2];
        const double gamma = std::stof(argv[3]);
        const std::string vignette_path = argv[4];

        // Set default demosaic mode if not in arguments
        int demosaic_mode = 48;
        if (argv[5]) {
            demosaic_mode = std::stoi(argv[5]);
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
        processImages(raw_files, demosaic_mode, gamma, vignette_path);

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();

        // Calculate elapsed time in milliseconds
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        std::cout << "Total processing time: " << duration << " s" << std::endl;

        return 0;
    }
}