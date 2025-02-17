#include <iostream>
#include "version.h"
#include "argparse.hpp"
#include "common_types.h"
#include "image_process.h"

int main(const int argc, char** argv) {

    // Arguments and flags
    argparse::ArgumentParser program("Raw2RGB", PROJECT_VERSION);

    // --generate_vignette_map mode
    program.add_argument("--generate_vignette_map")
        .help("Generate a vignette map from a given image")
        .default_value(false)
        .implicit_value(true);

    // Positional arguments
    program.add_argument("input_path")
        .help("Path to the input folder or image (depending on the operation)");
    program.add_argument("output_path")
        .help("Path to the output folder or file (depending on the operation)");

    // Optional arguments
    program.add_argument("--gamma")
        .help("Gamma correction value (default is 0.7)")
        .default_value(0.7)
        .scan<'g', double>();  // Parse into a double
    program.add_argument("--vignette_path")
        .help("Path to an existing vignette map")
        .default_value(std::string("vignette_map.bin"));  // Default vignette map
    program.add_argument("--demosaic_mode")
        .help("Demosaic mode to use (default is 48)")
        .default_value(48)
        .scan<'i', int>();


    // Run
    try {
        // Parse command-line arguments
        program.parse_args(argc, argv);

        if (program.get<bool>("--generate_vignette_map")) {
            std::cout << "Generating vignette map..." << std::endl;
            const auto img_load_path = program.get<std::string>("input_path");
            const auto map_save_path = program.get<std::string>("output_path");
            auto map = estimateVignetting(loadImage(img_load_path), map_save_path);
            std::cout << "Saved vignette map at: " << map_save_path << std::endl;
            return 0;

        } else {

            const auto img_load_path = program.get<std::string>("input_path");
            const auto img_save_path = program.get<std::string>("output_path");
            const auto gamma = program.get<double>("--gamma");
            const auto vignette_path = program.get<std::string>("--vignette_path");
            const int demosaic_mode = program.get<int>("--demosaic_mode");

            // Get list of raw files
            const std::vector<Image> raw_files = getRawFiles(img_load_path, img_save_path);
            if (raw_files.empty()) {
                std::cerr << "No raw files found in " << img_load_path << std::endl;
                return 1;
            }

            // Start timing processImages
            const auto start_time = std::chrono::high_resolution_clock::now();

            // Process the images
            processImages(raw_files, demosaic_mode, gamma, vignette_path);

            // End timing
            const auto end_time = std::chrono::high_resolution_clock::now();

            // Calculate elapsed time in milliseconds
            const auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
            std::cout << "Total processed files: " << raw_files.size() << std::endl;
            std::cout << "Total processing time: " << duration << " s" << std::endl;

            return 0;
        }
    } catch (const std::runtime_error& err) {
        // Catch errors related to argument parsing
        std::cerr << "Error: " << err.what() << std::endl;
        std::cerr << program; // Display the usage guide
        return 1;
    }
}