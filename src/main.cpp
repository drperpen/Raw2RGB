
#include <iostream>
#include "common_types.h"
#include "image_process.h"


int main(const int argc, char** argv){

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
        demosaic_mode = std::stoi(argv[3]);
    }

    const std::vector<Image> raw_files = getRawFiles(img_load_path, img_save_path);
    if (raw_files.empty()) {
        std::cerr << "No raw files found in " << img_load_path << std::endl;
        return 1;
    }

    processImages(raw_files, demosaic_mode, gamma);


    return 0;
}
