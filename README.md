# Raw2RGB

Raw2RGB is a C++ project designed for efficiently processing raw images, such as those captured by cameras in Bayer pattern format. It includes various image processing features like dead pixel correction, vignette removal, gamma correction, and demosaicing. The library is particularly suited for batch processing of raw images with an emphasis on performance and accuracy.

---

## Features

- **Load Images**: Load `.tiff` raw images with depth information.
- **Dead Pixel Correction**: Automatically detects and adjusts dead pixels in raw images.
- **Vignette Removal**: Apply a polynomial-based vignette correction model.
- **Gamma Correction**: Normalize pixel intensity values using custom gamma parameters.
- **Demosaicing**: Convert Bayer pattern raw images into a color image.
- **Batch Processing**: Process multiple raw images in a folder with output to a given directory.
- **Camera Undistortion**: Remove distortion based on camera matrix and distortion parameters.

---

## Project Structure

- **`image_process.h`**: Header file defining structures and functions for image processing.
- **`main.cpp`**: Main driver file (details are not shown but assumed to be present for executing the pipeline).
- **Functions**:
    - `loadImage`: Load an image into OpenCV's `cv::Mat` structure.
    - `correctDeadPixels`: Identifies and corrects artificial noise caused by dead pixels.
    - `applyVignetteCorrection`: Removes vignetting using polynomial correction.
    - `processImages`: Handles the entire image processing pipeline, including loading, correcting, demosaicing, and saving.
    - `saveImage`: Saves processed images to the given path.
    - `estimateVignetting`: Estimates vignette correction models from input raw images.

---

## Requirements

### Dependencies

- **OpenCV** (3.4+):
    - Required for image processing functions like demosaicing, median filtering, and basic matrix operations.

- **C++17 or Newer**:
    - The project makes use of modern C++ features like `std::filesystem` and `std::array`.

- **Compiler**:
    - GCC, Clang, or MSVC with C++17 support.

- **CMake**:
    - Used for building the project.

### Libraries and Frameworks

- [OpenCV](https://opencv.org/): Computer Vision Library
- [std::filesystem](https://en.cppreference.com/w/cpp/filesystem): File system operations for image loading and output

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/Raw2RGB.git
   cd Raw2RGB
   ```

2. Install dependencies:
    - Ensure OpenCV is installed on your system:
      ```bash
      sudo apt install libopencv-dev     # For Ubuntu/Debian
      brew install opencv               # For macOS
      ```

3. Build the project:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

4. Run the application:
   ```bash
   ./Raw2RGB <input_folder> <output_folder>
   ```

---

## Usage

### Command-Line Interface (CLI)

The project processes raw `.tiff` files in a folder and applies the image corrections.

**Example Command:**
```bash
./Raw2RGB raw_images/ processed_images/
```

### Processing Images

1. **Input Folder**: Place `.tiff` images in the specified input directory.
2. **Output Folder**: Processed `.png` images will be saved in the output directory.

---

## Configuration

You can configure several parameters to control the image processing pipeline:
- **Gamma**: Adjust gamma correction for desired brightness distribution.
- **Dead Pixel Correction Factors**:
    - Brightness threshold (`brightFactor`)
    - Darkness threshold (`darkFactor`)
- **Vignetting**:
    - Customize parameters `a`, `b`, `c` for vignette correction using `VignetteModel`.

These parameters can either be hardcoded or provided via configuration files (future work may include CLI options).

---

## Supported File Formats

- **Input**: `.tiff` images (supports 8-bit, 16-bit grayscale Bayer)
- **Output**: `.png` images (output in RGB format)

---

## Examples

### Example Workflow

- Place raw `.tiff` files in a folder called `raw_images/`.
- Run the command:
  ```bash
  ./Raw2RGB raw_images/ output_images/
  ```
- View processed files in the `output_images/` folder.

### Sample Output
Raw `.tiff` images are processed and saved as corrected `.png` files.

---

## Performance

- SIMD optimizations using **AVX**, **AVX-512**, and **OpenMP** ensure parallelized and vectorized computations for image processing.
- The project is tested with large raw image datasets to ensure scalability and speed.

---

## Future Improvements

- Add CLI options for parameter configuration.
- Extend support for additional raw image formats.
- Integrate further image correction features like noise reduction.

---

## Contribution

1. Clone the repository:
   ```bash
   git clone https://github.com/username/Raw2RGB.git
   ```

2. Create a new branch for your feature:
   ```bash
   git checkout -b feature_name
   ```

3. Push and create a pull request:
   ```bash
   git push origin feature_name
   ```

We welcome contributions from the community to expand the project's capabilities and usability.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- This application leverages [OpenCV](https://opencv.org/) for efficient image processing.
- Special thanks to contributors and community feedback that helped shape this tool.