/**
 * @file batch_image_filter.cu
 * @brief GPU-accelerated batch image processing using CUDA NPP.
 *
 * This program applies NPP (NVIDIA Performance Primitives) box filter
 * to a batch of images, demonstrating GPU-accelerated image processing
 * at scale. It supports configurable filter size, input/output directories,
 * and logs per-image and total processing times.
 */

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Check CUDA errors and exit on failure.
#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                << " - " << cudaGetErrorString(err) << std::endl;          \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

// Check NPP status and exit on failure.
#define CHECK_NPP(call)                                                    \
  do {                                                                     \
    NppStatus status = (call);                                             \
    if (status != NPP_SUCCESS) {                                           \
      std::cerr << "NPP error at " << __FILE__ << ":" << __LINE__          \
                << " - status: " << status << std::endl;                   \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

/**
 * @brief Configuration parsed from command-line arguments.
 */
struct Config {
  std::string input_dir = "sample_images";
  std::string output_dir = "output";
  int filter_width = 5;
  int filter_height = 5;
  std::string log_file = "processing_log.csv";
};

/**
 * @brief Parse command-line arguments into a Config struct.
 */
Config ParseArgs(int argc, char* argv[]) {
  Config config;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) break;
    std::string flag(argv[i]);
    std::string value(argv[i + 1]);
    if (flag == "-i") {
      config.input_dir = value;
    } else if (flag == "-o") {
      config.output_dir = value;
    } else if (flag == "-w") {
      config.filter_width = std::stoi(value);
    } else if (flag == "-h") {
      config.filter_height = std::stoi(value);
    } else if (flag == "-l") {
      config.log_file = value;
    }
  }
  return config;
}

/**
 * @brief List image files (.png, .jpg, .jpeg, .bmp) in a directory.
 */
std::vector<std::string> ListImageFiles(const std::string& dir) {
  std::vector<std::string> files;
  DIR* dp = opendir(dir.c_str());
  if (!dp) {
    std::cerr << "Error: cannot open directory " << dir << std::endl;
    return files;
  }
  struct dirent* entry;
  while ((entry = readdir(dp)) != nullptr) {
    std::string name(entry->d_name);
    std::string lower_name = name;
    for (auto& c : lower_name) c = std::tolower(c);
    if (lower_name.size() > 4) {
      std::string ext = lower_name.substr(lower_name.size() - 4);
      std::string ext5 = lower_name.size() > 5
                             ? lower_name.substr(lower_name.size() - 5)
                             : "";
      if (ext == ".png" || ext == ".jpg" || ext == ".bmp" ||
          ext5 == ".jpeg") {
        files.push_back(dir + "/" + name);
      }
    }
  }
  closedir(dp);
  std::sort(files.begin(), files.end());
  return files;
}

/**
 * @brief Apply NPP box filter to a single image.
 *
 * Loads the image with OpenCV, uploads to GPU, applies NPP box filter,
 * downloads result, and saves the output image.
 *
 * @param input_path  Path to the input image file.
 * @param output_path Path to save the filtered output image.
 * @param filter_w    Box filter kernel width (must be odd).
 * @param filter_h    Box filter kernel height (must be odd).
 * @return Processing time in milliseconds.
 */
double ProcessImage(const std::string& input_path,
                    const std::string& output_path,
                    int filter_w, int filter_h) {
  // Load image as 8-bit 3-channel BGR.
  cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Warning: could not read " << input_path << std::endl;
    return -1.0;
  }

  int width = img.cols;
  int height = img.rows;
  int channels = img.channels();
  int step_bytes = width * channels * sizeof(Npp8u);

  // Allocate device memory for source and destination.
  Npp8u* d_src = nullptr;
  Npp8u* d_dst = nullptr;
  int d_step_src = 0;
  int d_step_dst = 0;

  d_src = nppiMalloc_8u_C3(width, height, &d_step_src);
  d_dst = nppiMalloc_8u_C3(width, height, &d_step_dst);

  if (!d_src || !d_dst) {
    std::cerr << "Error: NPP memory allocation failed for " << input_path
              << std::endl;
    return -1.0;
  }

  // Copy host image to device.
  CHECK_CUDA(cudaMemcpy2D(d_src, d_step_src, img.data, step_bytes,
                           width * channels * sizeof(Npp8u), height,
                           cudaMemcpyHostToDevice));

  // Define ROI and filter parameters.
  // The ROI excludes the border so the filter doesn't read out of bounds.
  int border_x = filter_w / 2;
  int border_y = filter_h / 2;
  NppiSize roi = {width - 2 * border_x, height - 2 * border_y};
  NppiSize mask_size = {filter_w, filter_h};
  NppiPoint anchor = {border_x, border_y};

  // Offset source pointer to start of valid ROI.
  Npp8u* d_src_roi = d_src + border_y * d_step_src + border_x * channels;
  Npp8u* d_dst_roi = d_dst + border_y * d_step_dst + border_x * channels;

  // Copy source to destination first so border pixels are preserved.
  CHECK_CUDA(cudaMemcpy2D(d_dst, d_step_dst, d_src, d_step_src,
                           width * channels * sizeof(Npp8u), height,
                           cudaMemcpyDeviceToDevice));

  // Start GPU timing.
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));

  // Apply NPP box filter on the valid ROI.
  CHECK_NPP(nppiFilterBox_8u_C3R(
      d_src_roi, d_step_src, d_dst_roi, d_step_dst, roi, mask_size, anchor));

  // Stop GPU timing.
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

  // Copy result back to host.
  cv::Mat output(height, width, CV_8UC3);
  CHECK_CUDA(cudaMemcpy2D(output.data, step_bytes, d_dst, d_step_dst,
                           width * channels * sizeof(Npp8u), height,
                           cudaMemcpyDeviceToHost));

  // Save output image.
  cv::imwrite(output_path, output);

  // Cleanup.
  nppiFree(d_src);
  nppiFree(d_dst);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return static_cast<double>(elapsed_ms);
}

/**
 * @brief Print usage information.
 */
void PrintUsage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]\n"
            << "Options:\n"
            << "  -i <dir>   Input image directory (default: sample_images)\n"
            << "  -o <dir>   Output image directory (default: output)\n"
            << "  -w <int>   Filter kernel width, must be odd (default: 5)\n"
            << "  -h <int>   Filter kernel height, must be odd (default: 5)\n"
            << "  -l <file>  Log file path (default: processing_log.csv)\n"
            << std::endl;
}

int main(int argc, char* argv[]) {
  Config config = ParseArgs(argc, argv);

  std::cout << "=== GPU Batch Image Box Filter ===" << std::endl;
  std::cout << "Input directory:  " << config.input_dir << std::endl;
  std::cout << "Output directory: " << config.output_dir << std::endl;
  std::cout << "Filter size:      " << config.filter_width << "x"
            << config.filter_height << std::endl;

  // Print GPU device info.
  int device_id = 0;
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
  std::cout << "GPU Device:       " << prop.name << std::endl;
  std::cout << std::endl;

  // Discover input images.
  std::vector<std::string> image_files = ListImageFiles(config.input_dir);
  if (image_files.empty()) {
    std::cerr << "Error: no image files found in " << config.input_dir
              << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Found " << image_files.size() << " images to process."
            << std::endl;

  // Open log file.
  std::ofstream log_file(config.log_file);
  log_file << "image,width,height,filter_size,gpu_time_ms" << std::endl;

  // Process each image.
  double total_time_ms = 0.0;
  int processed_count = 0;

  auto wall_start = std::chrono::high_resolution_clock::now();

  for (const auto& input_path : image_files) {
    // Extract filename for output.
    std::string filename = input_path.substr(input_path.find_last_of("/") + 1);
    std::string output_path = config.output_dir + "/filtered_" + filename;

    // Read dimensions for logging.
    cv::Mat tmp = cv::imread(input_path, cv::IMREAD_COLOR);
    if (tmp.empty()) continue;
    int w = tmp.cols;
    int h = tmp.rows;
    tmp.release();

    double gpu_time = ProcessImage(input_path, output_path,
                                   config.filter_width, config.filter_height);

    if (gpu_time >= 0) {
      total_time_ms += gpu_time;
      processed_count++;

      std::cout << "  [" << processed_count << "/" << image_files.size()
                << "] " << filename << " (" << w << "x" << h << ")"
                << " - GPU: " << gpu_time << " ms" << std::endl;

      log_file << filename << "," << w << "," << h << ","
               << config.filter_width << "x" << config.filter_height << ","
               << gpu_time << std::endl;
    }
  }

  auto wall_end = std::chrono::high_resolution_clock::now();
  double wall_time_ms = std::chrono::duration<double, std::milli>(
                            wall_end - wall_start)
                            .count();

  // Print summary.
  std::cout << "\n=== Processing Summary ===" << std::endl;
  std::cout << "Images processed:    " << processed_count << std::endl;
  std::cout << "Total GPU time:      " << total_time_ms << " ms" << std::endl;
  std::cout << "Total wall time:     " << wall_time_ms << " ms" << std::endl;
  if (processed_count > 0) {
    std::cout << "Avg GPU time/image:  " << total_time_ms / processed_count
              << " ms" << std::endl;
  }
  std::cout << "Log saved to:        " << config.log_file << std::endl;
  std::cout << "Output images in:    " << config.output_dir << std::endl;

  log_file.close();
  CHECK_CUDA(cudaDeviceReset());
  return EXIT_SUCCESS;
}
