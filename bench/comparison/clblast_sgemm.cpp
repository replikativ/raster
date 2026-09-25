// Opt-in CLBlast SGEMM oracle/benchmark for Raster's public gemm-mnk! canary.
// Build and invocation are documented in generated-kernel-protocol.md.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <clblast_c.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cl_int status, const char* operation) {
  if (status != CL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
  }
}

static std::string device_string(cl_device_id device, cl_device_info field) {
  size_t size = 0;
  check(clGetDeviceInfo(device, field, 0, nullptr, &size), "clGetDeviceInfo size");
  std::vector<char> value(size);
  check(clGetDeviceInfo(device, field, size, value.data(), nullptr), "clGetDeviceInfo value");
  return std::string(value.data());
}

static cl_device_id select_intel_gpu() {
  cl_uint platform_count = 0;
  check(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs count");
  std::vector<cl_platform_id> platforms(platform_count);
  check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs list");
  for (auto platform : platforms) {
    cl_uint count = 0;
    const auto status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &count);
    if (status == CL_DEVICE_NOT_FOUND) continue;
    check(status, "clGetDeviceIDs count");
    std::vector<cl_device_id> devices(count);
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, count, devices.data(), nullptr),
          "clGetDeviceIDs list");
    for (auto device : devices) {
      if (device_string(device, CL_DEVICE_VENDOR).find("Intel") != std::string::npos) return device;
    }
  }
  throw std::runtime_error("no Intel OpenCL GPU found");
}

static size_t positive_size(const char* text) {
  const std::string value(text);
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("shape values must be positive decimal integers");
  const auto result = std::stoull(value);
  if (result == 0 || result > 4096) throw std::runtime_error("shape value is outside 1..4096");
  return static_cast<size_t>(result);
}

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("usage: clblast_sgemm M N K");
    const auto m = positive_size(argv[1]);
    const auto n = positive_size(argv[2]);
    const auto k = positive_size(argv[3]);
    if (m * n * k > 64000000) throw std::runtime_error("reference work exceeds 64M products");

    std::vector<float> a(m * k), b(k * n), c(m * n, 0.0f);
    for (size_t i = 0; i < a.size(); ++i) a[i] = float(int(i % 13) - 6) / 8.0f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = float(int(i % 11) - 5) / 8.0f;
    const auto device = select_intel_gpu();
    cl_int status = CL_SUCCESS;
    const auto context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    check(status, "clCreateContext");
    const auto queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    check(status, "clCreateCommandQueue");
    const auto a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, a.size() * sizeof(float), nullptr, &status);
    check(status, "clCreateBuffer A");
    const auto b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, b.size() * sizeof(float), nullptr, &status);
    check(status, "clCreateBuffer B");
    const auto c_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, c.size() * sizeof(float), nullptr, &status);
    check(status, "clCreateBuffer C");
    check(clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, a.size() * sizeof(float), a.data(),
                               0, nullptr, nullptr), "upload A");
    check(clEnqueueWriteBuffer(queue, b_buffer, CL_TRUE, 0, b.size() * sizeof(float), b.data(),
                               0, nullptr, nullptr), "upload B");

    auto gemm = [&]() {
      cl_event before = nullptr, gemm_event = nullptr, after = nullptr;
      check(clEnqueueMarkerWithWaitList(queue, 0, nullptr, &before), "before marker");
      auto active_queue = queue;
      const auto result = CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo,
                                      CLBlastTransposeNo, m, n, k, 1.0f,
                                      a_buffer, 0, k, b_buffer, 0, n, 0.0f,
                                      c_buffer, 0, n, &active_queue, &gemm_event);
      if (result != CLBlastSuccess)
        throw std::runtime_error("CLBlastSgemm failed: " + std::to_string(result));
      if (active_queue != queue) throw std::runtime_error("CLBlast changed the measured queue");
      check(clEnqueueMarkerWithWaitList(queue, 0, nullptr, &after), "after marker");
      check(clWaitForEvents(1, &after), "wait for GEMM");
      cl_ulong start = 0, end = 0;
      check(clGetEventProfilingInfo(before, CL_PROFILING_COMMAND_END, sizeof(start), &start, nullptr),
            "before timestamp");
      check(clGetEventProfilingInfo(after, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
            "after timestamp");
      clReleaseEvent(before);
      clReleaseEvent(gemm_event);
      clReleaseEvent(after);
      if (end < start) throw std::runtime_error("OpenCL event clock went backwards");
      return end - start;
    };

    auto validate = [&]() {
      check(clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, c.size() * sizeof(float),
                                c.data(), 0, nullptr, nullptr), "download C");
      double max_error = 0.0;
      for (size_t row = 0; row < m; ++row) {
        for (size_t column = 0; column < n; ++column) {
          double sum = 0.0;
          for (size_t inner = 0; inner < k; ++inner)
            sum += double(a[row * k + inner]) * double(b[inner * n + column]);
          max_error = std::max(max_error, std::abs(double(c[row * n + column]) - double(float(sum))));
        }
      }
      if (max_error > 1.0e-4) throw std::runtime_error("SGEMM differs from the independent CPU oracle");
      return max_error;
    };

    for (int i = 0; i < 4; ++i) gemm();
    const auto pre_error = validate();
    std::vector<cl_ulong> samples;
    for (int i = 0; i < 12; ++i) samples.push_back(gemm());
    const auto post_error = validate();
    std::cout << std::setprecision(12)
              << "{:baseline :clblast-sgemm :precision :strict-f32"
              << " :shape [" << m << " " << n << " " << k << "]"
              << " :device " << std::quoted(device_string(device, CL_DEVICE_NAME))
              << " :driver " << std::quoted(device_string(device, CL_DRIVER_VERSION))
              << " :clblast-version [" << CLBLAST_VERSION_MAJOR << " "
              << CLBLAST_VERSION_MINOR << " " << CLBLAST_VERSION_PATCH << "]"
              << " :event-clock :queue-markers :warmups 4 :samples-ns [";
    for (auto sample : samples) std::cout << sample << " ";
    std::cout << "] :max-absolute-error " << std::max(pre_error, post_error) << "}\n";

    clReleaseMemObject(c_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(a_buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
