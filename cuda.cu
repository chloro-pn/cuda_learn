#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::cerr << file << " " << line << " " << cudaGetErrorString(err) << std::endl;
    std::exit(-1);
  }
}

#define HANDLE_ERROR(err) HandleError(err, __FILE__, __LINE__)

// __global__修饰符告诉编译器，该函数在设备上运行
__global__ void add(const int* a, const int* b, int* c, int size) {
  // blockIdx.x Block的索引
  int block_index = blockIdx.x + blockIdx.y * gridDim.x;
  int index = block_index * blockDim.x + threadIdx.x;
  if (index < size)
    c[index] = b[index] + a[index];
  // __shared__ 标识共享内存，由一个线程块内的多个线程共享
  __shared__ float cache[32];
  int cache_index = threadIdx.x;
  cache[cache_index] = cache_index;
  // 对线程块中的线程进行同步，确保线程块中每个线程都执行完之前的语句后才执行下面的语句
  __syncthreads();
  // 由每个线程块的第一个线程执行规约操作
  if (threadIdx.x == 0) {
    int sum = 0;
    for(int i = 0; i < 32; ++i) {
      sum = sum + cache[i];
    }
    cache[0] = sum;
    c[block_index] = cache[0];
  }
}

extern "C" {
  void addKernel(int* a, int* b, int* c, int size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaSetDevice(0);
    // cudaMalloc在设备上分配内存，不能在主机代码中使用cudaMalloc分配的指针进行内存读写操作
    cudaMalloc((void**)&dev_a, sizeof(int) * size);
    cudaMalloc((void**)&dev_b, sizeof(int) * size);
    cudaMalloc((void**)&dev_c, sizeof(int) * size);
    cudaMemcpy(dev_a, a, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, sizeof(int) * size, cudaMemcpyHostToDevice);
    // 第一个参数指定执行核函数的并行线程块的数量(Block)，当使用数字时，默认为一维；使用dim3类型的变量指定2/3维数据
    // 第二个参数指定每个线程块中线程的数量
    // 这两个值都可以是多维的
    // 线程块的数量上限为65535
    // 每个线程块中线程的数量限制为maxThreadsPerBlock
    // 这些Block的集合也称为一个Grid
    // 线程块为2维度，共9个，每个块中有32个线程
    dim3 grid(3, 3);
    add<<<grid, 32>>>(dev_a, dev_b, dev_c, size);
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << " use time : ms" << elapsedTime << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
}

int main() {
  // 获取cuda设备的数量
  int count = 0; 
  HANDLE_ERROR(cudaGetDeviceCount(&count));
  std::cout << "cuda device's count " << count << std::endl;

  cudaDeviceProp prop;
  for(int i = 0; i < count; ++i) {
    // 获取指定cuda设备的属性
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    std::cout << "name : " << prop.name << std::endl;
    std::cout << "global mem(GB) : " << prop.totalGlobalMem / ( 1024 * 1024 * 1024.0) << std::endl;
  }

  int a[20];
  int b[20];
  int c[20];

  for(int i = 0; i < 20; ++i) {
    a[i] = i;
    b[i] = 2 * i;
    c[i] = 0;
  }

  addKernel(a, b, c, 20);
  for(int i = 0; i < 20; ++i) {
    std::cout << c[i] << " " ;
  }
  return 0;
}