#include <iostream>
#include <cuda.h>

#define CUDA_CALL(call)     \
{                           \
cudaError_t result = call;  \
if (cudaSuccess != result) {       \
    std::cerr << "CUDA error: " << result << " in " << __FILE__ << ":" << __LINE__ << " : " << cudaGetErrorString(result) << " (" << call << ")" << std::endl; \
} \
}


__global__ void add(int a, int b, int *c) 
{
    *c = a + b;
}

int main(int argc, char *argv[])
{
    int c;
    int *dev_c;
    CUDA_CALL(cudaMalloc((void **) &dev_c, sizeof(int)))

    add<<<1, 1>>>(2, 7, &c);

    CUDA_CALL(cudaMemcpy(&c,
            dev_c,
            sizeof(int),
            cudaMemcpyDeviceToHost));

    printf("2 + 7 = %d\n", c);
    cudaFree(dev_c);
    return 0;

}
