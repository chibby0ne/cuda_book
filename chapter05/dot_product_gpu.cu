#include "../common/book.h"

#define imin(a, b) (a < b ? a: b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x; 

    float temp = 0; 
    while (tid < N) {
        temp  += a[tid] * b[tid];
        // the threads increment their indices by the total number of threads
        // to ensure we don't miss any elements and don't multiply a pair twice
        tid += blockDim.x * gridDim.x;
    }

    // store product results
    cache[cacheIndex] = temp;

    // sync threads. We need to guarantee that all the threads have finished
    // writing to cache before any thread starts reading from it
    __syncthreads();

    // reduce all products i.e: sum all the products

    //  This is an ilustration procedure done by reduction, if the blockDim.x
    //  where 8, after the first iteration. 
    //  After log2(blocDim.x) iterations we array to only one element (the
    //  partial sum) in cache[0]
    //
    // [0] [1] [2] [3] [4] [5] [6] [7]
    //  |   |   |   |   |   |   |   |
    //  +   +   +   + 
    // (4) (5) (6) (7)
    //  |   |   |   |
    // [0] [1] [2] [3]
    // 
    // for reductions, threadsPerBlock must be a power of 2
    // because of this code:
    int i = blockDim.x / 2;
    while (i != 0) {
        // only use half of the threads (only threads up to blockDim.x / 2)
        if (cacheIndex < i) {
            cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // since there is only one number that needs storing to the global memory
    // we use a single thread
    if (cacheIndex == 0) {
        // each entry of c, contains the sum produced by one of the parallel blocks.
        c[blockIdx.x] = cache[0];
    }

}

int main(int argc, char *argv[])
{
    float *a, *b, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // allocate the memory on the CPU
    a = (float *) malloc(N * sizeof(float));
    b = (float *) malloc(N * sizeof(float));
    partial_c = (float *) malloc(blocksPerGrid * sizeof(float));


    // allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void **) &dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_partial_c, blocksPerGrid * sizeof(float)));


    // fill in the host memory with data
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // copy the arrays 'a' and 'b' into GPU mem
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    // copy the array 'dev_partial_c' into CPU mem
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0;
    for (int i = 0; i < blocksPerGrid ; ++i) {
        sum += partial_c[i];
    }

    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("Does GPU value %.6g = %.6g?\n", sum, 2 * sum_squares((float)(N - 1)));

    // free device mem
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    // free host mem
    free(a);
    free(b);
    free(partial_c);

    return 0;
}
