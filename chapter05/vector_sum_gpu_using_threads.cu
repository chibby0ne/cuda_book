#include "../common/book.h"

#define N (33 * 1024)

__global__ void add(long int *a, long int *b, long int *c)
{
   int tid = threadIdx.x + blockIdx.x * blockDim.x;
   while (tid < N) {
       c[tid] = a[tid] + b[tid];
       // in the GPU implementation we consider the number of parallel threads
       // launched to be the number of processors, although the actual GPU may
       // have fewer processing units that this, we think of each thread as
       // logically executing in parallle and the allow the hardware to
       // scheudle the actual execution
       tid += blockDim.x * gridDim.x;
   }
}

int main(int argc, char *argv[])
{
    long int a[N], b[N], c[N];
    long int *dev_a, *dev_b, *dev_c;

    HANDLE_ERROR(cudaMalloc((void **) &dev_a, N * sizeof(long int)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_b, N * sizeof(long int)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_c, N * sizeof(long int)));

    // fill the arrays 'a' and 'b' on the CPU
    for (long int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(long int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(long int), cudaMemcpyHostToDevice));

    add<<<128, 128>>>(dev_a, dev_b, dev_c);

    // copy the arrays 'c' back from the GPU to the CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(long int), cudaMemcpyDeviceToHost));

    // display the results
    for (int i = 0; i < N; ++i) {
        printf("%ld + %ld = %ld\n", a[i], b[i], c[i]);
    }

    bool success = true;
    for (int i = 0; i < N; ++i) {
        if ((a[i] + b[i]) != c[i]) {
            printf("Error: %d + %d != %d\n", a[i], b[i], c[i]);
            success = false;
        }
    }
    if (success) {
        printf("We did it!\n");
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
