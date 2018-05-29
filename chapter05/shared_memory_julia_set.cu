#include <cuda.h>
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024
#define PI 3.1415926535897932f

struct cuComplex {
    float r;
    float i;

    /* __device__ cuComplex(float a, float b): r(a), i(b) {} */
    __device__ cuComplex(float a, float b): r(a), i(b) {}
    __device__ float magnitude2(void) { return r * r + i * i; }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r , i + a.i);
    }
};

__device__ int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (float) (DIM/2 - x)/(DIM/2);
    float jy = scale * (float) (DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000) {
            return 0;
        }
    }
    return 1;
}

__global__ void kernel(unsigned char *ptr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.x * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;


    __shared__ float shared[16][16];
    // now calculate the value at that position
    const float period = 128.0f;

    // now calculate the value at that position
    int juliaValue = julia(x, y);

    shared[threadIdx.x][threadIdx.y] = 
        255 * (sinf(x*2.0f*PI/period) + 1.0f) *
        (sinf(y*2.0f*PI/period) + 1.0f) / 4.0f;

    // synchronization point between the write to shared memory and the
    // subsequent read from it
    __syncthreads();

    ptr[offset * 4 + 0] = 0;
    ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main(int argc, char *argv[])
{
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void **) &dev_bitmap, bitmap.image_size()));

    dim3 grid(DIM/16, DIM/16);
    dim3 threads(16, 16);
    kernel<<<grid, threads>>>(dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(),
                dev_bitmap,
                bitmap.image_size(),
                cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();
    cudaFree(dev_bitmap);
    return 0;
}
