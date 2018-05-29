/*
* This simple ray tracer will fiare a ray from each pixel and keep track of
* which rays hit which spheres. It will also track the depth of each of these
* hits. In the case where a ray passes thought multiple spheres, only the
* sphere closest to the camera can be seen. 
*
* This ray tracer will only support scenes of spheres, and the camera is
* restricted to the z-axis, facing the origin.
*
*/
#include <cuda.h>
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define INF 2e10f
#define SPHERES 20
#define rnd(x) (x * rand() / RAND_MAX)
#define DIM 10

struct Sphere {
    float r, b, g;
    float radius;
    float x, y, z;

    /** Computers whether the ray (from pixel ox, oy) intersects the sphere,
     * and if it does it computes the distance from the camera where the ray
     * hits the sphere.
     */
    __device__ float hit(float ox, float oy, float *n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};

// When wer use the modifier __constant__ we also change the declaration from
// pointer to statically allocated array. We no longer need cudaMalloc or
// cudaFree, but we do need to cmmit to a size for this array at compile-time.
__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char *ptr)
{
    // map from threadIdx/blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    // shift image to center on z-axis
    float ox = (x - DIM/2);
    float oy = (y - DIM/2);

    // for each ray check each sphere for intersection (hit)
    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; ++i) {
        float n;
        float t = s[i].hit(ox, oy, &n);
        // if the sphere is hit by the ray from pixel and is closer to the
        // camera than the last sphere we hit?
        if (t > maxz) {
            // store the depth and store the color.
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
        }
    }
    
    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

int main(int argc, char *argv[])
{
    // capture the start time
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR(cudaMalloc((void **) &dev_bitmap, bitmap.image_size()));

    // allocate memory for the Sphere dataset
    /* HANDLE_ERROR(cudaMalloc((void **)&s, sizeof(Sphere) * SPHERES)); */

    // Generate a random array of 20 ss, copy it to memory on device and
    // then free our temp memory
    Sphere *temp_s = (Sphere *) malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; ++i) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(1000.0f) + 20;
    }

    /* HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice)); */
    //
    // Special version of `cudaMemcpy when we copy from host memory to constant
    // memory in device.  cudaMemcpy and cudaMemcpyHostToDevice copies to
    // global memory, while cudaMemcpyToSymbol copies to constant memory
    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
    free(temp_s);

    // generate the bitmap from our sphere data
    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap);

    // copy our bitmat back from the GPU for display
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    // get stop time, and display timing results, and destroy events
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate: %3.1f ms\n", elapsedTime);

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    bitmap.display_and_exit();

    // free our memory
    cudaFree(dev_bitmap);
    cudaFree(s);

    return 0;
}
