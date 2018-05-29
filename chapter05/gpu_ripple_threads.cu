#include "../common/cpu_anim.h"
#include "../common/book.h"

#define DIM 480

struct DataBlock {
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
};

__global__  void kernel(unsigned char *ptr, int ticks)
{
    /*
       This is a representation of the way the blocks in the grid and threads
       in the block, would look like with their respective coordinates. In our
       case the blocks dimensions are 16x16 instead of 6x6 and grid dimensions
       are DIM/16 x DIM/16, instead of 2x2.
       
       Block (0, 0)        Block(1, 0)

       [] [] [] [] [] []  [] [] [] [] [] []
       [] [] [] [] [] []  [] [] [] [] [] []
       [] [] [] [] [] []  [] [] [] [] [] []
       [] [] [] [] [] []  [] [] [] [] [] []
       [] [] [] [] [] []  [] [] [] [] [] []
       [] [] [] [] [] []  [] [] [] [] [] []

       Block (0, 1)        Block(1, 1)

       [] [] [] [] [] []  [] [] [] [] [] []
       [] [] [] [] [] []  [] [] [] [] [] []
       [] [] [] [] [] []  [] [] [] [] [] []
       [] [] [] [] [] []  [] [] [] [] [] []
       [] [] [] [] [] []  [] [] [] [] [] []
       [] [] [] [] [] []  [] [] [] [] [] []

       Where each [] is a thread with its own coordinates
       [Thread (0, 0)] [Thread (1, 0)] [Thread (2, 0)] [Thread (3, 0)] [Thread (4, 0)] [Thread (5, 0)]
       [Thread (0, 1)] [Thread (1, 1)] [Thread (2, 1)] [Thread (3, 1)] [Thread (4, 1)] [Thread (5, 1)]
       [Thread (0, 2)] [Thread (1, 2)] [Thread (2, 2)] [Thread (3, 2)] [Thread (4, 2)] [Thread (5, 2)]
       [Thread (0, 3)] [Thread (1, 3)] [Thread (2, 3)] [Thread (3, 3)] [Thread (4, 3)] [Thread (5, 3)]
       [Thread (0, 4)] [Thread (1, 4)] [Thread (2, 4)] [Thread (3, 4)] [Thread (4, 4)] [Thread (5, 4)]
       [Thread (0, 5)] [Thread (1, 5)] [Thread (2, 5)] [Thread (3, 5)] [Thread (4, 5)] [Thread (5, 5)]

       Note the coordinate system IS DIFFERENT from the way you index a matrix.
       It is similar to the cartesian system starting with the origin (0,0) at the top left.

    */
    // map from threadIdx/BlockIdx to pixel positions
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // linearize these x and y values (try to visualize this as converting 2d matrix to a contiguos array positions)
    int offset = x + y * blockDim.x + gridDim.x;

    // now calculate the value at that position
    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);

    unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));
    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = grey;
    ptr[offset*4 + 3] = 255;
}

void generate_frame(DataBlock *d, int ticks)
{
    dim3 blocks(DIM/16, DIM/16);
    dim3 threads(16, 16);

    kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);

    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(),
                d->dev_bitmap,
                d->bitmap->image_size(),
                cudaMemcpyDeviceToHost));
}

void cleanup(DataBlock *d) {
    cudaFree(d->dev_bitmap);
}

int main(int argc, char *argv[])
{
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR(cudaMalloc((void **)&data.dev_bitmap, bitmap.image_size()));

    bitmap.anim_and_exit((void (*)(void *, int))generate_frame, (void (*)(void *))cleanup);
    return 0;
}

