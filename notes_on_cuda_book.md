# Chapter 3

Host
CPU and its memory

Device
GPU and its memory

Kernel
A function that executes on the device

__global__
CUDA C qualifier. It tells the compiler that the function should be compiled
to run on a device instead of the host.


<<< >>>
For a function, denote that tell the device how to setup the runtime.

We need to allocate memory on it before doing anything useful on the device ,
such as return values to the host.

cudaMalloc()
Tells the CUDA rutnime to allocate the memory on the device
First argument is a pointer to the pointer you want to hold the address of the
newly allocated memory, the second is the size of the allocation you want to
make.

NOTE: Do not dereference the pointer returned by cudaMalloc in code that executes on
the host!

Host code can:
* pass the pointer around
* do pointer arithmetic
* cast it to a different type

You *can* pass pointer allocated with cudaMalloc to functions that execute on
the device.

You *can* use pointers allocated with cudaMalloc to read/write memory from code
that executes on the device.

You *can* pass pointers allocated with cudaMalloc to functions that execute on
the host.

You *cannot* use pointers allocated with cudaMalloc to read/write memory from
code that executes on the host.

The same logic applies in for host pointers in the device.

Summary:
Host pointers can access memory from host memory
Device pointers can access memory from device memory

cudaFree()
Exactly like free()
Frees memory allocated by cudaMalloc().

Two of the most common methos for accessing device memory:
* Using device pointers (allocated with cudaMalloc)
* using cudaMemcpy

cudaMemcpy(void \*dest, void \* src, size_t size, flag cudaMemcpyFromTo)
Last parameter tells whether dest (and therefore src) is from Host or Device.
cudaMemcpyFromHostToDevice: src is from host, dest is to device
cudaMemcpyFromDeviceToHost: src is from device, dest is to host
cudaMemcpyFromDeviceToDevice: src is from device, dest is to device

If both src, dst are on the device then we would use memcpy()


## Chapter 4

Explanation of the numbers insides the triple angle brackets
add<<<N, 1>>>(a, b, c);

The first number is the number of parallel blocks in which we would like 
the device to execute our kernel.

We can think of the runtime creating N copies of the kernel and running them
in parallel.

We call each of these parallel invocations a *block*.

Block
One of the parallel copies of the program that get execute in parallel created
by the CUDA runtime.

How can the GPU tell from within the code which block it is currently running?

* blockIdx.x

Is one of the built-in variables that the CUDA runtime defines.
It contains the value of the block index for whichever block is currently
running the device code.

Why `blockIdx.x`? Why not just `blockIdx`?

Because CUDA allows you to define a group of blocks in two dimensions.
For problems with two dimensionsonal domains such as matrix math or image
processing it is conveninet to use two-dimensional indeixing to avoid annoying
translations from linear to rectangular indices.


Grid
A collection of parallel blocks. Can be a 1-dimensional or 2-dimensional
collection of blocks.

Each copy of the kernel can determine which block it is executing using the
builtin variable `blockIdx`

This specifies to the runtime system that we want a one-dimensional grid of N
blocks. (scalar values are one-dimensional).
THe GPU threads will have varying values of blockIdx.x (from 0 to N-1 due to
the if check and the number N)


Important:
*Check the results of every operation that can fail*.
That why we check for if (tid < N).

Julia set example:

dim3 is a CUDA built-in type to encapsulate multidimensional tuples.
The dim3 represents a 3-dimensional tuple so why do we use it for
2-dimensional grid?

Well dim3 value is what the CUDA runtime expects, and not specifying the 3rd
parameters the cuda runtime just assumes that last dimension is 1.

We use this dim3 type to pass it to the kernel to specify the grid size.

__global__
Function can be called from the host BUT run on the device.

__device__
Function can be called from __global__ function and from __device__ functions only.

gridDim
Another built-in variable.
Is a constant across all blocks and simply holds the dimensions of the grid
that was launched (with the dimensions of it, i.e gridDim.x and gridDim.y for
a 2-dimensional grid)

## Chapter 5



The CUDA runtime allows blocks to be split into threads

add<<<N, 1>>>(dev_a, dev_b, dev_c)

The first argument represents the number of blocks we want the CUDA runtime to
create on our behalf..

The second argument represents the numbers of threads per block we want the
CUDA runtime to create on our behalf.

In this previous examplae we launched:

N blocks x 1 thread/block = N parallel threads.

We could have launched N/2 blocks with 2 threads per block, N/4 blocks with 4
threads per blocks...

Parallel threads withing a block have the ability to do thins that parallel
blocks cannot do.

*NOTE: The hardware limits the numbers of blocks in a single launch to 65535
(2^16 -1) or probably maxThreadsDims[0] or maxThreadsDims[1] or
maxThreadsDims[2] field of the properties structure*

*NOTE: the hardware limits the number of threads per block to the
`maxThreadsPerBlock` field from the device properties structure. For GTX 1070
is 1024.*

How do we use a thread-based approach to add two vectors of greater size than
1024?
We use a combination of threads and blocks.

```
int tid = threadIdx.x + blockIdx.x * blockDim.x;
```

blockDim
Is a constant across all and stores the numbers of threads along each
dimension of the block.
It is a 3-dimensional value, like gridDim (although gridDim is only
2-dimensional? maybe not anymore for CUDA capability > 6.0 )

Solution:
Set an arbitrarily set the block size to some fixed number of threads: let's
say 128.

And then we set the number of blocks to N / 128, but this is a integer
division therefore any value of N < 128 will render to the number of blocks to
0, so we do a ceiling operation: (N + 127)/128

But since this will create 1 more threads than was is needed, we need the
check for the thread id.

```
if (tid < N) {
// do the operations
}
```


Important:
In the GPU implementation we consider the number of parallel threads launched
to be the number of processors, although the actual GPU may have fewer
processing units that this, we think of each thread as logically executing in
parallel and the allow the hardware to schedule the actual execution.

Related:
For the vector sum:
We increment the tid and do a while loop, because we have less "cores"
(threads, but we treat them as cores according to the previous paragraph) than
the number of elements we want to sum.
In a very similar way to the CPU implementation.


Although we routinely create and schedule this many threads on a GPU, one
would not dream of creating this many threads on a aCPU. Because CPU thread
management and scheduling must be done in software it simply cannot scale to
the number of threads that a GPU can. Because we can simply create a thread
for each data element we want to process, parallel programming on a GPU can be
far simpler than on a CPU.


* Shared Memory and Syncrhonization
So far the motivation for splitting blocks into threads was simply one of
working around HW limitations to the number of blocks we can have running.
There is another reason:

CUDA C makes available a region of memory called **shared memory**.
This memory region brings along another extension to the C language:
__shared__

__shared__:
Variables in shared memory are treated differently than typical variables by
the CUDA C Compiler.
It creates a copy of the variable for each block that you launch on the GPU.

Every thread in the block shares the memory bu threads cannot see or modify
the copy of this variable that is seen within other blocks, i.e threads from
different blocks cannot read/modify the shared variable of a another block.

Provides a way by which threads within a block can communicate and collaborate.
It is also fast since shared memory buffers reside physically on the GPU.

We also need a synchronization mechanism to avoid race conditions within
threads of the same block.

__syncthreads():
Guarantees that every thread in the block has completed instructions prior to
the __syncthreads() before the HW will execute the next instruction on any
thread.

Reduction
The process of taking an input array and performing some computations that
produce a smaller array of results.

More on reduction (Reason why we exit the kernel before the dot product
computation is complete)
In the case of a dot product between vectors, we always product exactly one
output regardless of the size of out input. It turns out GPUs tend to waste
its resources when performing the last steps of a reduction since the size of
the data is so small at that point.

Thread divergence:
When some threads need to do something while others don't.

Under normal circumstances divergent branches simply result in some threads
remaining idle, while the other threads actually execute the instructions on
in the branch.

__syncthreads():
CUDA architecture guarantees that *no thread* will advance to an instruction
byond the __syncthreads() until *every* thread in the block has executed the
__syncthread().

In the dot product of vectors:
If we move the second __syncthreads() call inside the if block, what will
happen is that the hardware will continue to way for these threads that have
not executed the __syncthreads() (and since the conditions is only reached in
half of the threads in the thread blocks), the GPU will freeze.
And because the GPU is waiting the CPU will also hang.

Important:
We add a syncrhonization point (`__syncthreads()`) between the write to shared
memory and the subsequent read from it and vice versa.

## Chapter 6
Constant memory and events

Constant memory:
We use for data that will not change over the course of a kernel execution.
In some situations using constant memory rather than global memory will reduce
the required memory bandwidth.

NVIDIA HW provides 64KB is constant memory, this memory is read-only i.e:
written in the host it can't be changed by the device.

There are two reasons why reading from the 64KB of constant memory can save
bandwith over standards reads of global memory:
1. A single read from constant memory can be broadcast to other *nearby* threads
   effectively saving up to (in the case of the ray tracer example) 15 reads.
2. Constant memory is cached, so consecutive reads of the same address will
   not incur any extra memory traffic.

What do they mean by *nearby*.

Warp 
From the weaving world, the group of threads being woven together into fabric
THe collection of 32 threads that are "woven together" and get executed in
lockstep. At every line in your program, each thread in a warp executes the
same instrction on different data.

When it comes to handling constant memory, NVIDIA HW can broadcast a single
memory read to each half-warp. A half-warp is a group of 16 threads.

That is, if every thread in a half-warp requrest data from the same address in
constant memory your GPU will generate only a single read request and
susequently broadcast the data to every thread. 

If you're reading a lot of data from constant memory you will generate only
1/16 (~ 6%) of the memory traffic as you would using global memory.

In addition because the memory is unchanged the HW can aggresively cache the
constant data on the GPU. So after a first read from an address in constant
memory, other half-warps' requesting the same address and therefore hitting
the constant cache, will generate no additional memory traffic.

Potential downside of using constant memory:
The half warp broadcast feature can actually slow performance.
It actually slows poerformance when all 16 threads read different address.

The tradeoff to allowing the broadcast of a single read to 16 threads is that
the 16 threads are allowed to place only a single read request at a time.
That is, If all 16 threads in a half-warp need different data from constant
memory, the 16 different reads get serialized taking 16 times the amount of
time to place the request. If they were using conventional global memory the
request could be issued at the same time.

To measure the time a GPU spends on a task we use the CUDA event API.

Event in CUDA
A GPU time stamp that is recorder at a user-specified point in time.  Since
the GPU itself is recording the time stamp it eliminates a lot of the problems
we might encounter when trying to time GPU execution with CPU times.

Using the Event API is easy:
1. Create an event 
2. Record an event

At the beginning of the code we create/record a start event.

To time a block of code, we will want to create both a start event and a stop
event.

When we launch our kernel the GPU beings executing our coude but the CPU
continues executing the next line of our program before the GPU finishes. This
makes timing like this tricky.
You should imagine calls to cudaEventRecord() as an instruction to record the
current time, being placed into the GPU's pending work queue. As a result, out
event won't actually be recorded until the GPU finishes everything prior to
the call to cudaEventRecord(). But we cannot *safely read* the value of the
stop event until the GPU has completed its prior work and recorded the stop
event. -> We need to tell the CPU to synchronize on an event.

For example:
```
cudaEventRecord(stop)
cudaEventSynchronize(cudaEvent_t stop)
```

cudaEventSynchronize(cudaEvent_t stop)
Tells the CPU to synchronize on an event.
Tells the runtime to block further instruction until the GPU has reached the
`cudaEvent_t stop.`
We cudaEventSynchronize returns we know that all GPU work before stop event
has complete so it is safe to read the time stamp recorded in stop.

cudaEventElapsedTime(float \*elapsedTime, cudaEvent_t start, cudaEvent_t stop)
Computes the elapsed time between two previously recorded events and returns
the value in the elapsedTime variable in miliseconds.

cudaEventDestroy(cudaEvent_t event)
Frees the memory of the event, just like cudaFree() does to a variable.

# Chapter 7

Texture Memory
You can use texture memory in CUDA.
It is cached on chip, so in some situation sit will provide higher effective
bandwith by reducing accesses to off-chip DRAM.

*Textue caches are designed for graphics applications where memory addreses
patterns exhibit strong spatial locality*, i.e: in a general computing
application it means that thread is like to read from an address "near" the
addres that nearby threads read threads read.

Example:

[ ][ ][ ][ ]
[ ][x][x][ ]
[ ][x][x][ ]
[ ][ ][ ][ ]

Thread 0 -> x
Thread 1 -> x
Thread 2 -> x
Thread 3 -> x

Arithmetically the four addresses are not consecuitive so the would not be
cached together in a typical CPU caching scheme. But since GPU texture caches
are designed to accelerate access patters like this one.

So this using texture memory instead of global memory for this cases, would
increase performance.


Example: 

Two dimensional head transfer simulation.
