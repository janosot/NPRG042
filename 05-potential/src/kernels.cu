#include "kernels.h"

#define BLOCK_SIZE 64

/*
 * Sample Kernel
 */
__global__ void my_kernel(float *src)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	src[idx] += 1.0f;
}

/*
 * This is how a kernel call should be wrapped in a regular function call,
 * so it can be easilly used in cpp-only code.
 */
void kernel_potential_calculate_velocities (float *d_velocities, float *d_positions, float *d_forces, float *d_masses, int num_particles)
{
	int num_blocks = num_particles / BLOCK_SIZE;
	int num_threads = BLOCK_SIZE;
	my_kernel<<<num_blocks, num_threads>>>(d_velocities);
}
