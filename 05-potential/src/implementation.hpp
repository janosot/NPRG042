#ifndef CUDA_POTENTIAL_IMPLEMENTATION_HPP
#define CUDA_POTENTIAL_IMPLEMENTATION_HPP

#include "kernels.h"

#include <interface.hpp>
#include <data.hpp>

#include <cuda_runtime.h>

#include <iostream>

/*
 * Final implementation of the tested program.
 */
template<typename F = float, typename IDX_T = std::uint32_t, typename LEN_T = std::uint32_t>
class ProgramPotential : public IProgramPotential<F, IDX_T, LEN_T>
{

public:
	typedef F coord_t;		// Type of point coordinates.
	typedef coord_t real_t;	// Type of additional float parameters.
	typedef IDX_T index_t;
	typedef LEN_T length_t;
	typedef Point<coord_t> point_t;
	typedef Edge<index_t> edge_t;

private:
	index_t point_count;			// Number of points.
	index_t edge_count;				// Number of edges.

	edge_t *edges;					// Array of edges.
	LEN_T *edges_lengths;				// Array of edge lengths.
	point_t *cup_velocities_;
	index_t iter_num;
	point_t *points;

public:
	virtual void initialize(index_t points, const std::vector<edge_t>& edges, const std::vector<length_t> &lengths, index_t iterations)
	{
		/*
		 * Initialize your implementation.
		 * Allocate/initialize buffers, transfer initial data to GPU...
		 */
		point_count = points;
		edge_count = edges.size();

		// allocate memory on the device
		CUCH(cudaSetDevice(0));
		CUCH(cudaMalloc((void**)&edges, sizeof(edge_t) * edge_count));
		CUCH(cudaMalloc((void**)&edges_lengths, sizeof(LEN_T) * edge_count));
		CUCH(cudaMalloc((void**)&cup_velocities_, sizeof(point_t) * point_count));
		CUCH(cudaMalloc((void**)&points, sizeof(point_t) * point_count));



	}


	virtual void iteration(std::vector<point_t> &points)
	{
		/*
		 * Perform one iteration of the simulation and update positions of the points.
		 */
		if (iter_num == 0)
		{
			CUCH(cudaMemcpyAsync(cup_points_, points.data(), points.size() * sizeof(point_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}
		step(iter_num);
		iter_num++;
		CUCH(cudaMemcpyAsync(points.data(), cup_points_, points.size() * sizeof(point_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}


	virtual void getVelocities(std::vector<point_t> &velocities)
	{
		/*
		 * Retrieve the velocities buffer from the GPU.
		 * This operation is for vreification only and it does not have to be efficient.
		 */
		velocities.resize(point_count);

		CUCH(cudaMemcpyAsync(velocities.data(), cup_velocities_, point_count * sizeof(point_t), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}
};

#endif