#ifndef __HAIR_H__
#define __HAIR_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "tools.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand_kernel.h>


struct ModelOBJ
{
	float* vertices;
	float* normals;
	float* faces;	//Triangles
	
	long bytes;
	long totalConnectedPoints;
	long totalConnectedTriangles;
};

namespace pilar
{
	//-- My Code
	struct Sphere 
	{
		Vector3f pos;
		float radius;
	};

	struct HairState
	{
		float* AA;
		float* bb;
		float* xx;

		Vector3f* root;
		Vector3f* normal;
		Vector3f* position;	
		Vector3f* save_pos;
		Vector3f* pos;		//previous position
		Vector3f* posc;		//candidate position
		Vector3f* posh;		//half position
		Vector3f* velocity;
		Vector3f* velh;		//half velocity
		Vector3f* force;

		Vector3f gravity;

		int numStrands;
		int numParticles;
		int numComponents;

		float mass;
		float k_edge;
		float k_bend;
		float k_twist;
		float k_extra;
		float d_edge;
		float d_bend;
		float d_twist;
		float d_extra;
		float length_e;
		float length_b;
		float length_t;
		
		Sphere* Head;
		//ModelOBJ* model;
		//float* vertices;
		//float* normals;
		//float* faces;
		
		curandStatePhilox4_32_10_t* rng;
	};
}

__device__ void buildAB(float dt, pilar::HairState* state);
__device__ void conjugate(pilar::HairState* state);
__device__ void calcVelocities(float dt, pilar::HairState* state);
__device__ void updateSprings(float dt, pilar::HairState* state);
__device__ void applyForce(pilar::Vector3f appliedForce, pilar::HairState* state);
__device__ void updateVelocities(float dt, pilar::HairState* state);
__device__ void updatePositions(float dt, pilar::HairState* state);
__device__ void updateParticles(float dt, pilar::HairState* state);
__device__ void applyStrainLimiting(float dt, pilar::HairState* state);

__global__ void initialise(pilar::HairState* state);
__global__ void update(float dt, pilar::HairState* state);

static void* mallocBytes(int bytes);
extern "C" void mallocStrands(pilar::HairState * h_state, pilar::HairState * &d_state);
extern "C" void freeStrands(pilar::HairState* h_state, pilar::HairState* d_state);
extern "C" void copyRoots(pilar::Vector3f* roots, pilar::Vector3f* normals, pilar::HairState* h_state);
extern "C" void copyState(pilar::HairState* h_state, pilar::HairState* d_state);
extern "C" void initialisePositions(pilar::HairState* h_state, pilar::HairState* d_state);
extern "C" void updateStrands(float dt, pilar::HairState* h_state, pilar::HairState* d_state);

#endif