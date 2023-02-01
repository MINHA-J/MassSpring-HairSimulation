// Fill out your copyright notice in the Description page of Project Settings.
#include "MyHairSim.h"
// UE4 Headers
#include "Engine/World.h"
#include "DrawDebugHelpers.h"
#include "UObject/ConstructorHelpers.h"

static bool check_tex_opaque(unsigned int tex);
MeshCustom::MeshCustom()
{
	vbo_vertices = 0;
	vbo_normals = 0;
	vbo_texcoords = 0;
	vbo_colors = 0;
	ibo = 0;

	num_vertices = 0;
	num_indices = 0;

	//mtl->tex = 0;
	//mtl->diffuse = Vector3f(1, 1, 1);
	//mtl->shininess = 50;
}

MeshCustom::~MeshCustom()
{
	delete bbox;
}

void MeshCustom::calc_bbox()
{
	if (vertices.empty()) {
		bbox->v0 = Vector3f(0, 0, 0);
		bbox->v1 = Vector3f(0, 0, 0);

		return;
	}

	bbox = new Aabb(Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), -Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));
	bbox->v1 = -bbox->v0;

	for (size_t i = 0; i < vertices.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (vertices[i][j] < bbox->v0[j])
				bbox->v0[j] = vertices[i][j];
			if (vertices[i][j] > bbox->v1[j])
				bbox->v1[j] = vertices[i][j];
		}
	}
}

namespace pilar
{
	void initDistanceField(ModelOBJ *obj, float* grid, HairState* state)
	{
		float result[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM];

		//Initialise distance field to inifinity
		for (int xx = 0; xx < DOMAIN_DIM; xx++)
			for (int yy = 0; yy < DOMAIN_DIM; yy++)
				for (int zz = 0; zz < DOMAIN_DIM; zz++)
					result[xx][yy][zz] = FLT_MAX;

		//calculate triangle normal scaling factor
		float delta = 0.25f;
		float echo = state->cell_width * delta;

		int numVertices = obj->totalConnectedPoints / POINTS_PER_VERTEX;
		int numTriangles = obj->totalConnectedTriangles / TOTAL_FLOATS_IN_TRIANGLE;

		//read in each triangle with its normal data
		for (int i = 0; i < numTriangles; i++)
		{
			float triangle[3][POINTS_PER_VERTEX];

			for (int j = 0; j < POINTS_PER_VERTEX; j++)
			{
				triangle[j][0] = obj->faces[i*TOTAL_FLOATS_IN_TRIANGLE + j * 3];
				triangle[j][1] = obj->faces[i*TOTAL_FLOATS_IN_TRIANGLE + j * 3 + 1];
				triangle[j][2] = obj->faces[i*TOTAL_FLOATS_IN_TRIANGLE + j * 3 + 2];
			}

			float normal[POINTS_PER_VERTEX];
			normal[0] = obj->normals[i*TOTAL_FLOATS_IN_TRIANGLE];
			normal[1] = obj->normals[i*TOTAL_FLOATS_IN_TRIANGLE + 1];
			normal[2] = obj->normals[i*TOTAL_FLOATS_IN_TRIANGLE + 2];

			//build prism
			float prism[6][POINTS_PER_VERTEX];
			for (int j = 0; j < POINTS_PER_VERTEX; j++)
			{
				prism[j][0] = triangle[j][0] + echo * normal[0];
				prism[j][1] = triangle[j][1] + echo * normal[1];
				prism[j][2] = triangle[j][2] + echo * normal[2];
				prism[j + 3][0] = triangle[j][0] - echo * normal[0];
				prism[j + 3][1] = triangle[j][1] - echo * normal[1];
				prism[j + 3][2] = triangle[j][2] - echo * normal[2];
			}

			//Axis-aligned bounding box
			float aabb[2][POINTS_PER_VERTEX]; //-x,-y,-z,+x,+y,+z
			aabb[0][0] = FLT_MAX;
			aabb[0][1] = FLT_MAX;
			aabb[0][2] = FLT_MAX;
			aabb[1][0] = -FLT_MAX;
			aabb[1][1] = -FLT_MAX;
			aabb[1][2] = -FLT_MAX;

			//Build the aabb using the minimum and maximum values of the prism
			for (int j = 0; j < 6; j++)
			{
				//minimum x, y, z
				aabb[0][0] = std::min(prism[j][0], aabb[0][0]);
				aabb[0][1] = std::min(prism[j][1], aabb[0][1]);
				aabb[0][2] = std::min(prism[j][2], aabb[0][2]);

				//maximum x, y, z
				aabb[1][0] = std::max(prism[j][0], aabb[1][0]);
				aabb[1][1] = std::max(prism[j][1], aabb[1][1]);
				aabb[1][2] = std::max(prism[j][2], aabb[1][2]);
			}

			//normalise to the grid
			aabb[0][0] = (aabb[0][0] + state->domain_half - state->cell_half) / state->cell_width;
			//aabb[0][1] = (aabb[0][1] + DOMAIN_HALF + (delta/2) - CELL_HALF) / CELL_WIDTH;
			aabb[0][1] = (aabb[0][1] + state->domain_half - state->cell_half) / state->cell_width;
			aabb[0][2] = (aabb[0][2] + state->domain_half - state->cell_half) / state->cell_width;
			aabb[1][0] = (aabb[1][0] + state->domain_half - state->cell_half) / state->cell_width;
			//aabb[1][1] = (aabb[1][1] + DOMAIN_HALF + (delta/2) - CELL_HALF) / CELL_WIDTH;
			aabb[1][1] = (aabb[1][1] + state->domain_half - state->cell_half) / state->cell_width;
			aabb[1][2] = (aabb[1][2] + state->domain_half - state->cell_half) / state->cell_width;

			//round aabb
			aabb[0][0] = floor(aabb[0][0]);
			aabb[0][1] = floor(aabb[0][1]);
			aabb[0][2] = floor(aabb[0][2]);
			aabb[1][0] = ceil(aabb[1][0]);
			aabb[1][1] = ceil(aabb[1][1]);
			aabb[1][2] = ceil(aabb[1][2]);

			int iaabb[2][POINTS_PER_VERTEX];
			iaabb[0][0] = int(aabb[0][0]);
			iaabb[0][1] = int(aabb[0][1]);
			iaabb[0][2] = int(aabb[0][2]);
			iaabb[1][0] = int(aabb[1][0]);
			iaabb[1][1] = int(aabb[1][1]);
			iaabb[1][2] = int(aabb[1][2]);

			//build edge vectors
			float edge[3][POINTS_PER_VERTEX];
			edge[0][0] = triangle[1][0] - triangle[0][0];
			edge[0][1] = triangle[1][1] - triangle[0][1];
			edge[0][2] = triangle[1][2] - triangle[0][2];
			edge[1][0] = triangle[2][0] - triangle[1][0];
			edge[1][1] = triangle[2][1] - triangle[1][1];
			edge[1][2] = triangle[2][2] - triangle[1][2];
			edge[2][0] = triangle[0][0] - triangle[2][0];
			edge[2][1] = triangle[0][1] - triangle[2][0];
			edge[2][2] = triangle[0][2] - triangle[2][0];

			//build edge normal vectors by cross product with triangle normal
			float edgeNormal[3][POINTS_PER_VERTEX];
			edgeNormal[0][0] = normal[1] * edge[0][2] - normal[2] * edge[0][1];
			edgeNormal[0][1] = normal[2] * edge[0][0] - normal[0] * edge[0][2];
			edgeNormal[0][2] = normal[0] * edge[0][1] - normal[1] * edge[0][0];
			edgeNormal[1][0] = normal[1] * edge[1][2] - normal[2] * edge[1][1];
			edgeNormal[1][1] = normal[2] * edge[1][0] - normal[0] * edge[1][2];
			edgeNormal[1][2] = normal[0] * edge[1][1] - normal[1] * edge[1][0];
			edgeNormal[2][0] = normal[1] * edge[2][2] - normal[2] * edge[2][1];
			edgeNormal[2][1] = normal[2] * edge[2][0] - normal[0] * edge[2][2];
			edgeNormal[2][2] = normal[0] * edge[2][1] - normal[1] * edge[2][0];

			for (int xx = iaabb[0][0]; xx <= iaabb[1][0]; xx++)
			{
				for (int yy = iaabb[0][1]; yy <= iaabb[1][1]; yy++)
				{
					for (int zz = iaabb[0][2]; zz <= iaabb[1][2]; zz++)
					{
						//Denormalise from grid to centre of cell
						float xpos = xx * state->cell_width - state->domain_half + state->cell_half;
						// float ypos = yy * CELL_WIDTH - DOMAIN_HALF - (delta / 2) + CELL_HALF;
						float ypos = yy * state->cell_width - state->domain_half + state->cell_half;
						float zpos = zz * state->cell_width - state->domain_half + state->cell_half;

						//dot product between gridpoint and triangle normal
						float dvalue = (xpos - triangle[0][0]) * normal[0] + (ypos - triangle[0][1]) * normal[1] + (zpos - triangle[0][2]) * normal[2];

						//Test whether the point lies within the triangle voronoi region
						float planeTest[3];
						planeTest[0] = xpos * edgeNormal[0][0] + ypos * edgeNormal[0][1] + zpos * edgeNormal[0][2] - triangle[0][0] * edgeNormal[0][0] - triangle[0][1] * edgeNormal[0][1] - triangle[0][2] * edgeNormal[0][2];
						planeTest[1] = xpos * edgeNormal[1][0] + ypos * edgeNormal[1][1] + zpos * edgeNormal[1][2] - triangle[1][0] * edgeNormal[1][0] - triangle[1][1] * edgeNormal[1][1] - triangle[1][2] * edgeNormal[1][2];
						planeTest[2] = xpos * edgeNormal[2][0] + ypos * edgeNormal[2][1] + zpos * edgeNormal[2][2] - triangle[2][0] * edgeNormal[2][0] - triangle[2][1] * edgeNormal[2][1] - triangle[2][2] * edgeNormal[2][2];

						if (!(planeTest[0] < 0.0f && planeTest[1] < 0.0f && planeTest[2] < 0.0f))
						{
							//Cross products
							float regionNormal[3][POINTS_PER_VERTEX];
							regionNormal[0][0] = normal[1] * edgeNormal[0][2] - normal[2] * edgeNormal[0][1];
							regionNormal[0][1] = normal[2] * edgeNormal[0][0] - normal[0] * edgeNormal[0][2];
							regionNormal[0][2] = normal[0] * edgeNormal[0][1] - normal[1] * edgeNormal[0][0];
							regionNormal[1][0] = normal[1] * edgeNormal[1][2] - normal[2] * edgeNormal[1][1];
							regionNormal[1][1] = normal[2] * edgeNormal[1][0] - normal[0] * edgeNormal[1][2];
							regionNormal[1][2] = normal[0] * edgeNormal[1][1] - normal[1] * edgeNormal[1][0];
							regionNormal[2][0] = normal[1] * edgeNormal[2][2] - normal[2] * edgeNormal[2][1];
							regionNormal[2][1] = normal[2] * edgeNormal[2][0] - normal[0] * edgeNormal[2][2];
							regionNormal[2][2] = normal[0] * edgeNormal[2][1] - normal[1] * edgeNormal[2][0];

							float regionTest[3][2];
							//Test if the point lies between the planes that define the first edge's voronoi region.
							regionTest[0][0] = -xpos * regionNormal[0][0] - ypos * regionNormal[0][1] - zpos * regionNormal[0][2] + triangle[0][0] * regionNormal[0][0] + triangle[0][1] * regionNormal[0][1] + triangle[0][2] * regionNormal[0][2];
							regionTest[0][1] = xpos * regionNormal[0][0] + ypos * regionNormal[0][1] + zpos * regionNormal[0][2] - triangle[1][0] * regionNormal[0][0] - triangle[1][1] * regionNormal[0][1] - triangle[1][2] * regionNormal[0][2];
							//Test if the point lies between the planes that define the second edge's voronoi region.
							regionTest[1][0] = -xpos * regionNormal[1][0] - ypos * regionNormal[1][1] - zpos * regionNormal[1][2] + triangle[1][0] * regionNormal[1][0] + triangle[1][1] * regionNormal[1][1] + triangle[1][2] * regionNormal[1][2];
							regionTest[1][1] = xpos * regionNormal[1][0] + ypos * regionNormal[1][1] + zpos * regionNormal[1][2] - triangle[2][0] * regionNormal[1][0] - triangle[2][1] * regionNormal[1][1] - triangle[2][2] * regionNormal[1][2];
							//Test if the point lies between the planes that define the third edge's voronoi region.
							regionTest[2][0] = -xpos * regionNormal[2][0] - ypos * regionNormal[2][1] - zpos * regionNormal[2][2] + triangle[2][0] * regionNormal[1][0] + triangle[2][1] * regionNormal[1][1] + triangle[2][2] * regionNormal[1][2];
							regionTest[2][1] = xpos * regionNormal[2][0] + ypos * regionNormal[2][1] + zpos * regionNormal[2][2] - triangle[0][0] * regionNormal[1][0] - triangle[0][1] * regionNormal[1][1] - triangle[0][2] * regionNormal[1][2];

							if (planeTest[0] >= 0.0f && regionTest[0][0] < 0.0f && regionTest[0][1] < 0.0f)
							{
								float aa[POINTS_PER_VERTEX];
								float bb[POINTS_PER_VERTEX];
								float cc[POINTS_PER_VERTEX];
								float dd[POINTS_PER_VERTEX];

								aa[0] = xpos - triangle[0][0];
								aa[1] = ypos - triangle[0][1];
								aa[2] = zpos - triangle[0][2];
								bb[0] = xpos - triangle[1][0];
								bb[1] = ypos - triangle[1][1];
								bb[2] = zpos - triangle[1][2];
								cc[0] = triangle[1][0] - triangle[0][0];
								cc[1] = triangle[1][1] - triangle[0][1];
								cc[2] = triangle[1][2] - triangle[0][2];

								dd[0] = aa[1] * bb[2] - aa[2] * bb[1];
								dd[1] = aa[2] * bb[0] - aa[0] * bb[2];
								dd[2] = aa[0] * bb[1] - aa[1] * bb[0];

								float dist = sqrtf(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]) / sqrtf(cc[0] * cc[0] + cc[1] * cc[1] + cc[2] * cc[2]);

								dvalue = (dvalue >= 0.0f) ? dist : -dist;

							}
							else if (planeTest[1] >= 0.0f && regionTest[1][0] < 0.0f && regionTest[1][1] < 0.0f)
							{
								float aa[POINTS_PER_VERTEX];
								float bb[POINTS_PER_VERTEX];
								float cc[POINTS_PER_VERTEX];
								float dd[POINTS_PER_VERTEX];

								aa[0] = xpos - triangle[1][0];
								aa[1] = ypos - triangle[1][1];
								aa[2] = zpos - triangle[1][2];
								bb[0] = xpos - triangle[2][0];
								bb[1] = ypos - triangle[2][1];
								bb[2] = zpos - triangle[2][2];
								cc[0] = triangle[2][0] - triangle[1][0];
								cc[1] = triangle[2][1] - triangle[1][1];
								cc[2] = triangle[2][2] - triangle[1][2];

								dd[0] = aa[1] * bb[2] - aa[2] * bb[1];
								dd[1] = aa[2] * bb[0] - aa[0] * bb[2];
								dd[2] = aa[0] * bb[1] - aa[1] * bb[0];

								float dist = sqrtf(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]) / sqrtf(cc[0] * cc[0] + cc[1] * cc[1] + cc[2] * cc[2]);

								dvalue = (dvalue >= 0.0f) ? dist : -dist;
							}
							else if (planeTest[2] >= 0.0f && regionTest[2][0] < 0.0f && regionTest[2][1] < 0.0f)
							{
								float aa[POINTS_PER_VERTEX];
								float bb[POINTS_PER_VERTEX];
								float cc[POINTS_PER_VERTEX];
								float dd[POINTS_PER_VERTEX];

								aa[0] = xpos - triangle[2][0];
								aa[1] = ypos - triangle[2][1];
								aa[2] = zpos - triangle[2][2];
								bb[0] = xpos - triangle[0][0];
								bb[1] = ypos - triangle[0][1];
								bb[2] = zpos - triangle[0][2];
								cc[0] = triangle[0][0] - triangle[2][0];
								cc[1] = triangle[0][1] - triangle[2][1];
								cc[2] = triangle[0][2] - triangle[2][2];

								dd[0] = aa[1] * bb[2] - aa[2] * bb[1];
								dd[1] = aa[2] * bb[0] - aa[0] * bb[2];
								dd[2] = aa[0] * bb[1] - aa[1] * bb[0];

								float dist = sqrtf(dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]) / sqrtf(cc[0] * cc[0] + cc[1] * cc[1] + cc[2] * cc[2]);

								dvalue = (dvalue >= 0.0f) ? dist : -dist;
							}
							else
							{
								float dist[3];
								dist[0] = sqrtf((xpos - triangle[0][0])*(xpos - triangle[0][0]) + (ypos - triangle[0][1])*(ypos - triangle[0][1]) + (zpos - triangle[0][2])*(zpos - triangle[0][2]));
								dist[1] = sqrtf((xpos - triangle[1][0])*(xpos - triangle[1][0]) + (ypos - triangle[1][1])*(ypos - triangle[1][1]) + (zpos - triangle[1][2])*(zpos - triangle[1][2]));
								dist[2] = sqrtf((xpos - triangle[2][0])*(xpos - triangle[2][0]) + (ypos - triangle[2][1])*(ypos - triangle[2][1]) + (zpos - triangle[2][2])*(zpos - triangle[2][2]));

								dvalue = (dvalue >= 0.0f) ? std::min(dist[0], std::min(dist[1], dist[2])) : -std::min(dist[0], std::min(dist[1], dist[2]));
							}
						}

						if (result[xx][yy][zz] < FLT_MAX)
						{
							if (std::abs(dvalue) < std::abs(result[xx][yy][zz]) && dvalue > 0.0f && result[xx][yy][zz] < 0.0f)
							{
								result[xx][yy][zz] = dvalue;
							}
							else if (std::abs(dvalue) < std::abs(result[xx][yy][zz]) && dvalue >= 0.0f && result[xx][yy][zz] > 0.0f)
							{
								result[xx][yy][zz] = dvalue;
							}
							else if (std::abs(dvalue) < std::abs(result[xx][yy][zz]) && dvalue <= 0.0f && result[xx][yy][zz] < 0.0f)
							{
								result[xx][yy][zz] = dvalue;
							}
						}
						else
						{
							result[xx][yy][zz] = dvalue;
						}
					}
				}
			}
		}

		// Save Result
		for (int i = 0; i < DOMAIN_DIM; i++)
			for (int j = 0; j < DOMAIN_DIM; j++)
				for (int k = 0; k < DOMAIN_DIM; k++)
				{
					int indexgrid = (DOMAIN_DIM*DOMAIN_DIM*i) + (DOMAIN_DIM * j) + k;
					if (indexgrid < DOMAIN_DIM*DOMAIN_DIM*DOMAIN_DIM)
					{
						// ?
						grid[indexgrid] = result[i][j][k];
						//if (result[i][j][k] < FLT_MAX)
						//	UE_LOG(LogTemp, Error, TEXT("ERR::Result %d(%d, %d, %d) %f\n"), indexgrid,i, j, k, grid[indexgrid]);
					}
				}
	}

	CUHair::CUHair(int numStrands,
		int numParticles,
		int numComponents,
		int numModel,
		float mass,
		float k_edge,
		float k_bend,
		float k_twist,
		float k_extra,
		float d_edge,
		float d_bend,
		float d_twist,
		float d_extra,
		float length_e,
		float length_b,
		float length_t,
		int domain_dim,
		float domain_width,
		float domain_half,
		float cell_width,
		float cell_half,
		Vector3f gravity,
		Vector3f* roots,
		Vector3f* normals,
		ModelOBJ* model,
		float* grid)
	{
		h_state = new HairState;
		h_state->root = roots;
		h_state->normal = normals;
		h_state->numStrands = numStrands;
		h_state->numParticles = numParticles;
		h_state->numComponents = numComponents;
		h_state->gravity = gravity;
		h_state->mass = mass;
		h_state->k_edge = k_edge;
		h_state->k_bend = k_bend;
		h_state->k_twist = k_twist;
		h_state->k_extra = k_extra;
		h_state->d_edge = d_edge;
		h_state->d_bend = d_bend;
		h_state->d_twist = d_twist;
		h_state->d_extra = d_extra;
		h_state->length_e = length_e;
		h_state->length_b = length_b;
		h_state->length_t = length_t;
		h_state->domain_dim = domain_dim;
		h_state->domain_width = domain_width;
		h_state->domain_half = domain_half;
		h_state->cell_width = cell_width;
		h_state->cell_half = cell_half;
		h_state->model = model;
		h_state->numModel = numModel;
		h_state->grid = grid;

		initDistanceField(model, grid, h_state);

		head = new Sphere;
		head->pos = Vector3f(0.0f, 50.f, 60.f);
		head->radius = 75.f;
		h_state->Head = head;
		d_state = 0;

		get_state = new pilar::HairState;
		get_state->root = roots;
		get_state->normal = normals;
		pilar::Vector3f* position = new pilar::Vector3f[numStrands*numParticles];
		get_state->position = position;
		pilar::Vector3f* pos = new pilar::Vector3f[numStrands*numParticles];
		get_state->pos = pos;
		pilar::Vector3f* velocity = new pilar::Vector3f[numStrands*numParticles];
		get_state->velocity = velocity;
		get_state->grid = grid;
		get_state->numStrands = numStrands;
		get_state->numParticles = numParticles;
		get_state->numComponents = numComponents;
		get_state->gravity = gravity;
		get_state->mass = mass;
		get_state->k_edge = k_edge;
		get_state->k_bend = k_bend;
		get_state->k_twist = k_twist;
		get_state->k_extra = k_extra;
		get_state->d_edge = d_edge;
		get_state->d_bend = d_bend;
		get_state->d_twist = d_twist;
		get_state->d_extra = d_extra;
		get_state->length_e = length_e;
		get_state->length_b = length_b;
		get_state->length_t = length_t;
		get_state->domain_dim = domain_dim;
		get_state->domain_width = domain_width;
		get_state->domain_half = domain_half;
		get_state->cell_width = cell_width;
		get_state->cell_half = cell_half;

		//Allocate memory on GPU          
		mallocStrands(h_state, d_state);

		//Copy root positions and normal directions to GPU
		copyRoots(roots, normals, grid, h_state);

		//Copy object model data to GPU
		//copyModel(model, h_state);
	}

	CUHair::~CUHair()
	{
		freeStrands(h_state, d_state);

		delete h_state;
		delete get_state;
	}

	void CUHair::initialise(Vector3f* position)
	{
		//h_state->position = position;
		checkCudaErrors(cudaMemcpy(h_state->position, position, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyHostToDevice));

		//Copy intialised state to the GPU
		copyState(h_state, d_state);

		//Intialise particle positions on the GPU
		initialisePositions(h_state, d_state);

		//checkCudaErrors(cudaMemcpy(get_state->root, d_state->root, d_state->numStrands * sizeof(*d_state->root), cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy(get_state->normal, d_state->normal, d_state->numStrands * sizeof(*d_state->normal), cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy(get_state->position, d_state->position, d_state->numParticles * d_state->numStrands * sizeof(*d_state->position), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(h_state, d_state, sizeof(d_state), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->root, h_state->root, h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->position, h_state->position, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->pos, h_state->pos, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->velocity, h_state->velocity, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->grid, h_state->grid, h_state->domain_dim * h_state->domain_dim * h_state->domain_dim * sizeof(float), cudaMemcpyDeviceToHost));
	}

	void CUHair::update(float dt, Vector3f* position)
	{
		//h_state->position = position;
		checkCudaErrors(cudaMemcpy(h_state->root, get_state->root, h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(h_state->pos, get_state->pos, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(h_state->position, position, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(h_state->velocity, get_state->velocity, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(h_state->grid, get_state->grid, h_state->domain_dim * h_state->domain_dim * h_state->domain_dim * sizeof(float), cudaMemcpyHostToDevice));

		updateStrands(dt, h_state, d_state);

		checkCudaErrors(cudaMemcpy(h_state, d_state, sizeof(d_state), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->root, h_state->root, h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->position, h_state->position, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->pos, h_state->pos, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->velocity, h_state->velocity, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->grid, h_state->grid, h_state->domain_dim * h_state->domain_dim * h_state->domain_dim * sizeof(float), cudaMemcpyDeviceToHost));
	}
}

UMyHairSim::UMyHairSim(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	PrimaryComponentTick.bCanEverTick = true;
	bTickInEditor = true;

	m_StaticMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MyStaticMesh"));
	m_StaticMesh->AttachTo(this);
	static ConstructorHelpers::FObjectFinder<UStaticMesh> FoundObject(TEXT("StaticMesh'/Game/Geometry/Meshes/TotalModel1.TotalModel1'"));
	if (FoundObject.Succeeded())
	{
		m_StaticMesh->SetStaticMesh(FoundObject.Object);
	}
	
	//if (FoundMaterial.Succeeded())
	//{
	//	DefaultMaterial = FoundMaterial.Object;
	//}
	//m_ProcedureMesh = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("MyProcedureMesh"));
	//m_ProcedureMesh->AttachTo(this);

	bShowStaticMesh = true;
	hair_length = 1.f;

	m_objects.clear();
	mesh_head = nullptr;
	m_model = nullptr;

	root_hair.clear();
	hairs = nullptr;

	//SplineMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SplineStaticMesh"));
	//DefaultMaterial = CreateDefaultSubobject<UMaterial>(TEXT("SplineMaterial"));
	SplineComponent = CreateDefaultSubobject<USplineComponent>(TEXT("SplinePath"));
	static ConstructorHelpers::FObjectFinder<UMaterial> FoundMaterial(TEXT("Material'/Game/StarterContent/Materials/M_Wood_Walnut.M_Wood_Walnut'"));
	if (FoundMaterial.Succeeded())
	{
		DefaultMaterial = FoundMaterial.Object;
	}
	//static ConstructorHelpers::FObjectFinder<UStaticMesh> FoundSpline(TEXT("StaticMesh'/Game/Geometry/Meshes/HairPiece1.HairPiece1'"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> FoundSpline(TEXT("StaticMesh'/Game/Geometry/Meshes/Braid_Plane_001.Braid_Plane_001'"));
	if (FoundSpline.Succeeded())
	{
		SplineMesh = FoundSpline.Object;
	}
}

void UMyHairSim::load_meshes()
{
	m_objects.clear();
	MeshCustom* mesh = new MeshCustom();

	UStaticMesh* usm = m_StaticMesh->GetStaticMesh();
	if (m_StaticMesh == nullptr) { UE_LOG(LogTemp, Error, TEXT("ERR::HeadMesh::No Static Mesh Set")); }

	// Store Static Mesh LOD0 Buffer Pointers
	FStaticMeshLODResources* lod0 = *(usm->RenderData->LODResources.GetData());
	smData.vb = &(lod0->VertexBuffers.PositionVertexBuffer); // Pos
	smData.smvb = &(lod0->VertexBuffers.StaticMeshVertexBuffer); // Static Mesh Buffer
	smData.cvb = &(lod0->VertexBuffers.ColorVertexBuffer); // Colour
	smData.ib = &(lod0->IndexBuffer); // Tri Inds

	smData.vert_count = smData.vb->GetNumVertices();
	smData.ind_count = smData.ib->GetNumIndices();
	smData.tri_count = smData.ind_count / 3;
	particleCount = smData.vert_count;
	//
	mesh->num_vertices = smData.vert_count;
	mesh->num_indices = smData.ind_count;

#ifdef DEBUG_PRINT_LOG
	UE_LOG(LogTemp, Warning, TEXT("DBG::Static Mesh Vertex Count == %d | Index Count = %d"), smData.vert_count, smData.ind_count);
#endif

	// Initalize smData Arrays. 
	smData.Pos.AddDefaulted(smData.vert_count);
	smData.Col.AddDefaulted(smData.vert_count);
	smData.Normal.AddDefaulted(smData.vert_count);
	smData.Tang.AddDefaulted(smData.vert_count);
	smData.UV.AddDefaulted(smData.vert_count);
	smData.Ind.AddDefaulted(smData.ind_count);
	smData.Tris.AddDefaulted(smData.tri_count);
	Particles.AddDefaulted(particleCount);

	// Need to add checks to delete previous procedual mesh data if exists.
	//ClearAllMeshSections();

	smData.has_uv = smData.smvb->GetNumTexCoords() != 0;
	smData.has_col = lod0->bHasColorVertexData;

	// Static mesh Data Buffer --> Array Deserialization.
	for (int32 i = 0; i < smData.vert_count; ++i)
	{
		// SMesh-ProcMesh Init
		smData.Pos[i] = smData.vb->VertexPosition(i); // Pass Verts Without Component Location Offset initally.
		smData.Normal[i] = smData.smvb->VertexTangentZ(i);
		smData.Tang[i] = FProcMeshTangent(FVector(smData.smvb->VertexTangentX(i).X, smData.smvb->VertexTangentX(i).Y, smData.smvb->VertexTangentX(i).Z), false);
		smData.has_col == true ? smData.Col[i] = smData.cvb->VertexColor(i) : smData.Col[i] = FColor(255, 255, 255);
		smData.has_uv == true ? smData.UV[i] = smData.smvb->GetVertexUV(i, 0) : smData.UV[i] = FVector2D(0.0f); // Only support 1 UV Channel fnow.

		// Particle Init
		FVector vertPtPos = GetComponentLocation() + smData.vb->VertexPosition(i); // Pts With Component Location Offset
		Particles[i].Position = vertPtPos;
		Particles[i].ID = i;
		lod0->bHasColorVertexData == true ? Particles[i].Color = smData.cvb->VertexColor(i) : Particles[i].Color = FColor(255, 255, 255);

		Model_vertices.Add(vertPtPos);

		// Mesh Init
		Vector3f MeshVertex = Vector3f(
			GetComponentLocation().X + smData.vb->VertexPosition(i).X,
			GetComponentLocation().Y + smData.vb->VertexPosition(i).Y,
			GetComponentLocation().Z + smData.vb->VertexPosition(i).Z);
		mesh->vertices.push_back(MeshVertex);

		Vector3f MeshNormal = Vector3f(
			smData.smvb->VertexTangentZ(i).X,
			smData.smvb->VertexTangentZ(i).Y,
			smData.smvb->VertexTangentZ(i).Z);
		mesh->normals.push_back(MeshNormal);

		if (smData.has_uv)
		{
			Vector2f MeshUV = Vector2f(smData.UV[i].X, smData.UV[i].Y);
			mesh->texcoords.push_back(MeshUV);
		}

		if (smData.has_col)
		{
			Vector3f MeshCol = Vector3f(smData.Col[i].R, smData.Col[i].G, smData.Col[i].B);
			mesh->colors.push_back(MeshCol);
		}
	}

	// Indices 
	for (int32 i = 0; i < smData.ind_count; ++i)
	{
		smData.Ind[i] = static_cast<int32>(smData.ib->GetIndex(i));
		mesh->indices.push_back(smData.Ind[i]);
	}

	//Build Mesh Section
	//m_ProcedureMesh->CreateMeshSection(0, smData.Pos, smData.Ind, smData.Normal, smData.UV, smData.Col, smData.Tang, false);
	//bShowStaticMesh = false;
	//m_StaticMesh->SetVisibility(bShowStaticMesh);
	//StateExists = true;

	// Build Tri Array and Per Vert Shared Tri Array
	// BuildTriArrays();

	m_objects.push_back(mesh);
}

#define K_ANC 4.0
#define DAMPING 1.5
Vector3f UMyHairSim::calc_rand_point(const Triangle& tr, Vector3f* bary)
{
	float u = (float)rand() / (float)RAND_MAX;
	float v = (float)rand() / (float)RAND_MAX;

	if (u + v > 1) {
		u = 1 - u;
		v = 1 - v;
	}

	float c = 1 - (u + v);

	Vector3f rp = u * tr.v[0] + v * tr.v[1] + c * tr.v[2];

	bary->x() = u;
	bary->y() = v;
	bary->z() = c;

	return rp;
}

void UMyHairSim::get_spawn_triangles(const MeshCustom* m, float thresh, std::vector<Triangle>* faces)
{
	if (m == NULL) { UE_LOG(LogTemp, Error, TEXT("Invalid mesh.")); return; };

	float min_y = FLT_MAX;
	float max_y = -FLT_MAX;
	for (size_t i = 0; i < m->indices.size() / 3; i++)
	{
		bool is_spawn = true;
		int idx[3];
		for (int j = 0; j < 3; j++)
		{
			idx[j] = m->indices[i * 3 + j];
			float c = (m->colors[idx[j]].x() + m->colors[idx[j]].y() + m->colors[idx[j]].z()) / 3;
			if (c >= thresh)
			{
				is_spawn = false;
				break;
			}
		}

		if (is_spawn)
		{
			Triangle t;
			for (int j = 0; j < 3; j++)
			{
				t.v[j] = m->vertices[idx[j]];
				t.n[j] = m->normals[idx[j]];
				if (t.v[j].y() < min_y)
					min_y = t.v[j].y();
				if (t.v[j].y() > max_y)
					max_y = t.v[j].y();
			}
			faces->push_back(t);
		}
	}
	UE_LOG(LogTemp, Warning, TEXT("DBG::get_spawn_triangles - AABB : min y : % f max y : % f\n"), min_y, max_y);
}

float length_sq(const Vector3f& v)
{
	return v.x() * v.x() + v.y() * v.y() + v.z() * v.z();
}
bool UMyHairSim::init_HairRoot(const MeshCustom* m, int num_spawns, float thresh)
{
	std::vector<Triangle> faces;
	kdtree* kd = kd_create(3);
	const float min_dist = 0.05;

	if (m == NULL) { UE_LOG(LogTemp, Error, TEXT("Invalid mesh.")); return false; };

	if (root_hair.size() < num_spawns)
	{
		get_spawn_triangles(m, thresh, &faces);

		for (int i = 0; i < num_spawns; i++)
		{
			// Poisson
			int rnum = (float)((float)rand() / (float)RAND_MAX) * faces.size();
			Triangle rtriangle = faces[rnum];
			Vector3f bary;
			Vector3f rpoint = calc_rand_point(rtriangle, &bary);

			kdres* res = kd_nearest3f(kd, rpoint.x(), rpoint.y(), rpoint.z());

			if (res && !kd_res_end(res))
			{
				Vector3f nearest;
				kd_res_item3f(res, &nearest.x(), &nearest.y(), &nearest.z());
				kd_res_free(res);

				float distance_sq = length_sq(rpoint - nearest);
				if (distance_sq < min_dist * min_dist)
					continue;
			}

			HairStrand strand;
			/* weighted sum of the triangle's vertex normals */
			strand.spawn_dir = (rtriangle.n[0] * bary.x() + rtriangle.n[1] * bary.y() + rtriangle.n[2] * bary.z()).normalized();
			strand.spawn_pt = rpoint;
			root_hair.push_back(strand);

			kd_insert3f(kd, rpoint.x(), rpoint.y(), rpoint.z(), 0);
		}

		kd_free(kd);

		for (size_t i = 0; i < root_hair.size(); i++)
		{
			root_hair[i].pos = root_hair[i].spawn_pt + root_hair[i].spawn_dir * hair_length;
		}

		UWorld* world = GetWorld();
		//DrawDebugSphere(world, FVector(0, 50, 60), 75, 26, FColor::Blue, true, -1, 0, 1);
		//UE_LOG(LogType, Warning, TEXT("DBG::init_HairRoot - Show Hair: %d"), root_hair.size());
		//for (int32 i = 0; i < root_hair.size() - 1; ++i)
		//{
		//	FVector vec(root_hair[i].spawn_pt.x(), root_hair[i].spawn_pt.y(), root_hair[i].spawn_pt.z());
		//	DrawDebugPoint(world, vec, 9.f, FColor(255, 0, 255), true, 5.f);
		//}
	}
	return true;
}

void UMyHairSim::loadModel(ModelOBJ* obj)
{
	UStaticMesh* usm = m_StaticMesh->GetStaticMesh();
	if (m_StaticMesh == nullptr) { UE_LOG(LogTemp, Error, TEXT("ERR::HeadMesh::No Static Mesh Set")); }

	// Store Static Mesh LOD0 Buffer Pointers
	FStaticMeshLODResources* lod0 = *(usm->RenderData->LODResources.GetData());
	smData.vb = &(lod0->VertexBuffers.PositionVertexBuffer); // Pos
	smData.smvb = &(lod0->VertexBuffers.StaticMeshVertexBuffer); // Static Mesh Buffer
	smData.cvb = &(lod0->VertexBuffers.ColorVertexBuffer); // Colour
	smData.ib = &(lod0->IndexBuffer); // Tri Inds
	smData.vert_count = smData.vb->GetNumVertices();
	smData.ind_count = smData.ib->GetNumIndices();
	smData.tri_count = smData.ind_count / 3;
	particleCount = smData.vert_count;

	obj->totalConnectedPoints = 0;
	obj->totalConnectedTriangles = 0;
	obj->vertices = new float[smData.vert_count * 3];
	obj->normals = new float[smData.ind_count * 3];
	obj->faces = new float[smData.ind_count * 3];

	int triangleIndex = 0;
	int normalIndex = 0;

	UWorld* world = GetWorld();

	// v: x, y, z의 형식의 데이터
	for (int32 i = 0; i < smData.vert_count; ++i)
	{
		// ? 좌표계 문제인가 싶어 y, z 바꿔봄
		obj->vertices[obj->totalConnectedPoints] = smData.vb->VertexPosition(i).X;
		obj->vertices[obj->totalConnectedPoints + 1] = smData.vb->VertexPosition(i).Z;
		obj->vertices[obj->totalConnectedPoints + 2] = smData.vb->VertexPosition(i).Y;

		//FVector v(obj->vertices[obj->totalConnectedPoints], obj->vertices[obj->totalConnectedPoints + 2], obj->vertices[obj->totalConnectedPoints + 1]);
		//DrawDebugPoint(world, v, 2, FColor::Blue, true);
		obj->totalConnectedPoints += POINTS_PER_VERTEX;
	}

	int num = 0;
	for (int i = 0; i < smData.tri_count; ++i)
	{
		int vertexNumber[4] = { 0, 0, 0 };
		vertexNumber[0] = smData.ib->GetIndex(3*i);
		vertexNumber[1] = smData.ib->GetIndex(3*i + 1);
		vertexNumber[2] = smData.ib->GetIndex(3*i + 2);

		int tCounter = 0;
		for (int j = 0; j < POINTS_PER_VERTEX; j++)
		{
			obj->faces[triangleIndex + tCounter] = obj->vertices[3 * vertexNumber[j]];
			obj->faces[triangleIndex + tCounter + 1] = obj->vertices[3 * vertexNumber[j] + 1];
			obj->faces[triangleIndex + tCounter + 2] = obj->vertices[3 * vertexNumber[j] + 2];
			tCounter += POINTS_PER_VERTEX;
		}

		float coord1[3] = { obj->faces[triangleIndex], obj->faces[triangleIndex + 1], obj->faces[triangleIndex + 2] };
		float coord2[3] = { obj->faces[triangleIndex + 3], obj->faces[triangleIndex + 4], obj->faces[triangleIndex + 5] };
		float coord3[3] = { obj->faces[triangleIndex + 6], obj->faces[triangleIndex + 7], obj->faces[triangleIndex + 8] };

		// Debug
		//FVector vec1(obj->faces[triangleIndex], obj->faces[triangleIndex + 1], obj->faces[triangleIndex + 2]);
		//FVector vec2(obj->faces[triangleIndex + 3], obj->faces[triangleIndex + 4], obj->faces[triangleIndex + 5]);
		//FVector vec3(obj->faces[triangleIndex + 6], obj->faces[triangleIndex + 7], obj->faces[triangleIndex + 8]);
		//DrawDebugLine(world, vec1, vec2, FColor::Blue, true, -1, 0, 0.5);
		//DrawDebugLine(world, vec2, vec3, FColor::Blue, true, -1, 0, 0.5);
		//DrawDebugLine(world, vec3, vec1, FColor::Blue, true, -1, 0, 0.5);

		/* calculate Vector1 and Vector2 */
		float va[3], vb[3], vr[3], val;

		va[0] = coord1[0] - coord2[0];
		va[1] = coord1[1] - coord2[1];
		va[2] = coord1[2] - coord2[2];

		vb[0] = coord1[0] - coord3[0];
		vb[1] = coord1[1] - coord3[1];
		vb[2] = coord1[2] - coord3[2];


		/* cross product */
		vr[0] = va[1] * vb[2] - vb[1] * va[2];
		vr[1] = vb[0] * va[2] - va[0] * vb[2];
		vr[2] = va[0] * vb[1] - vb[0] * va[1];

		/* normalization factor */
		val = sqrtf(vr[0] * vr[0] + vr[1] * vr[1] + vr[2] * vr[2]);

		float norm[3];
		norm[0] = vr[0] / val;
		norm[1] = vr[1] / val;
		norm[2] = vr[2] / val;

		tCounter = 0;
		for (int j = 0; j < POINTS_PER_VERTEX; j++)
		{
			obj->normals[normalIndex + tCounter] = norm[0];
			obj->normals[normalIndex + tCounter + 1] = norm[1];
			obj->normals[normalIndex + tCounter + 2] = norm[2];
			tCounter += POINTS_PER_VERTEX;
		}

		triangleIndex += TOTAL_FLOATS_IN_TRIANGLE;
		normalIndex += TOTAL_FLOATS_IN_TRIANGLE;
		obj->totalConnectedTriangles += TOTAL_FLOATS_IN_TRIANGLE;

		//UE_LOG(LogTemp, Warning, TEXT("DBG::vertex %d -> x: %d, y: %d, z: %d\n"), num, vertexNumber[0], vertexNumber[1], vertexNumber[2]);
		num++;
	}

	// Draw
	//for (int i = 0; i < triangleIndex; i+= TOTAL_FLOATS_IN_TRIANGLE)
	//{
	//	FVector vec1(obj->faces[i], obj->faces[i+1], obj->faces[i+2]);
	//	FVector vec2(obj->faces[i+3], obj->faces[i + 4], obj->faces[i + 5]);
	//	FVector vec3(obj->faces[i+6], obj->faces[i + 7], obj->faces[i + 8]);
	//	DrawDebugLine(world, vec1, vec2, FColor::Blue, true, -1, 0, 0.5);
	//	DrawDebugLine(world, vec2, vec3, FColor::Blue, true, -1, 0, 0.5);
	//	DrawDebugLine(world, vec3, vec1, FColor::Blue, true, -1, 0, 0.5);
	//}

	//UE_LOG(LogTemp, Warning, TEXT("DBG::index triangle: %d, normal: %d\n"), triangleIndex, normalIndex);
}

void UMyHairSim::UpdateModel(ModelOBJ* obj, pilar::Vector3f mov)
{
	int triangleIndex = 0;
	int normalIndex = 0;
	UWorld* world = GetWorld();
	FVector distance = FVector(mov.x, mov.y, mov.z);

	// v: x, y, z의 형식의 데이터
	for (int32 i = 0; i < smData.vert_count; ++i)
	{
		Model_vertices[i] += distance;

		// ? 좌표계 문제인가 싶어 y, z 바꿔봄
		//obj->vertices[obj->totalConnectedPoints] = m_ProcedureMesh->GetProcMeshSection(0)->ProcVertexBuffer[i].Position.X;
		//obj->vertices[obj->totalConnectedPoints + 1] = m_ProcedureMesh->GetProcMeshSection(0)->ProcVertexBuffer[i].Position.Z;
		//obj->vertices[obj->totalConnectedPoints + 2] = m_ProcedureMesh->GetProcMeshSection(0)->ProcVertexBuffer[i].Position.Y;

		//-- For Debug Model
		//FVector v(m_ProcedureMesh->GetProcMeshSection(0)->ProcVertexBuffer[i].Position.X, 
		//	m_ProcedureMesh->GetProcMeshSection(0)->ProcVertexBuffer[i].Position.Y,
		//	m_ProcedureMesh->GetProcMeshSection(0)->ProcVertexBuffer[i].Position.Z);
		//DrawDebugPoint(world, v, 2, FColor(52, 220, 239), false, 0.01f);
	}

	for (int i = 0; i < smData.tri_count; ++i)
	{
		int vertexNumber[4] = { 0, 0, 0 };
		vertexNumber[0] = smData.ib->GetIndex(3 * i);
		vertexNumber[1] = smData.ib->GetIndex(3 * i + 1);
		vertexNumber[2] = smData.ib->GetIndex(3 * i + 2);

		int tCounter = 0;
		for (int j = 0; j < POINTS_PER_VERTEX; j++)
		{
			//DrawDebugPoint(world, Model_vertices[vertexNumber[j]], 2, FColor(52, 220, 239), false, 0.01f);
			obj->faces[triangleIndex + tCounter] = Model_vertices[vertexNumber[j]].X;
			obj->faces[triangleIndex + tCounter + 1] = Model_vertices[vertexNumber[j]].Z;
			obj->faces[triangleIndex + tCounter + 2] = Model_vertices[vertexNumber[j]].Y;
			tCounter += POINTS_PER_VERTEX;
		}

		float coord1[3] = { obj->faces[triangleIndex], obj->faces[triangleIndex + 1], obj->faces[triangleIndex + 2] };
		float coord2[3] = { obj->faces[triangleIndex + 3], obj->faces[triangleIndex + 4], obj->faces[triangleIndex + 5] };
		float coord3[3] = { obj->faces[triangleIndex + 6], obj->faces[triangleIndex + 7], obj->faces[triangleIndex + 8] };

		/* calculate Vector1 and Vector2 */
		float va[3], vb[3], vr[3], val;

		va[0] = coord1[0] - coord2[0];
		va[1] = coord1[1] - coord2[1];
		va[2] = coord1[2] - coord2[2];

		vb[0] = coord1[0] - coord3[0];
		vb[1] = coord1[1] - coord3[1];
		vb[2] = coord1[2] - coord3[2];


		/* cross product */
		vr[0] = va[1] * vb[2] - vb[1] * va[2];
		vr[1] = vb[0] * va[2] - va[0] * vb[2];
		vr[2] = va[0] * vb[1] - vb[0] * va[1];

		/* normalization factor */
		val = sqrtf(vr[0] * vr[0] + vr[1] * vr[1] + vr[2] * vr[2]);

		float norm[3];
		norm[0] = vr[0] / val;
		norm[1] = vr[1] / val;
		norm[2] = vr[2] / val;

		tCounter = 0;
		for (int j = 0; j < POINTS_PER_VERTEX; j++)
		{
			obj->normals[normalIndex + tCounter] = norm[0];
			obj->normals[normalIndex + tCounter + 1] = norm[1];
			obj->normals[normalIndex + tCounter + 2] = norm[2];
			tCounter += POINTS_PER_VERTEX;
		}

		triangleIndex += TOTAL_FLOATS_IN_TRIANGLE;
		normalIndex += TOTAL_FLOATS_IN_TRIANGLE;
	}
}

void UMyHairSim::InitHairModel()
{
	load_meshes();

	if (m_objects.empty())
	{
		UE_LOG(LogTemp, Error, TEXT("ERR::HeadMesh::No Static Mesh Set")); return;
	}

	for (size_t i = 0; i < m_objects.size(); i++)
	{
		m_objects[i]->calc_bbox();
		mesh_head = m_objects[i];
	}

	//	coll_sphere.radius = 1.0;
	//	coll_sphere.center = Vec3(0, 0.6, 0.53);
	//int numStrands = NUMSTRANDS;
	int numStrands = 50;

	if (!init_HairRoot(mesh_head, numStrands, THRESH))
	{
		UE_LOG(LogTemp, Error, TEXT("ERR::HairMesh::Failed to initialize hair"));
		return;
	}

	// Root positions, Normal directions, Gravity
	pilar::Vector3f* roots = new pilar::Vector3f[numStrands];
	pilar::Vector3f* normals = new pilar::Vector3f[numStrands];
	for (int32 i = 0; i < root_hair.size(); ++i)
	{
		roots[i] = pilar::Vector3f(
			root_hair[i].spawn_pt.x(),
			root_hair[i].spawn_pt.z(),
			root_hair[i].spawn_pt.y());

		normals[i] = pilar::Vector3f((root_hair[i].spawn_pt.x()), (root_hair[i].spawn_pt.z()), 0);
		//normals[i] = pilar::Vector3f(m_normal.X, m_normal.Z, m_normal.Y);
	}

	//Gravity
	pilar::Vector3f gravity(0.0f, GRAVITY, 0.0f);

	//Load geometry from file
	m_model = new ModelOBJ;
	loadModel(m_model);
	float* grid = new float[DOMAIN_DIM*DOMAIN_DIM*DOMAIN_DIM];

	// Initialise get_state 
	hairs = new pilar::CUHair(numStrands, NUMPARTICLES, NUMCOMPONENTS, 1, MASS,
		K_EDGE, K_BEND, K_TWIST, K_EXTRA,
		D_EDGE, D_BEND, D_TWIST, D_EXTRA,
		LENGTH_EDGE, LENGTH_BEND, LENGTH_TWIST,
		DOMAIN_DIM, DOMAIN_WIDTH, DOMAIN_HALF, CELL_WIDTH, CELL_HALF,
		gravity,
		roots,
		normals,
		m_model,
		grid);

	//Initialise positions along normals on the gpu
	hairs->initialise(hairs->get_state->position);

	UE_LOG(LogTemp, Warning, TEXT("DBG::InitHairModel - finish"));
	
}

void UMyHairSim::DoOnceSimulation()
{
	InitHairModel();
	GEngine->AddOnScreenDebugMessage(-1, 3.0f, FColor::Blue, TEXT("Init Hair Model"));

	pilar::Vector3f* pos = hairs->get_state->position;

	for (int32 h = 0; h < hairs->h_state->numStrands; ++h)
	{
		for (int32 par = 0; par < hairs->h_state->numParticles; ++par)
		{
			UE_LOG(LogType, Warning, TEXT("DoOnceSimulation - Update Hair %d"), h * hairs->h_state->numParticles + par);
			UWorld* world = GetWorld();

			FVector vec1(
				pos[h * hairs->h_state->numParticles + par].x,
				pos[h * hairs->h_state->numParticles + par].z,
				pos[h * hairs->h_state->numParticles + par].y);
			//DrawDebugPoint(world, vec1, 5.f, FColor(255, 255, 0), true);

			if (par + 1 < hairs->h_state->numParticles)
			{
				FVector vec2(
					pos[h * hairs->h_state->numParticles + par + 1].x,
					pos[h * hairs->h_state->numParticles + par + 1].z,
					pos[h * hairs->h_state->numParticles + par + 1].y);

				//DrawDebugLine(world, vec1, vec2, FColor::Emerald, true, -1, 0, 2.f);
			}
		}
	}
	UpdateHairSpline();
	UE_LOG(LogType, Warning, TEXT("DoOnceSimulation - Finish"));
}

void UMyHairSim::UpdateHairSpline()
{
	/****************************************
	이전의 Spline 위치와 Mesh를 없애 할당을 하나로 함
	*****************************************/
	if (SplineHairs.Num() > 0)
	{
		for (int32 i = 0; i< SplineHairs.Num(); i++)
		{
			SplineHairs[i]->ClearSplinePoints(true);
		}
	}

	if (SplineHairMeshes.Num() > 0)
	{
		for (int32 i = 0; i < SplineHairMeshes.Num(); i++)
		{ 
			if (SplineHairMeshes[i])
				SplineHairMeshes[i]->DestroyComponent();
		}
	}
	SplineHairs.Empty();
	SplineHairMeshes.Empty();

	for (int32 h = 0; h < hairs->h_state->numStrands; h++)
	{
		//USplineComponent* SplineComponent = NewObject<USplineComponent>(this, USplineComponent::StaticClass());
		FVector root(hairs->get_state->root[h].x, hairs->get_state->root[h].z, hairs->get_state->root[h].y);
		FVector first(
			hairs->get_state->position[h * hairs->h_state->numParticles].x,
			hairs->get_state->position[h * hairs->h_state->numParticles].z,
			hairs->get_state->position[h * hairs->h_state->numParticles].y);
		SplineComponent->AddSplinePointAtIndex(root, 0, ESplineCoordinateSpace::World);
		//SplineComponent->AddSplinePointAtIndex(first, 1, ESplineCoordinateSpace::World);

		for (int32 SplineCount = 0; SplineCount < hairs->h_state->numParticles; SplineCount++)
		{
			FVector vector(
				hairs->get_state->position[h * hairs->h_state->numParticles + SplineCount].x,
				hairs->get_state->position[h * hairs->h_state->numParticles + SplineCount].z,
				hairs->get_state->position[h * hairs->h_state->numParticles + SplineCount].y);
			SplineComponent->AddSplinePointAtIndex(vector, SplineCount + 1, ESplineCoordinateSpace::World);
			FVector Tangent = SplineComponent->GetLocationAtSplinePoint(SplineCount+1, ESplineCoordinateSpace::Local) - SplineComponent->GetLocationAtSplinePoint(SplineCount, ESplineCoordinateSpace::Local);
			
			USplineMeshComponent* HairSplineComponent = NewObject<USplineMeshComponent>(this, USplineMeshComponent::StaticClass());
			HairSplineComponent->SetForwardAxis(ESplineMeshAxis::Z);
			if (DefaultMaterial)
			{
				HairSplineComponent->SetMaterial(0, DefaultMaterial);
			}
			HairSplineComponent->SetStaticMesh(SplineMesh);			
			// static move
			HairSplineComponent->SetMobility(EComponentMobility::Movable);
			HairSplineComponent->CreationMethod = EComponentCreationMethod::UserConstructionScript;
			// Register
			HairSplineComponent->RegisterComponentWithWorld(GetWorld());
			// Spline Component에 따라 크기 위치를 변경함
			HairSplineComponent->AttachToComponent(SplineComponent, FAttachmentTransformRules::KeepRelativeTransform);
			//HairSplineComponent->SetRelativeScale3D(FVector(1.0f, 1.0f, 1.0f));
			//HairSplineComponent->SetStartScale(FVector2D(0.05f, 0.05f));
			//HairSplineComponent->SetEndScale(FVector2D(0.05f, 0.05f));

			// 시작 지점
			const FVector StartPoint = SplineComponent->GetLocationAtSplinePoint(SplineCount, ESplineCoordinateSpace::Local);
			const FVector StartTangent = Tangent;
			//const FVector StartTangent = SplineComponent->GetTangentAtDistanceAlongSpline(SplineComponent->GetSplineLength(), ESplineCoordinateSpace::Local);
			const FVector EndPoint = SplineComponent->GetLocationAtSplinePoint(SplineCount + 1, ESplineCoordinateSpace::Local);
			const FVector EndTangent = -1 * Tangent;
			//const FVector EndTangent = SplineComponent->GetTangentAtDistanceAlongSpline(SplineComponent->GetSplineLength(), ESplineCoordinateSpace::Local);
			//const FVector EndTangent = SplineComponent->GetTangentAtSplinePoint(SplineCount + 1, ESplineCoordinateSpace::Local);
			HairSplineComponent->SetStartAndEnd(StartPoint, StartTangent, EndPoint, EndTangent, true);
			//HairSplineComponent->SetCollisionEnabled(ECollisionEnabled::PhysicsOnly);
			
			SplineHairMeshes.Add(HairSplineComponent);
		}
		SplineHairs.Add(SplineComponent);
	}
}

void UMyHairSim::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	if (bStartSimulate)
	{
		m_before = pilar::Vector3f(GetComponentToWorld().GetLocation().X, GetComponentToWorld().GetLocation().Y, GetComponentToWorld().GetLocation().Z);
		UWorld* world = GetWorld();
		if (bIsInitMesh)
		{
			GEngine->AddOnScreenDebugMessage(-1, 3.0f, FColor::Blue, TEXT("Init Hair Model"));
			InitHairModel();
			DrawDebugBox(world, FVector(0,0,0), FVector(CELL_WIDTH * DOMAIN_DIM, CELL_WIDTH * DOMAIN_DIM, CELL_WIDTH * DOMAIN_DIM), FColor::Purple, true, -1, 0, 5);
			bIsInitMesh = false;
		}

		//---Move distance
		m_move = m_before - m_after;

		//---If Move, Update Model Grid
		if (m_move.length() > 0)
		{
			UpdateModel(m_model, m_move);
			pilar::initDistanceField(m_model, hairs->get_state->grid, hairs->get_state);
		}

		//---Update Hair
		pilar::Vector3f* position = hairs->get_state->position;
		hairs->update(abs(DeltaTime) / 10.0f, position);
		//hairs->collide(abs(DeltaTime) / 10.0f);

		for (int32 h = 0; h < hairs->h_state->numStrands; h++)
		{
			hairs->get_state->root[h].x += m_move.x;
			hairs->get_state->root[h].y += m_move.z;
			hairs->get_state->root[h].z += m_move.y;

			FVector root(hairs->get_state->root[h].x, hairs->get_state->root[h].z, hairs->get_state->root[h].y);
			FVector first(
				hairs->get_state->position[h * hairs->h_state->numParticles].x + m_move.x,
				hairs->get_state->position[h * hairs->h_state->numParticles].z + m_move.y,
				hairs->get_state->position[h * hairs->h_state->numParticles].y + m_move.z);
			//DrawDebugLine(world, root, first, FColor::Emerald, false, -1, 0, 0.8f);

			for (int32 par = 0; par < hairs->h_state->numParticles; par++)
			{
				hairs->get_state->position[h * hairs->h_state->numParticles + par].x += m_move.x;
				hairs->get_state->position[h * hairs->h_state->numParticles + par].y += m_move.z;
				hairs->get_state->position[h * hairs->h_state->numParticles + par].z += m_move.y;

				//--- For Debug Particle position & velocity
				//UE_LOG(LogType, Log, TEXT("Position %d Hair, %d Particle - x:%f, y:%f, z:%f"), 
				//	h, par,
				//	hairs->get_state->position[h * hairs->h_state->numParticles + par].x,
				//	hairs->get_state->position[h * hairs->h_state->numParticles + par].z,
				//	hairs->get_state->position[h * hairs->h_state->numParticles + par].y);
				//UE_LOG(LogType, Log, TEXT("Velocity %d Hair, %d Particle - x:%f, y:%f, z:%f"),
				//	h, par,
				//	hairs->get_state->velocity[h * hairs->h_state->numParticles + par].x,
				//	hairs->get_state->velocity[h * hairs->h_state->numParticles + par].z,
				//	hairs->get_state->velocity[h * hairs->h_state->numParticles + par].y);

				//--- For visualize
				//FVector vec1(
				//	hairs->get_state->position[h * hairs->h_state->numParticles + par].x,
				//	hairs->get_state->position[h * hairs->h_state->numParticles + par].z,
				//	hairs->get_state->position[h * hairs->h_state->numParticles + par].y);
				//DrawDebugPoint(world, vec1, 4.f, FColor::Green, false, 0.1f);

				if (par + 1 < hairs->h_state->numParticles)
				{
					FVector vec2(
						hairs->get_state->position[h * hairs->h_state->numParticles + par + 1].x + m_move.x,
						hairs->get_state->position[h * hairs->h_state->numParticles + par + 1].z + m_move.y,
						hairs->get_state->position[h * hairs->h_state->numParticles + par + 1].y + m_move.z);

					//--- For visualize
					//DrawDebugLine(world, vec1, vec2, FColor::Red, false, -1, 0, 0.8f);
				}
			}
			//UE_LOG(LogType, Error, TEXT("Update %f time, Hair %d"), abs(DeltaTime) / 10.0f, h);
			m_after = pilar::Vector3f(GetComponentToWorld().GetLocation().X, GetComponentToWorld().GetLocation().Y, GetComponentToWorld().GetLocation().Z);
		}
		UpdateHairSpline();
	}
}

