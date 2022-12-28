// Fill out your copyright notice in the Description page of Project Settings.
#include "MyHairSim.h"
// UE4 Headers
#include "Engine/World.h"
#include "DrawDebugHelpers.h"

static bool check_tex_opaque(unsigned int tex);
Mesh::Mesh()
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
// Mesh's Info
//vector<Mesh*> load_meshes(const char* fname)
//{
	// BuildProceduralMeshComponent 쪽에 구현해놓음
//}

void Mesh::calc_bbox()
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
	CUHair::CUHair(int numStrands,
		int numParticles,
		int numComponents,
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
		Vector3f gravity,
		Vector3f* roots,
		Vector3f* normals
		//ModelOBJ* model
	)
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

		head = new Sphere;
		head->pos = Vector3f(0.0f, 50.f, 60.f);
		head->radius = 75.f;
		h_state->Head = head;

		d_state = 0;

		//Get_state = h_state;
		//Allocate memory on GPU          
		mallocStrands(h_state, d_state);
		//checkCudaErrors(cudaMalloc((void**)&d_state, sizeof(pilar::HairState)));

		//Copy root positions and normal directions to GPU
		copyRoots(roots, normals, h_state);

		get_state = new HairState;
		get_state->root = roots;
		get_state->normal = normals;

		//Copy object model data to GPU
		//copyModel(model, h_state);
	}

	CUHair::~CUHair()
	{
		freeStrands(h_state, d_state);

		delete h_state;
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

	}

	void CUHair::update(float dt, Vector3f* position)
	{
		//h_state->position = position;
		checkCudaErrors(cudaMemcpy(h_state->position, position, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyHostToDevice));

		updateStrands(dt, h_state, d_state);

		checkCudaErrors(cudaMemcpy(h_state, d_state, sizeof(d_state), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->root, h_state->root, h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->position, h_state->position, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->pos, h_state->pos, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(get_state->velocity, h_state->velocity, h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f), cudaMemcpyDeviceToHost));

	}
}

UMyHairSim::UMyHairSim(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	PrimaryComponentTick.bCanEverTick = true;
	bTickInEditor = true;

	m_StaticMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MyStaticMesh"));
	smData.vb = nullptr;
	smData.cvb = nullptr;
	smData.smvb = nullptr;
	smData.ib = nullptr;

	bShowStaticMesh = true;
	hair_length = 1.f;
}

void UMyHairSim::load_meshes()
{
	m_meshes.clear();
	Mesh* mesh = new Mesh();

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
	ClearAllMeshSections();

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
	CreateMeshSection(0, smData.Pos, smData.Ind, smData.Normal, smData.UV, smData.Col, smData.Tang, false);
	bShowStaticMesh = false;
	m_StaticMesh->SetVisibility(bShowStaticMesh);
	//StateExists = true;

	// Build Tri Array and Per Vert Shared Tri Array
	// BuildTriArrays();

	/*
	// -- LoadOBJ(const char* filename, ModelOBJ* obj)
	model->totalConnectedPoints = 0;
	model->totalConnectedTriangles = 0;
	model->vertices = new float[smData.vert_count];
	model->normals = new float[smData.vert_count*4];
	model->faces = new float[smData.vert_count * 4];
	model->bytes = smData.vert_count * 4;
	int triangleIndex = 0;
	int normalIndex = 0;

	int POINTS_PER_VERTEX = 3;
	int TOTAL_FLOATS_IN_TRIANGLE = 9;

	for (int32 i = 0; i < Particles.Num(); ++i)
	{
		model->vertices[model->totalConnectedPoints] = Particles[i].Position.X;
		model->vertices[model->totalConnectedPoints+1] = Particles[i].Position.X;
		model->vertices[model->totalConnectedPoints+2] = Particles[i].Position.X;

		model->totalConnectedPoints += POINTS_PER_VERTEX;
	}
	// -- Here: ogl.cpp line73
	for (int32 i = 0; i < smData.tri_count; i += 2)
	{
		model->faces[i] = smData.Tris[i].X;
		model->faces[i + 1] = smData.Tris[i].Y;
		model->faces[i + 2] = smData.Tris[i].Z;
		model->totalConnectedTriangles += 1;
	}
	for (int32 i = 0; i < smData.vert_count*3; ++i)
	{
		model->vertices[model->totalConnectedPoints] = Particles[i].Position.X;
		model->vertices[model->totalConnectedPoints + 1] = Particles[i].Position.X;
		model->vertices[model->totalConnectedPoints + 2] = Particles[i].Position.X;

	}
	*/

	m_meshes.push_back(mesh);

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

void UMyHairSim::get_spawn_triangles(const Mesh* m, float thresh, std::vector<Triangle>* faces)
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
	UE_LOG(LogTemp, Warning, TEXT("DBG::spawn tri AABB : min y : % f max y : % f\n"), min_y, max_y);
	/*	printf("spawn tri AABB: min y: %f max y: %f\n", min_y, max_y);*/
}

float length_sq(const Vector3f& v)
{
	return v.x() * v.x() + v.y() * v.y() + v.z() * v.z();
}
bool UMyHairSim::init_HairRoot(const Mesh* m, int num_spawns, float thresh)
{
	std::vector<Triangle> faces;
	kdtree* kd = kd_create(3);
	const float min_dist = 0.05;

	if (m == NULL) { UE_LOG(LogTemp, Error, TEXT("Invalid mesh.")); return false; };

	if (hair.size() < num_spawns)
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
			hair.push_back(strand);

			kd_insert3f(kd, rpoint.x(), rpoint.y(), rpoint.z(), 0);
		}

		kd_free(kd);

		for (size_t i = 0; i < hair.size(); i++)
		{
			hair[i].pos = hair[i].spawn_pt + hair[i].spawn_dir * hair_length;
		}

		UWorld* world = GetWorld();
		DrawDebugSphere(world, FVector(0, 50, 60), 75, 26, FColor::Blue, true, -1, 0, 1);
		UE_LOG(LogType, Error, TEXT("Show Hair: %d"), hair.size());
		for (int32 i = 0; i < hair.size() - 1; ++i)
		{
			FVector vec(hair[i].spawn_pt.x(), hair[i].spawn_pt.y(), hair[i].spawn_pt.z());
			DrawDebugPoint(world, vec, 9.f, FColor(255, 0, 255), true, 5.f);
		}
	}
	return true;
}

void UMyHairSim::InitHairModel()
{
	load_meshes();

	if (m_meshes.empty())
	{
		UE_LOG(LogTemp, Error, TEXT("ERR::HeadMesh::No Static Mesh Set")); return;
	}

	for (size_t i = 0; i < m_meshes.size(); i++)
	{
		m_meshes[i]->calc_bbox();
		mesh_head = m_meshes[i];
	}

	//	coll_sphere.radius = 1.0;
	//	coll_sphere.center = Vec3(0, 0.6, 0.53);

	if (!init_HairRoot(mesh_head, NUMSTRANDS, THRESH))
	{
		UE_LOG(LogTemp, Error, TEXT("ERR::HairMesh::Failed to initialize hair"));
		return;
	}

	// Root positions, Normal directions, Gravity
	pilar::Vector3f* roots = new pilar::Vector3f[NUMSTRANDS];
	pilar::Vector3f* normals = new pilar::Vector3f[NUMSTRANDS];
	for (int32 i = 0; i < hair.size(); ++i)
	{
		roots[i] = pilar::Vector3f(
			hair[i].spawn_pt.x(),
			hair[i].spawn_pt.z(),
			hair[i].spawn_pt.y());
		normals[i] = pilar::Vector3f(m_normal.X, m_normal.Y, m_normal.Z);
	}

	//Gravity
	pilar::Vector3f gravity(0.0f, GRAVITY, 0.0f);

	//Load geometry from file
	//loadOBJ("monkey.obj", &model); -- 일단 빼고 진행

	hairs = new pilar::CUHair(NUMSTRANDS, NUMPARTICLES, NUMCOMPONENTS, MASS,
		K_EDGE, K_BEND, K_TWIST, K_EXTRA,
		D_EDGE, D_BEND, D_TWIST, D_EXTRA,
		LENGTH_EDGE, LENGTH_BEND, LENGTH_TWIST,
		gravity,
		roots,
		normals);

	pilar::Vector3f* position = new pilar::Vector3f[NUMSTRANDS*NUMPARTICLES];
	hairs->get_state->position = position;

	pilar::Vector3f* pos = new pilar::Vector3f[NUMSTRANDS*NUMPARTICLES];
	hairs->get_state->pos = pos;

	pilar::Vector3f* velo = new pilar::Vector3f[NUMSTRANDS*NUMPARTICLES];
	hairs->get_state->velocity = velo;

	//Initialise positions along normals on the gpu
	hairs->initialise(position);

	for (int32 h = 0; h < hairs->h_state->numStrands; h++)
	{
		for (int32 par = 0; par < hairs->h_state->numParticles; par++)
		{
			FVector vec1(
				hairs->get_state->position[h * hairs->h_state->numParticles + par].x,
				hairs->get_state->position[h * hairs->h_state->numParticles + par].y,
				hairs->get_state->position[h * hairs->h_state->numParticles + par].z);
		}
	}

	IsInitMesh++;
}

void UMyHairSim::DoOnceSimulation()
{
	if (IsInitMesh == 0)
	{
		InitHairModel();
	}

	//SimpleHair->set_transform(head_xform);
	//SimpleHair->update(DeltaTime);
	//SimpleHair->draw();
	pilar::Vector3f* pos = hairs->get_state->position;

	for (int32 h = 0; h < hairs->h_state->numStrands; h++)
	{
		for (int32 par = 0; par < hairs->h_state->numParticles; par++)
		{
			UE_LOG(LogType, Error, TEXT("Update Hair %d"), h * hairs->h_state->numParticles + par);
			UWorld* world = GetWorld();

			FVector vec1(
				hairs->get_state->position[h * hairs->h_state->numParticles + par].x,
				hairs->get_state->position[h * hairs->h_state->numParticles + par].z,
				hairs->get_state->position[h * hairs->h_state->numParticles + par].y);
			DrawDebugPoint(world, vec1, 5.f, FColor(255, 255, 0), true);

			if (par + 1 < hairs->h_state->numParticles)
			{
				FVector vec2(
					hairs->get_state->position[h * hairs->h_state->numParticles + par + 1].x,
					hairs->get_state->position[h * hairs->h_state->numParticles + par + 1].z,
					hairs->get_state->position[h * hairs->h_state->numParticles + par + 1].y);

				DrawDebugLine(world, vec1, vec2, FColor::Emerald, true, -1, 0, 2.f);
			}
		}
	}
}

void UMyHairSim::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	if (bStartSimulate)
	{
		if (IsInitMesh == 0)
		{
			InitHairModel();
		}

		//SimpleHair->set_transform(head_xform);
		//SimpleHair->update(DeltaTime);
		//SimpleHair->draw();

		pilar::Vector3f* pos = hairs->get_state->pos;
		hairs->update(abs(DeltaTime) / 10.0f, pos);

		for (int32 h = 0; h < hairs->h_state->numStrands; h++)
		{
			UWorld* world = GetWorld();
			for (int32 par = 0; par < hairs->h_state->numParticles; par++)
			{
				FVector vec1(
					hairs->get_state->position[h * hairs->h_state->numParticles + par].x,
					hairs->get_state->position[h * hairs->h_state->numParticles + par].z,
					hairs->get_state->position[h * hairs->h_state->numParticles + par].y);
				DrawDebugPoint(world, vec1, 4.f, FColor::Green, false, 0.2f);

				if (par + 1 < hairs->h_state->numParticles)
				{
					FVector vec2(
						hairs->get_state->position[h * hairs->h_state->numParticles + par + 1].x,
						hairs->get_state->position[h * hairs->h_state->numParticles + par + 1].z,
						hairs->get_state->position[h * hairs->h_state->numParticles + par + 1].y);

					DrawDebugLine(world, vec1, vec2, FColor::Red, false, -1, 0, 0.8f);
				}
			}
			UE_LOG(LogType, Error, TEXT("Update %f time, Hair %d"), abs(DeltaTime) / 10.0f, h);
		}
	}
}

