// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "ThirdParty\Eigen\Dense"
#include "kdtree.h"
#include <vector>
#include "CoreMinimal.h"

#include "constants.h"
#include "cu_hair.h"
#include "cuda_runtime.h"

#include "Components/SplineComponent.h"
#include "Components/SplineMeshComponent.h"
#include "Components/MeshComponent.h"
#include "ProceduralMeshComponent.h"
#include "MyHairSim.generated.h"

using namespace std;
typedef Eigen::Vector3f Vector3f;
typedef Eigen::Vector2f Vector2f;
typedef Eigen::Matrix4f Matrix4f;

/**
 * 
 */

 //+------------------ Cuda Library ------------------+//
extern "C"
void mallocStrands(pilar::HairState* h_state, pilar::HairState* &d_state);

extern "C"
void freeStrands(pilar::HairState* h_state, pilar::HairState* d_state);

extern "C"
void initialisePositions(pilar::HairState* h_state, pilar::HairState* d_state);

extern "C"
void updateStrands(float dt, pilar::HairState* h_state, pilar::HairState* d_state);

extern "C"
void copyRoots(pilar::Vector3f* roots, pilar::Vector3f* normals, float* grid, pilar::HairState* h_state);

extern "C"
void copyState(pilar::HairState* h_state, pilar::HairState* d_state);
//+----------------------------------------------------+//

struct FParticle
{
	FParticle()
		: Position(0, 0, 0)
		, Color(255, 255, 255, 255)
		, ID(-1)
		, C_idx(-1)
		, state(1)
		, conCount(0)
	{}

	FVector Position;
	FColor Color;

	int32 ID;
	uint32 C_idx;
	int8 state, conCount;
};

enum
{
	MESH_VERTEX = 1,
	MESH_NORMAL = 2,
	MESH_TEXCOORDS = 4,
	MESH_COLOR = 8,
	MESH_INDEX = 16
};

struct Aabb
{
	typedef Eigen::Vector3f Vector3f;

	Vector3f v0;
	Vector3f v1;

	Aabb(Vector3f _v0, Vector3f _v1) : v0(_v0), v1(_v1) {};
};

struct Material
{
	typedef Eigen::Vector3f Vector3f;

	Vector3f diffuse;
	Vector3f specular;
	float shininess;

	unsigned int tex;
	bool tex_opaque;
};

class MeshCustom
{
private:
	typedef Eigen::Vector3f Vector3f;
	typedef Eigen::Vector2f Vector2f;

	unsigned int vbo_vertices;
	unsigned int vbo_normals;
	unsigned int vbo_texcoords;
	unsigned int vbo_colors;
	unsigned int ibo;

public:
	MeshCustom();
	~MeshCustom();

	Aabb* bbox; //? Attach*
	Material* mtl;

	int num_vertices;
	int num_indices;

	vector<uint16_t> indices;
	vector<Vector3f> vertices;
	vector<Vector2f> texcoords;
	vector<Vector3f> normals;
	vector<Vector3f> colors;

	void draw() const;
	void update_vbo(unsigned int which);

	void calc_bbox();
};

struct HairStrand
{
	typedef Eigen::Vector3f Vector3f;

	Vector3f pos;
	Vector3f velocity;
	Vector3f spawn_pt;
	Vector3f spawn_dir;
};

struct Triangle
{
	typedef Eigen::Vector3f Vector3f;

	Vector3f v[3];
	Vector3f n[3];
};

namespace pilar
{
	void initDistanceField(ModelOBJ *obj, float* grid, HairState* state);

	class CUHair
	{
	protected:

	public:
		HairState* h_state; //Host_state
		HairState* d_state; //Device_state

		HairState* get_state;

		Sphere* head; //Model head

		CUHair(int numStrands,
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
			float* grid
		);
		~CUHair();

		void initialise(Vector3f* position);
		void update(float dt, Vector3f* position);
	};
}

#define MESH_ALL (0xffffffff)
#define MAX_NUM_SPAWNS NUMSTRANDS //1600
#define THRESH 0.5

UCLASS()
class HAIRSIMULATION_API UMyHairSim : public UMeshComponent
{
	GENERATED_BODY()
	
private:
	typedef Eigen::Vector3f Vector3f;
	typedef Eigen::Vector2f Vector2f;
	typedef Eigen::Matrix4f Matrix4f;
	typedef Eigen::Matrix3f Matrix3f;

	float hair_length;

	// -- Particle Data
	TArray<FParticle> Particles;
	TArray<FVector> Normals;
	//TArray<FParticle*> VolSamplePts;
	int32 particleCount;
	bool StateExists;

public:
	struct
	{
		// Static Mesh Deserialized(Á÷·ÄÈ­)
		TArray<FVector>            Pos;
		TArray<FColor>             Col;
		TArray<FVector>            Normal;
		TArray<FProcMeshTangent>   Tang;
		TArray<FVector2D>          UV;
		TArray<int32>              Ind;
		TArray<FIntVector>         Tris;

		// Vertices Shared Tris
		TArray<int32>* vtris;

		// SM Buffer Ptrs
		FPositionVertexBuffer* vb = nullptr; // Position Vertex Buffer (Position)
		FStaticMeshVertexBuffer* smvb = nullptr; // Static Mesh Buffer (Static Mesh)
		FColorVertexBuffer* cvb = nullptr; // Color Vertex Buffer (Color)
		FRawStaticIndexBuffer* ib = nullptr; // Tri Index Buffer (Index)

		int32 vert_count, ind_count, adj_count, tri_count;
		bool has_uv, has_col;
	} smData;

	// Mesh Properties
	UPROPERTY(VisibleAnywhere, BlueprintReadWrite)
		UStaticMeshComponent* m_StaticMesh;
	//UPROPERTY(VisibleAnywhere, BlueprintReadWrite)
	//	UProceduralMeshComponent * m_ProcedureMesh;

	UPROPERTY(EditAnywhere, Category = "Mesh")
		bool bShowStaticMesh = true;
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "Mesh")
		void load_meshes();
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "Hair")
		void DoOnceSimulation();
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Hair")
		bool bStartSimulate;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Hair")
		bool bIsInitMesh = true;
	UPROPERTY(EditAnywhere, CallInEditor, Category = "Hair")
		FVector m_normal = FVector(1.0f, 1.0f, 0.0f);
	
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Spline")
		UStaticMesh* SplineMesh;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Spline")
		class UMaterialInterface* DefaultMaterial;
	UPROPERTY(VisibleAnyWhere, BlueprintReadWrite, Category = "Spline", meta= (AllowPrivateAccess = "true"))
		USplineComponent* SplineComponent;

	TArray<USplineComponent*> SplineHairs;
	TArray<USplineMeshComponent*> SplineHairMeshes;
	//void OnRegister() override;
	vector<MeshCustom*> m_objects;
	MeshCustom* mesh_head;
	ModelOBJ* m_model;
	ModelOBJ* m_hand;

	vector<HairStrand> root_hair;
	pilar::Vector3f m_before = pilar::Vector3f(0.0f, 0.0f, 0.0f);
	pilar::Vector3f m_after = pilar::Vector3f(0.0f, 0.0f, 0.0f);
	pilar::Vector3f m_move = pilar::Vector3f(0.0f, 0.0f, 0.0f);
	pilar::CUHair* hairs;
	float prevTime = 0.0f;

	UMyHairSim(const FObjectInitializer& ObjectInitializer);

	TArray<FVector> Model_vertices;
	bool init_HairRoot(const MeshCustom* m, int num_spawns, float thresh = 0.4);
	void loadModel(ModelOBJ* obj);
	void UpdateModel(ModelOBJ* obj, pilar::Vector3f mov);

	void InitHairModel();
	void UpdateHairSpline();

	void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;
	static Vector3f calc_rand_point(const Triangle& tr, Vector3f* bary);
	static void get_spawn_triangles(const MeshCustom* m, float thresh, std::vector<Triangle>* faces);
};
