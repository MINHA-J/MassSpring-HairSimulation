// Fill out your copyright notice in the Description page of Project Settings.
#pragma once
#include "TestHairCharacter.h"
#include "HeadMountedDisplayFunctionLibrary.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Components/InputComponent.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/Controller.h"
#include "GameFramework/SpringArmComponent.h"

#include "SkeletalMeshTypes.h"
#include "Engine/Public/Rendering/SkeletalMeshLODRenderData.h"
#include "Engine/Public/Rendering/SkeletalMeshRenderData.h"
#include "Engine/Public/Rendering/SkeletalMeshModel.h"
#include "Engine/SkeletalMesh.h"
#include "Components/SkinnedMeshComponent.h"
#include "Components/SkeletalMeshComponent.h"

#include "Engine/World.h"
#include "DrawDebugHelpers.h"

// Sets default values
ATestHairCharacter::ATestHairCharacter()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	// Set size for collision capsule
	GetCapsuleComponent()->InitCapsuleSize(42.f, 96.0f);

	// set our turn rates for input
	BaseTurnRate = 45.f;
	BaseLookUpRate = 45.f;

	// Don't rotate when the controller rotates. Let that just affect the camera.
	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;

	// Configure character movement
	GetCharacterMovement()->bOrientRotationToMovement = true; // Character moves in the direction of input...	
	GetCharacterMovement()->RotationRate = FRotator(0.0f, 540.0f, 0.0f); // ...at this rotation rate
	GetCharacterMovement()->JumpZVelocity = 600.f;
	GetCharacterMovement()->AirControl = 0.2f;

	// Create a camera boom (pulls in towards the player if there is a collision)
	CameraBoom = CreateDefaultSubobject<USpringArmComponent>(TEXT("CameraBoom"));
	CameraBoom->SetupAttachment(RootComponent);
	CameraBoom->TargetArmLength = 200.0f; // The camera follows at this distance behind the character	
	CameraBoom->bUsePawnControlRotation = true; // Rotate the arm based on the controller

	// Create a follow camera
	FollowCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FollowCamera"));
	FollowCamera->SetupAttachment(CameraBoom, USpringArmComponent::SocketName); // Attach the camera to the end of the boom and let the boom adjust to match the controller orientation
	FollowCamera->bUsePawnControlRotation = false; // Camera does not rotate relative to arm

	//m_StaticMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MyStaticMesh"));
	//m_StaticMesh->SetVisibility(false);
	m_objects.clear();
	HairRoots.clear();

	m_model = nullptr;
	m_hairs = nullptr;

	SplineComponent = CreateDefaultSubobject<USplineComponent>(TEXT("SplinePath"));
}

//////////////////////////////////////////////////////////////////////////
// Input

void ATestHairCharacter::SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent)
{
	// Set up gameplay key bindings
	check(PlayerInputComponent);
	PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &ACharacter::Jump);
	PlayerInputComponent->BindAction("Jump", IE_Released, this, &ACharacter::StopJumping);

	PlayerInputComponent->BindAxis("MoveForward", this, &ATestHairCharacter::MoveForward);
	PlayerInputComponent->BindAxis("MoveRight", this, &ATestHairCharacter::MoveRight);

	// We have 2 versions of the rotation bindings to handle different kinds of devices differently
	// "turn" handles devices that provide an absolute delta, such as a mouse.
	// "turnrate" is for devices that we choose to treat as a rate of change, such as an analog joystick
	//PlayerInputComponent->BindAxis("Turn", this, &APawn::AddControllerYawInput);
	//PlayerInputComponent->BindAxis("TurnRate", this, &ATestHairCharacter::TurnAtRate);
	//PlayerInputComponent->BindAxis("LookUp", this, &APawn::AddControllerPitchInput);
	//PlayerInputComponent->BindAxis("LookUpRate", this, &ATestHairCharacter::LookUpAtRate);

	// handle touch devices
	PlayerInputComponent->BindTouch(IE_Pressed, this, &ATestHairCharacter::TouchStarted);
	PlayerInputComponent->BindTouch(IE_Released, this, &ATestHairCharacter::TouchStopped);

	// VR headset functionality
	PlayerInputComponent->BindAction("ResetVR", IE_Pressed, this, &ATestHairCharacter::OnResetVR);
}

void ATestHairCharacter::OnResetVR()
{
	UHeadMountedDisplayFunctionLibrary::ResetOrientationAndPosition();
}

void ATestHairCharacter::TouchStarted(ETouchIndex::Type FingerIndex, FVector Location)
{
	Jump();
}

void ATestHairCharacter::TouchStopped(ETouchIndex::Type FingerIndex, FVector Location)
{
	StopJumping();
}

void ATestHairCharacter::TurnAtRate(float Rate)
{
	// calculate delta for this frame from the rate information
	AddControllerYawInput(Rate * BaseTurnRate * GetWorld()->GetDeltaSeconds());
}

void ATestHairCharacter::MoveForward(float Value)
{
	if ((Controller != nullptr) && (Value != 0.0f))
	{
		// find out which way is forward
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);

		// get forward vector
		const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
		AddMovementInput(Direction, Value);
	}
}

void ATestHairCharacter::MoveRight(float Value)
{
	if ((Controller != nullptr) && (Value != 0.0f))
	{
		// find out which way is right
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);

		// get right vector 
		const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
		// add movement in that direction
		AddMovementInput(Direction, Value);
	}
}

void ATestHairCharacter::LookUpAtRate(float Rate)
{
	// calculate delta for this frame from the rate information
	AddControllerPitchInput(Rate * BaseLookUpRate * GetWorld()->GetDeltaSeconds());
}

//+------------------------------------------------------------
void ATestHairCharacter::LoadMeshes()
{
	m_objects.clear();
	MeshCustom* cmesh = new MeshCustom();
	
	USkeletalMeshComponent* SMComponent = GetMesh();
	USkeletalMesh* SM = SMComponent->SkeletalMesh;
	FSkeletalMeshModel* SMResource = SM->GetImportedModel();
	FSkeletalMeshLODModel &SMModel = SMResource->LODModels[0];
	FSkeletalMeshRenderData* SMRenderData = SM->GetResourceForRendering();
	FSkeletalMeshLODRenderData& SMLodRender = SMRenderData->LODRenderData[0];
	//FSkinWeightVertexBuffer* SWVertexBuffer = SMComponent->GetSkinWeightBuffer(0);

	UWorld* world = GetWorld();
	
	smData.vb = SMComponent->GetSkinWeightBuffer(0);
	//smData.vb = &(SMLodRender.StaticVertexBuffers.PositionVertexBuffer);
	smData.smvb = &(SMLodRender.StaticVertexBuffers.StaticMeshVertexBuffer);
	smData.cvb = &(SMLodRender.StaticVertexBuffers.ColorVertexBuffer);
	//smData.ib = SMLodRender.MultiSizeIndexContainer.GetIndexBuffer();
	smData.has_uv = smData.smvb->GetNumTexCoords() != 0;
	smData.has_col = SM->bHasVertexColors;
	TMap<FVector, FColor> VertexColors = SM->GetVertexColorData(0);

	smData.vert_count = smData.vb->GetNumVertices();
	//smData.vert_count = 0;

#ifdef DEBUG_PRINT_LOG
	UE_LOG(LogTemp, Warning, TEXT("DBG::Static Mesh Vertex Count == %d | Index Count = %d"), smData.vert_count, smData.ind_count);
#endif
	// Initalize smData Arrays. 
	//smData.Pos.AddDefaulted(smData.vert_count);

	//Update every vertex position in every render section
	for (int32 j = 0; j < SMLodRender.RenderSections.Num(); j++)
	{
		for (int32 i = 0; i < SMLodRender.RenderSections[j].GetNumVertices(); i++)
		{	
			FVector VertexPosition = SMComponent->GetSkinnedVertexPosition(
				SMComponent, 
				SMLodRender.RenderSections[j].BaseVertexIndex + i, 
				SMLodRender, 
				*smData.vb) 
				+ GetActorLocation() + SMComponent->GetRelativeLocation();

			//VertexPosition = VertexPosition.RotateAngleAxis(90, FVector(0, 1 ,0));
			VertexPosition = VertexPosition.RotateAngleAxis(180, FVector(0, 0, 1)); 
			//VertexPosition = VertexPosition.RotateAngleAxis(90, FVector(1, 0, 0));

			//FVector VertexPosition = SMLodRender.StaticVertexBuffers.PositionVertexBuffer.VertexPosition(i) + GetActorLocation() + FVector(0, 0, -97);

			Model_vertices.Add(VertexPosition);
			smData.Pos.Add(VertexPosition);

			// Mesh Init
			Vector3f MeshVertex = Vector3f(VertexPosition.X, VertexPosition.Y, VertexPosition.Z);
			cmesh->vertices.push_back(MeshVertex);

			Vector3f MeshNormal = Vector3f(
				smData.smvb->VertexTangentZ(i).X,
				smData.smvb->VertexTangentZ(i).Y,
				smData.smvb->VertexTangentZ(i).Z);
			cmesh->normals.push_back(MeshNormal);

			if (smData.has_col)
			{
				FColor vc = SMComponent->GetVertexColor(SMLodRender.RenderSections[j].BaseVertexIndex + i);
				Vector3f MeshCol = Vector3f(vc.R, vc.G, vc.B);
				cmesh->colors.push_back(MeshCol);
			}

			//DrawDebugPoint(world, VertexPosition, 2, FColor(52, 220, 239), true);
		}
	}
	
	uint32 num = SMModel.NumVertices;
	TArray<uint32> indices = SMModel.IndexBuffer;
	smData.ind_count = indices.Num();
	smData.tri_count = smData.ind_count / 3;

	// Initalize smData Arrays. 
	smData.Ind.AddDefaulted(smData.ind_count);

	// Indices 
	for (int32 i = 0; i < smData.ind_count; ++i)
	{
		smData.Ind[i] = static_cast<int32>(indices[i]);
		cmesh->indices.push_back(smData.Ind[i]);
	}

	m_objects.push_back(cmesh);
}


//+------------------ InitHairRoot
#define K_ANC 4.0
#define DAMPING 1.5
Vector3f ATestHairCharacter::calc_rand_point(const Triangle& tr, Vector3f* bary)
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
void ATestHairCharacter::get_spawn_triangles(const MeshCustom* m, float thresh, std::vector<Triangle>* faces)
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
float ATestHairCharacter::length_sq(const Vector3f& v)
{
	return v.x() * v.x() + v.y() * v.y() + v.z() * v.z();
}
bool ATestHairCharacter::InitHairRoot(const MeshCustom* m, int num_spawns, float thresh)
{
	std::vector<Triangle> faces;
	kdtree* kd = kd_create(3);
	const float min_dist = 0.05;

	if (m == NULL) { UE_LOG(LogTemp, Error, TEXT("Invalid mesh.")); return false; };

	if (HairRoots.size() < num_spawns)
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
			HairRoots.push_back(strand);

			kd_insert3f(kd, rpoint.x(), rpoint.y(), rpoint.z(), 0);
		}

		kd_free(kd);

		for (size_t i = 0; i < HairRoots.size(); i++)
		{
			HairRoots[i].pos = HairRoots[i].spawn_pt;
		}

		UWorld* world = GetWorld();
		//-- Visualize Hair Root
		//DrawDebugSphere(world, FVector(0, 50, 60), 75, 26, FColor::Blue, true, -1, 0, 1);
		UE_LOG(LogType, Warning, TEXT("DBG::init_HairRoot - Show Hair: %d"), HairRoots.size());
		//for (int32 i = 0; i < HairRoots.size() - 1; ++i)
		//{
		//	FVector vec(HairRoots[i].spawn_pt.x(), HairRoots[i].spawn_pt.y(), HairRoots[i].spawn_pt.z());
		//	DrawDebugPoint(world, vec, 9.f, FColor(255, 0, 255), true, 5.f);
		//}
	}
	return true;
}

void ATestHairCharacter::LoadModel(ModelOBJ* obj)
{
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
		obj->vertices[obj->totalConnectedPoints] = smData.Pos[i].X;
		obj->vertices[obj->totalConnectedPoints + 1] = smData.Pos[i].Z;
		obj->vertices[obj->totalConnectedPoints + 2] = smData.Pos[i].Y;

		//FVector v(obj->vertices[obj->totalConnectedPoints], obj->vertices[obj->totalConnectedPoints + 2], obj->vertices[obj->totalConnectedPoints + 1]);
		//DrawDebugPoint(world, v, 2, FColor::Blue, true);
		obj->totalConnectedPoints += POINTS_PER_VERTEX;
	}

	int num = 0;
	for (int i = 0; i < smData.tri_count; ++i)
	{
		int vertexNumber[4] = { 0, 0, 0 };
		vertexNumber[0] = smData.Ind[3 * i];
		vertexNumber[1] = smData.Ind[3 * i + 1];
		vertexNumber[2] = smData.Ind[3 * i + 2];

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

		FVector vec1(obj->faces[triangleIndex], obj->faces[triangleIndex + 1], obj->faces[triangleIndex + 2]);
		FVector vec2(obj->faces[triangleIndex + 3], obj->faces[triangleIndex + 4], obj->faces[triangleIndex + 5]);
		FVector vec3(obj->faces[triangleIndex + 6], obj->faces[triangleIndex + 7], obj->faces[triangleIndex + 8]);
		//DrawDebugLine(world, vec1, vec2, FColor::Blue, true, -1, 0, 0.1);
		//DrawDebugLine(world, vec2, vec3, FColor::Blue, true, -1, 0, 0.1);
		//DrawDebugLine(world, vec3, vec1, FColor::Blue, true, -1, 0, 0.1);

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
}

void ATestHairCharacter::UpdateModel(ModelOBJ* obj, pilar::Vector3f mov)
{
	int triangleIndex = 0;
	int normalIndex = 0;
	UWorld* world = GetWorld();
	FVector distance = FVector(mov.x, mov.y, mov.z);

	USkeletalMeshComponent* SMComponent = GetMesh();
	USkeletalMesh* SM = SMComponent->SkeletalMesh;
	FSkeletalMeshRenderData* SMRenderData = SM->GetResourceForRendering();
	FSkeletalMeshLODRenderData& SMLodRender = SMRenderData->LODRenderData[0];

	for (int32 j = 0; j < SMLodRender.RenderSections.Num(); j++)
	{
		for (int32 i = 0; i < SMLodRender.RenderSections[j].GetNumVertices(); i++)
		{
			FVector VertexPosition = SMComponent->GetSkinnedVertexPosition(
				SMComponent,
				SMLodRender.RenderSections[j].BaseVertexIndex + i,
				SMLodRender,
				*smData.vb)
				+ GetActorLocation() + SMComponent->GetRelativeLocation();
			//VertexPosition = VertexPosition.RotateAngleAxis(180, FVector(0, 0, 1));
			VertexPosition = VertexPosition.RotateAngleAxis(180, FVector(0, 0, 1));
			Model_vertices[i] = VertexPosition;
		}
	}

	// v: x, y, z의 형식의 데이터
	//for (int32 i = 0; i < smData.vert_count; ++i)
	//{
	//	Model_vertices[i] += distance;
	//}

	for (int i = 0; i < smData.tri_count; ++i)
	{
		int vertexNumber[4] = { 0, 0, 0 };
		vertexNumber[0] = smData.Ind[3 * i];
		vertexNumber[1] = smData.Ind[3 * i + 1];
		vertexNumber[2] = smData.Ind[3 * i + 2];

		int tCounter = 0;
		for (int j = 0; j < POINTS_PER_VERTEX; j++)
		{
			obj->faces[triangleIndex + tCounter] = Model_vertices[vertexNumber[j]].X;
			obj->faces[triangleIndex + tCounter + 1] = Model_vertices[vertexNumber[j]].Z;
			obj->faces[triangleIndex + tCounter + 2] = Model_vertices[vertexNumber[j]].Y;
			tCounter += POINTS_PER_VERTEX;
		}

		float coord1[3] = { obj->faces[triangleIndex], obj->faces[triangleIndex + 1], obj->faces[triangleIndex + 2] };
		float coord2[3] = { obj->faces[triangleIndex + 3], obj->faces[triangleIndex + 4], obj->faces[triangleIndex + 5] };
		float coord3[3] = { obj->faces[triangleIndex + 6], obj->faces[triangleIndex + 7], obj->faces[triangleIndex + 8] };

		FVector vec1(obj->faces[triangleIndex], obj->faces[triangleIndex + 2], obj->faces[triangleIndex + 1]);
		FVector vec2(obj->faces[triangleIndex + 3], obj->faces[triangleIndex + 5], obj->faces[triangleIndex + 4]);
		FVector vec3(obj->faces[triangleIndex + 6], obj->faces[triangleIndex + 8], obj->faces[triangleIndex + 7]);
		//DrawDebugLine(world, vec1, vec2, FColor::Blue, false, 0.5f, 0, 0.1);
		//DrawDebugLine(world, vec2, vec3, FColor::Blue, false, 0.5f, 0, 0.1);
		//DrawDebugLine(world, vec3, vec1, FColor::Blue, false, 0.5f, 0, 0.1);

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

void ATestHairCharacter::InitHairModel()
{
	//---Save Skeletal Mesh Info
	LoadMeshes();
	if (m_objects.empty())
	{
		UE_LOG(LogTemp, Error, TEXT("ERR::InitHairModel::No Static Mesh Set")); return;
	}

	MeshCustom* CharacterMesh = nullptr;
	for (size_t i = 0; i < m_objects.size(); i++)
	{
		m_objects[i]->calc_bbox();
		CharacterMesh = m_objects[i];
	}

	int numStrands = 40; 
	if (!InitHairRoot(CharacterMesh, numStrands, THRESH))
	{
		UE_LOG(LogTemp, Error, TEXT("ERR::InitHairModel::Failed to initialize hair"));
		return;
	}

	//---Initialise CUHair
	// Root positions, Normal directions
	pilar::Vector3f* roots = new pilar::Vector3f[numStrands];
	pilar::Vector3f* normals = new pilar::Vector3f[numStrands];
	for (int32 i = 0; i < HairRoots.size(); ++i)
	{
		roots[i] = pilar::Vector3f(
			HairRoots[i].spawn_pt.x(),
			HairRoots[i].spawn_pt.z(),
			HairRoots[i].spawn_pt.y());

		//normals[i] = pilar::Vector3f((HairRoots[i].spawn_pt.x()), (HairRoots[i].spawn_pt.z()), 0);
		normals[i] = pilar::Vector3f(m_normal.X, m_normal.Z, m_normal.Y);
	}

	// Gravity
	pilar::Vector3f gravity(0.0f, -GRAVITY, 0.0f);

	//Load geometry from file
	m_model = new ModelOBJ;
	LoadModel(m_model);
	float* grid = new float[DOMAIN_DIM*DOMAIN_DIM*DOMAIN_DIM];

	// Initialise get_state 
	//m_hairs = new pilar::CUHair(NUMSTRANDS, NUMPARTICLES, NUMCOMPONENTS, 1, MASS,
	//	K_EDGE, K_BEND, K_TWIST, K_EXTRA,
	//	D_EDGE, D_BEND, D_TWIST, D_EXTRA,
	//	LENGTH_EDGE, LENGTH_BEND, LENGTH_TWIST,
	//	DOMAIN_DIM, DOMAIN_WIDTH, DOMAIN_HALF, CELL_WIDTH, CELL_HALF,
	//	gravity,
	//	roots,
	//	normals,
	//	m_model,
	//	grid);

	m_hairs = new pilar::CUHair(numStrands, NUMPARTICLES, NUMCOMPONENTS, 1, MASS,
		3200.f, 80.f, 80.f, 4.905f,
		32000.f, 2500.f, 2500.f, 125.f,
		5.f, 5.f, 5.f,
		DOMAIN_DIM, 690.f, 345.f, 6.9f, 3.45f,
		gravity,
		roots,
		normals,
		m_model,
		grid);

	//Initialise positions along normals on the gpu
	m_hairs->initialise(m_hairs->get_state->position);

	UE_LOG(LogTemp, Warning, TEXT("DBG::InitHairModel - finish"));
}

void ATestHairCharacter::UpdateHairSpline()
{
	/****************************************
	이전의 Spline 위치와 Mesh를 없애 할당을 하나로 함
	*****************************************/
	if (m_SplineHairs.Num() > 0)
	{
		for (int32 i = 0; i < m_SplineHairs.Num(); i++)
		{
			m_SplineHairs[i]->ClearSplinePoints(true);
		}
	}

	if (m_SplineHairMeshes.Num() > 0)
	{
		for (int32 i = 0; i < m_SplineHairMeshes.Num(); i++)
		{
			if (m_SplineHairMeshes[i])
				m_SplineHairMeshes[i]->DestroyComponent();
		}
	}
	m_SplineHairs.Empty();
	m_SplineHairMeshes.Empty();

	for (int32 h = 0; h < m_hairs->h_state->numStrands; h++)
	{
		//USplineComponent* SplineComponent = NewObject<USplineComponent>(this, USplineComponent::StaticClass());
		FVector root(
			m_hairs->get_state->root[h].x, 
			m_hairs->get_state->root[h].z,
			m_hairs->get_state->root[h].y);
		FVector first(
			m_hairs->get_state->position[h * m_hairs->h_state->numParticles].x,
			m_hairs->get_state->position[h * m_hairs->h_state->numParticles].z,
			m_hairs->get_state->position[h * m_hairs->h_state->numParticles].y);
		SplineComponent->AddSplinePointAtIndex(root, 0, ESplineCoordinateSpace::World);
		//SplineComponent->AddSplinePointAtIndex(first, 1, ESplineCoordinateSpace::World);

		for (int32 SplineCount = 0; SplineCount < m_hairs->h_state->numParticles; SplineCount++)
		{
			FVector vector(
				m_hairs->get_state->position[h * m_hairs->h_state->numParticles + SplineCount].x,
				m_hairs->get_state->position[h * m_hairs->h_state->numParticles + SplineCount].z,
				m_hairs->get_state->position[h * m_hairs->h_state->numParticles + SplineCount].y);
			SplineComponent->AddSplinePointAtIndex(vector, SplineCount + 1, ESplineCoordinateSpace::World);
			FVector Tangent = SplineComponent->GetLocationAtSplinePoint(SplineCount + 1, ESplineCoordinateSpace::Local) - SplineComponent->GetLocationAtSplinePoint(SplineCount, ESplineCoordinateSpace::Local);

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

			m_SplineHairMeshes.Add(HairSplineComponent);
		}
		m_SplineHairs.Add(SplineComponent);
	}
}

void ATestHairCharacter::DoOnceSimulation()
{
	InitHairModel();
	GEngine->AddOnScreenDebugMessage(-1, 3.0f, FColor::Blue, TEXT("Init Hair Model"));

	pilar::Vector3f* pos = m_hairs->get_state->position;

	for (int32 h = 0; h < m_hairs->h_state->numStrands; ++h)
	{
		for (int32 par = 0; par < m_hairs->h_state->numParticles; ++par)
		{
			UE_LOG(LogType, Warning, TEXT("DoOnceSimulation - Update Hair %d"), h * m_hairs->h_state->numParticles + par);
			UWorld* world = GetWorld();

			FVector vec1(
				pos[h * m_hairs->h_state->numParticles + par].x,
				pos[h * m_hairs->h_state->numParticles + par].z,
				pos[h * m_hairs->h_state->numParticles + par].y);
			//DrawDebugPoint(world, vec1, 5.f, FColor(255, 255, 0), 2.f);

			if (par + 1 < m_hairs->h_state->numParticles)
			{
				FVector vec2(
					pos[h * m_hairs->h_state->numParticles + par + 1].x,
					pos[h * m_hairs->h_state->numParticles + par + 1].z,
					pos[h * m_hairs->h_state->numParticles + par + 1].y);

				//DrawDebugLine(world, vec1, vec2, FColor::Emerald, true, -1, 0, 2.f);
			}
		}
	}
	UpdateHairSpline();
	UE_LOG(LogType, Warning, TEXT("DoOnceSimulation - Finish"));
}


// Called every frame
void ATestHairCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (bStartSimulate)
	{
		USkeletalMeshComponent* SMComponent = GetMesh();
		USkeletalMesh* SM = SMComponent->SkeletalMesh;
		FSkeletalMeshRenderData* SMRenderData = SM->GetResourceForRendering();
		FSkeletalMeshLODRenderData& SMLodRender = SMRenderData->LODRenderData[0];

		m_before = pilar::Vector3f(GetActorLocation().X, GetActorLocation().Y, GetActorLocation().Z);
		FVector VertexPosition = SMComponent->GetSkinnedVertexPosition(
			SMComponent, 
			SMLodRender.RenderSections[1].BaseVertexIndex + 10,
			SMRenderData->LODRenderData[0], 
			*SMComponent->GetSkinWeightBuffer(0)) + SMComponent->GetRelativeLocation();
		VertexPosition = VertexPosition.RotateAngleAxis(180, FVector(0, 0, 1));
		m_MeshBefore = pilar::Vector3f(VertexPosition.X, VertexPosition.Y, VertexPosition.Z);

		UWorld* world = GetWorld();
		if (bIsInitMesh)
		{
			GEngine->AddOnScreenDebugMessage(-1, 3.0f, FColor::Blue, TEXT("Init Hair Model"));
			InitHairModel();

			VertexPosition = SMComponent->GetSkinnedVertexPosition(
				SMComponent, 
				SMLodRender.RenderSections[1].BaseVertexIndex + 10,
				SMRenderData->LODRenderData[0],
				*SMComponent->GetSkinWeightBuffer(0))
				+ SMComponent->GetRelativeLocation();
			VertexPosition = VertexPosition.RotateAngleAxis(108, FVector(0, 0, 1));
			m_MeshBefore = pilar::Vector3f(VertexPosition.X, VertexPosition.Y, VertexPosition.Z);
			DrawDebugBox(world, 
				FVector(0, 0, 0), 
				FVector(m_hairs->get_state->cell_width * DOMAIN_DIM, m_hairs->get_state->cell_width * DOMAIN_DIM, m_hairs->get_state->cell_width * DOMAIN_DIM),
				FColor::Purple, true, -1, 0, 5);
			bIsInitMesh = false;
		}

		//---Move distance
		m_move = m_before - m_after;
		if (m_Meshafter.length() > 0)
		{
			m_Meshmove = m_MeshBefore - m_Meshafter;
		}

		//---If Move, Update Model Grid
		if (m_move.length() > 0 || m_Meshmove.length() > 0)
		{
			UpdateModel(m_model, m_move);
			pilar::initDistanceField(m_model, m_hairs->get_state->grid, m_hairs->get_state);
		}

		//---Update Hair
		pilar::Vector3f* position = m_hairs->get_state->position;
		m_hairs->update(abs(DeltaTime) / 10.0f, position);
		for (int32 h = 0; h < m_hairs->h_state->numStrands; h++)
		{
			m_hairs->get_state->root[h].x += m_move.x + m_Meshmove.x;
			m_hairs->get_state->root[h].y += m_move.z + m_Meshmove.z;
			//m_hairs->get_state->root[h].z += m_move.y;// +m_Meshmove.y;

			//m_hairs->get_state->root[h].x += m_move.x;
			//m_hairs->get_state->root[h].y += m_move.z;
			//m_hairs->get_state->root[h].z += m_move.y;

			FVector root(m_hairs->get_state->root[h].x, m_hairs->get_state->root[h].z, m_hairs->get_state->root[h].y);
			FVector first(
				m_hairs->get_state->position[h * m_hairs->h_state->numParticles].x + m_move.x + m_Meshmove.x,
				m_hairs->get_state->position[h * m_hairs->h_state->numParticles].z + m_move.y + m_Meshmove.y,
				m_hairs->get_state->position[h * m_hairs->h_state->numParticles].y + m_move.z);
			//FVector first(
			//	m_hairs->get_state->position[h * m_hairs->h_state->numParticles].x + m_move.x,
			//	m_hairs->get_state->position[h * m_hairs->h_state->numParticles].z + m_move.y,
			//	m_hairs->get_state->position[h * m_hairs->h_state->numParticles].y + m_move.z);
			//DrawDebugLine(world, root, first, FColor::Emerald, false, -1, 0, 0.8f);

			for (int32 par = 0; par < m_hairs->h_state->numParticles; par++)
			{
				m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].x += m_move.x + m_Meshmove.x;
				m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].y += m_move.z + m_Meshmove.z;
				//m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].z += m_move.y + m_Meshmove.y;	

				//m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].x += m_move.x;
				//m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].y += m_move.z;
				//m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].z += m_move.y;

				//--- For Debug Particle position & velocity
				//UE_LOG(LogType, Log, TEXT("Position %d Hair, %d Particle - x:%f, y:%f, z:%f"),
				//	h, par,
				//	m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].x,
				//	m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].z,
				//	m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].y);
				//UE_LOG(LogType, Log, TEXT("Velocity %d Hair, %d Particle - x:%f, y:%f, z:%f"),
				//	h, par,
				//	m_hairs->get_state->velocity[h * m_hairs->h_state->numParticles + par].x,
				//	m_hairs->get_state->velocity[h * m_hairs->h_state->numParticles + par].z,
				//	m_hairs->get_state->velocity[h * m_hairs->h_state->numParticles + par].y);

				//--- For visualize
				FVector vec1(
					m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].x,
					m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].z,
					m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par].y);
				//DrawDebugPoint(world, vec1, 4.f, FColor::Green, false, 0.1f);

				if (par + 1 < m_hairs->h_state->numParticles)
				{
					FVector vec2(
						m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par + 1].x + m_move.x + m_Meshmove.x,
						m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par + 1].z + m_move.y + m_Meshmove.y,
						m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par + 1].y + m_move.z);
					
					//FVector vec2(
					//	m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par + 1].x + m_move.x,
					//	m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par + 1].z + m_move.y,
					//	m_hairs->get_state->position[h * m_hairs->h_state->numParticles + par + 1].y + m_move.z);


					//--- For visualize
					//DrawDebugLine(world, vec1, vec2, FColor::Red, false, -1, 0, 0.8f);
				}
			}

			m_after = pilar::Vector3f(GetActorLocation().X, GetActorLocation().Y, GetActorLocation().Z);

			VertexPosition = SMComponent->GetSkinnedVertexPosition(
				SMComponent,
				SMLodRender.RenderSections[1].BaseVertexIndex + 10,
				SMRenderData->LODRenderData[0],
				*SMComponent->GetSkinWeightBuffer(0))
				+ SMComponent->GetRelativeLocation();
			VertexPosition = VertexPosition.RotateAngleAxis(180, FVector(0, 0, 1));
			m_Meshafter = pilar::Vector3f(VertexPosition.X, VertexPosition.Y, VertexPosition.Z);
			//DrawDebugPoint(world, VertexPosition, 4.f, FColor::Green, false, 0.1f);
			//UE_LOG(LogType, Log, TEXT("Particle - x:%f, y:%f, z:%f"), m_Meshmove.x, m_Meshmove.y, m_Meshmove.z);
		}
		UpdateHairSpline();
	}
}

