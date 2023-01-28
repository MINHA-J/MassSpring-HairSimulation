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
	CameraBoom->TargetArmLength = 300.0f; // The camera follows at this distance behind the character	
	CameraBoom->bUsePawnControlRotation = true; // Rotate the arm based on the controller

	// Create a follow camera
	FollowCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FollowCamera"));
	FollowCamera->SetupAttachment(CameraBoom, USpringArmComponent::SocketName); // Attach the camera to the end of the boom and let the boom adjust to match the controller orientation
	FollowCamera->bUsePawnControlRotation = false; // Camera does not rotate relative to arm

	//m_StaticMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MyStaticMesh"));
	//m_StaticMesh->SetVisibility(false);
	m_objects.clear();
	HairRoots.clear();
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
	PlayerInputComponent->BindAxis("Turn", this, &APawn::AddControllerYawInput);
	PlayerInputComponent->BindAxis("TurnRate", this, &ATestHairCharacter::TurnAtRate);
	PlayerInputComponent->BindAxis("LookUp", this, &APawn::AddControllerPitchInput);
	PlayerInputComponent->BindAxis("LookUpRate", this, &ATestHairCharacter::LookUpAtRate);

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
	smData.smvb = &(SMLodRender.StaticVertexBuffers.StaticMeshVertexBuffer);
	smData.cvb = &(SMLodRender.StaticVertexBuffers.ColorVertexBuffer);
	//FMultiSizeIndexContainer IndexContainer = SMLodRender.MultiSizeIndexContainer;
	//smData.ib = SMLodRender.MultiSizeIndexContainer.GetIndexBuffer();
	smData.has_uv = smData.smvb->GetNumTexCoords() != 0;
	smData.has_col = SM->bHasVertexColors;
	TMap<FVector, FColor> VertexColors = SM->GetVertexColorData(0);

	smData.vert_count = smData.vb->GetNumVertices();

#ifdef DEBUG_PRINT_LOG
	UE_LOG(LogTemp, Warning, TEXT("DBG::Static Mesh Vertex Count == %d | Index Count = %d"), smData.vert_count, smData.ind_count);
#endif
	// Initalize smData Arrays. 
	smData.Pos.AddDefaulted(smData.vert_count);

	//Update every vertex position in every render section
	for (int32 j = 0; j < SMLodRender.RenderSections.Num(); j++)
	{
		for (int32 i = 0; i < SMLodRender.RenderSections[j].GetNumVertices(); i++)
		{
			FVector VP = SMComponent->GetSkinnedVertexPosition(SMComponent, i, SMLodRender, *smData.vb);
			FVector VertexPosition = SMComponent->GetSkinnedVertexPosition(SMComponent, i, SMLodRender, *smData.vb)
				+ GetActorLocation()
				+ SMComponent->GetRelativeLocation()
				+ SMComponent->GetRelativeRotation().Vector();

			Model_vertices.Add(VertexPosition);

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
				if (SMComponent->GetVertexColor(i) != FColor::White)
					UE_LOG(LogTemp, Warning, TEXT("vertex color is not white"));

				FColor vc = SMComponent->GetVertexColor(i);
				Vector3f MeshCol = Vector3f(vc.R, vc.G, vc.B);
				cmesh->colors.push_back(MeshCol);
			}

			DrawDebugPoint(world, VertexPosition, 2, FColor(52, 220, 239), false, 0.1f);
		}
	}

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
		//DrawDebugSphere(world, FVector(0, 50, 60), 75, 26, FColor::Blue, true, -1, 0, 1);
		UE_LOG(LogType, Warning, TEXT("DBG::init_HairRoot - Show Hair: %d"), HairRoots.size());
		for (int32 i = 0; i < HairRoots.size() - 1; ++i)
		{
			FVector vec(HairRoots[i].spawn_pt.x(), HairRoots[i].spawn_pt.y(), HairRoots[i].spawn_pt.z());
			DrawDebugPoint(world, vec, 9.f, FColor(255, 0, 255), true, 5.f);
		}
	}
	return true;
}

void ATestHairCharacter::InitHairModel()
{
	//Save Skeletal Mesh Info
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

	if (!InitHairRoot(CharacterMesh, NUMSTRANDS, THRESH))
	{
		UE_LOG(LogTemp, Error, TEXT("ERR::InitHairModel::Failed to initialize hair"));
		return;
	}

	// Initialise CUHair

}

// Called every frame
void ATestHairCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	if (bStartSimulate)
	{
		UWorld* world = GetWorld();
		if (bIsInitMesh)
		{
			GEngine->AddOnScreenDebugMessage(-1, 3.0f, FColor::Blue, TEXT("Init Hair Model"));
			InitHairModel();
			bIsInitMesh = false;
		}
	}
}

