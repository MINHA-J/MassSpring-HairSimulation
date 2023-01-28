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
	FSkeletalMeshRenderData* SMRenderData = SM->GetResourceForRendering();
	FSkeletalMeshLODRenderData& SMLodRender = SMRenderData->LODRenderData[0];
	FSkinWeightVertexBuffer& SWVertexBuffer = *(SMComponent->GetSkinWeightBuffer(0));

	UWorld* world = GetWorld();

	smData.smvb = &(SMLodRender.StaticVertexBuffers.StaticMeshVertexBuffer);

	//Update every vertex position in every render section
	for (int32 j = 0; j < SMLodRender.RenderSections.Num(); j++)
	{
		for (int32 i = 0; i < SMLodRender.RenderSections[j].GetNumVertices(); i++)
		{
			FVector VertexPosition = SMComponent->GetSkinnedVertexPosition(SMComponent, i, SMLodRender, SWVertexBuffer)
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

			if (smData.has_uv)
			{
				Vector2f MeshUV = Vector2f(smData.UV[i].X, smData.UV[i].Y);
				cmesh->texcoords.push_back(MeshUV);
			}

			if (smData.has_col)
			{
				Vector3f MeshCol = Vector3f(smData.Col[i].R, smData.Col[i].G, smData.Col[i].B);
				cmesh->colors.push_back(MeshCol);
			}
			DrawDebugPoint(world, VertexPosition, 2, FColor(52, 220, 239), false, 0.1f);
		}
	}

	// Indices 
	for (int32 i = 0; i < smData.ind_count; ++i)
	{
		smData.Ind[i] = static_cast<int32>(smData.ib->GetIndex(i));
		cmesh->indices.push_back(smData.Ind[i]);
	}

	m_objects.push_back(cmesh);
}

void ATestHairCharacter::InitHairModel()
{
	//Save Skeletal Mesh Info
	LoadMeshes();
}

// Called every frame
void ATestHairCharacter::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	LoadMeshes();
}

