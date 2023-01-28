// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "ThirdParty\Eigen\Dense"
#include "kdtree.h"
#include <vector>
#include "constants.h"
#include "cu_hair.h"
#include "cuda_runtime.h"
#include "MyHairSim.h"

#include "Components/SplineComponent.h"
#include "Components/SplineMeshComponent.h"

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "TestHairCharacter.generated.h"


UCLASS(config = Game)
class HAIRSIMULATION_API ATestHairCharacter : public ACharacter
{
	GENERATED_BODY()

	/** Camera boom positioning the camera behind the character */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera, meta = (AllowPrivateAccess = "true"))
	class USpringArmComponent* CameraBoom;

	/** Follow camera */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera, meta = (AllowPrivateAccess = "true"))
	class UCameraComponent* FollowCamera;

private:
	typedef Eigen::Vector3f Vector3f;
	typedef Eigen::Vector2f Vector2f;
	typedef Eigen::Matrix4f Matrix4f;
	typedef Eigen::Matrix3f Matrix3f;
	
protected:
	/** Resets HMD orientation in VR. */
	void OnResetVR();

	/** Called for forwards/backward input */
	void MoveForward(float Value);

	/** Called for side to side input */
	void MoveRight(float Value);

	/**
	 * Called via input to turn at a given rate.
	 * @param Rate	This is a normalized rate, i.e. 1.0 means 100% of desired turn rate
	 */
	void TurnAtRate(float Rate);

	/**
	 * Called via input to turn look up/down at a given rate.
	 * @param Rate	This is a normalized rate, i.e. 1.0 means 100% of desired turn rate
	 */
	void LookUpAtRate(float Rate);

	/** Handler for when a touch input begins. */
	void TouchStarted(ETouchIndex::Type FingerIndex, FVector Location);

	/** Handler for when a touch input stops. */
	void TouchStopped(ETouchIndex::Type FingerIndex, FVector Location);

	// APawn interface
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;
	// End of APawn interface

public:	
	// Sets default values for this character's properties
	ATestHairCharacter();

	/** Base turn rate, in deg/sec. Other scaling may affect final turn rate. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera)
		float BaseTurnRate;
	/** Base look up/down rate, in deg/sec. Other scaling may affect final rate. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = Camera)
		float BaseLookUpRate;
	/** Returns CameraBoom subobject **/
	FORCEINLINE class USpringArmComponent* GetCameraBoom() const { return CameraBoom; }
	/** Returns FollowCamera subobject **/
	FORCEINLINE class UCameraComponent* GetFollowCamera() const { return FollowCamera; }
	
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

	//UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	//	UStaticMeshComponent* m_StaticMesh;

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "Hair Mesh")
		void LoadMeshes(); // Save Skeletal Mesh Info

	void InitHairModel(); // Init Hair root, Init CUHair

	vector<MeshCustom*> m_objects;
	TArray<FVector> Model_vertices;

	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
