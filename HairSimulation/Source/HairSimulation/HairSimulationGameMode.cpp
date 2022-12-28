// Copyright Epic Games, Inc. All Rights Reserved.

#include "HairSimulationGameMode.h"
#include "HairSimulationCharacter.h"
#include "UObject/ConstructorHelpers.h"

AHairSimulationGameMode::AHairSimulationGameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPersonCPP/Blueprints/ThirdPersonCharacter"));
	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}
