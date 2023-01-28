// Fill out your copyright notice in the Description page of Project Settings.


#include "TestHairCharacterGameMode.h"

ATestHairCharacterGameMode::ATestHairCharacterGameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPersonCPP/Blueprints/MyTestHairCharacter"));

	if (PlayerPawnBPClass.Class != NULL)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}
}