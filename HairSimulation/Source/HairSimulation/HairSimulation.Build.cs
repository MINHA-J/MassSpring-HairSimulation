// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class HairSimulation : ModuleRules
{
    private string project_root_path
    {
        get { return Path.Combine(ModuleDirectory, "../.."); }
    }

    public HairSimulation(ReadOnlyTargetRules Target) : base(Target)
	{

		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay" });

        string custom_cuda_lib_include = "CUDALib/include";
        string custom_cuda_lib_lib = "CUDALib/lib";

        PublicIncludePaths.Add(Path.Combine(project_root_path, custom_cuda_lib_include));
        PublicAdditionalLibraries.Add(Path.Combine(project_root_path, custom_cuda_lib_lib, "Cuda_lib_test.lib"));

        string cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3";
        string cuda_include = "include";
        string cuda_lib = "lib/x64";

        PublicIncludePaths.Add(Path.Combine(cuda_path, cuda_include));

        string cuda_sample_path = "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v11.3/common/inc";
        PublicIncludePaths.Add(cuda_sample_path);

        PublicAdditionalLibraries.Add(Path.Combine(cuda_path, cuda_lib, "cudart_static.lib"));
    }
}
