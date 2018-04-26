# Compiler flags...
CPP_COMPILER = g++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=-I"../src/libaten" -I"../src/libidaten" -I"../3rdparty/glew/include" -I"../3rdparty/glm" 
Release_Include_Path=-I"../src/libaten" -I"../src/libidaten" -I"../3rdparty/glew/include" -I"../3rdparty/glm" 

# Library paths...
Debug_Library_Path=
Release_Library_Path=

# Additional libraries...
Debug_Libraries=-Wl,--no-as-needed -Wl,--start-group -lcudart  -Wl,--end-group
Release_Libraries=-Wl,--no-as-needed -Wl,--start-group -lcudart  -Wl,--end-group

# Preprocessor definitions...
Debug_Preprocessor_Definitions=-D GCC_BUILD -D __AT_DEBUG__ -D __AT_CUDA__ 
Release_Preprocessor_Definitions=-D GCC_BUILD -D NDEBUG -D __AT_CUDA__ 

# Implictly linked object files...
Debug_Implicitly_Linked_Objects=
Release_Implicitly_Linked_Objects=

# Compiler flags...
Debug_Compiler_Flags=-O0 -g -std=c++11 -fopenmp 
Release_Compiler_Flags=-O2 -std=c++11 -fopenmp 

# Location of the CUDA Toolkit
CUDA_PATH?="/usr/local/cuda"
NVCC:= $(CUDA_PATH)/bin/nvcc -ccbin $(CPP_COMPILER)

Debug_Include_Path+=-I"/usr/local/cuda/include"
Release_Include_Path+=-I"/usr/local/cuda/include"

Debug_Library_Path+=-L"/usr/local/cuda/lib64"
Release_Library_Path+=-L"/usr/local/cuda/lib64"

# Builds all configurations for this project...
.PHONY: build_all_configurations
build_all_configurations: Debug Release 

# Builds the Debug configuration...
.PHONY: Debug
Debug: create_folders x64/Debug/libidaten/src/libidaten/cuda/cudaGLresource.o x64/Debug/libidaten/src/libidaten/cuda/cudamemory.o x64/Debug/libidaten/src/libidaten/kernel/pathtracing.o x64/Debug/libidaten/src/libidaten/kernel/renderer.o x64/Debug/libidaten/src/libaten/light/arealight.o x64/Debug/libidaten/src/libaten/light/ibl.o x64/Debug/libidaten/src/libaten/light/light.o x64/Debug/libidaten/src/libaten/material/layer.o x64/Debug/libidaten/src/libaten/math/mat4.o x64/Debug/libidaten/src/libaten/sampler/sobol.o x64/Debug/libidaten/src/libaten/renderer/envmap.o x64/Debug/libidaten/src/libaten/os/linux/misc/timer_linux.o x64/Debug/libidaten/src/libaten/misc/color.o x64/Debug/libidaten/src/libidaten/svgf/svgf.o x64/Debug/libidaten/src/libaten/material/carpaint.o x64/Debug/libidaten/src/libaten/material/FlakesNormal.o x64/Debug/libidaten/src/libaten/material/disney_brdf.o x64/Debug/libidaten/src/libaten/geometry/sphere.o x64/Debug/libidaten/src/libidaten/cuda/cudaTextureResource.o x64/Debug/libidaten/src/libaten/camera/pinhole.o x64/Debug/libidaten/src/libaten/material/beckman.o x64/Debug/libidaten/src/libaten/material/blinn.o x64/Debug/libidaten/src/libaten/material/ggx.o x64/Debug/libidaten/src/libaten/material/material.o x64/Debug/libidaten/src/libaten/material/oren_nayar.o x64/Debug/libidaten/src/libaten/material/refraction.o x64/Debug/libidaten/src/libaten/material/specular.o x64/Debug/libidaten/src/libidaten/kernel/LBVHBuilder.o x64/Debug/libidaten/src/libidaten/kernel/qbvh.o x64/Debug/libidaten/src/libidaten/kernel/RadixSort.o x64/Debug/libidaten/src/libidaten/kernel/sample_texture_impl.o x64/Debug/libidaten/src/libidaten/kernel/sbvh.o x64/Debug/libidaten/src/libidaten/kernel/Skinning.o x64/Debug/libidaten/src/libidaten/kernel/stackless_bvh.o x64/Debug/libidaten/src/libidaten/kernel/stackless_qbvh.o x64/Debug/libidaten/src/libidaten/svgf/svgf_atrous.o x64/Debug/libidaten/src/libidaten/svgf/svgf_debug.o x64/Debug/libidaten/src/libidaten/svgf/svgf_init.o x64/Debug/libidaten/src/libidaten/svgf/svgf_pt.o x64/Debug/libidaten/src/libidaten/svgf/svgf_ssrt.o x64/Debug/libidaten/src/libidaten/svgf/svgf_tile.o x64/Debug/libidaten/src/libidaten/svgf/svgf_tp.o x64/Debug/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o x64/Debug/libidaten/src/libidaten/kernel/bvh.o x64/Debug/libidaten/src/libidaten/kernel/compaction.o x64/Debug/libidaten/src/libidaten/kernel/intersect.o x64/Debug/libidaten/src/libidaten/kernel/light.o x64/Debug/libidaten/src/libidaten/kernel/material.o x64/Debug/libidaten/src/libidaten/kernel/pathtracing_impl.o x64/Debug/libidaten/gpuCode.o 
	ar rcs x64/Debug/libidaten.a x64/Debug/libidaten/src/libidaten/cuda/cudaGLresource.o x64/Debug/libidaten/src/libidaten/cuda/cudamemory.o x64/Debug/libidaten/src/libidaten/kernel/pathtracing.o x64/Debug/libidaten/src/libidaten/kernel/renderer.o x64/Debug/libidaten/src/libaten/light/arealight.o x64/Debug/libidaten/src/libaten/light/ibl.o x64/Debug/libidaten/src/libaten/light/light.o x64/Debug/libidaten/src/libaten/material/layer.o x64/Debug/libidaten/src/libaten/math/mat4.o x64/Debug/libidaten/src/libaten/sampler/sobol.o x64/Debug/libidaten/src/libaten/renderer/envmap.o x64/Debug/libidaten/src/libaten/os/linux/misc/timer_linux.o x64/Debug/libidaten/src/libaten/misc/color.o x64/Debug/libidaten/src/libidaten/svgf/svgf.o x64/Debug/libidaten/src/libaten/material/carpaint.o x64/Debug/libidaten/src/libaten/material/FlakesNormal.o x64/Debug/libidaten/src/libaten/material/disney_brdf.o x64/Debug/libidaten/src/libaten/geometry/sphere.o x64/Debug/libidaten/src/libidaten/cuda/cudaTextureResource.o x64/Debug/libidaten/src/libaten/camera/pinhole.o x64/Debug/libidaten/src/libaten/material/beckman.o x64/Debug/libidaten/src/libaten/material/blinn.o x64/Debug/libidaten/src/libaten/material/ggx.o x64/Debug/libidaten/src/libaten/material/material.o x64/Debug/libidaten/src/libaten/material/oren_nayar.o x64/Debug/libidaten/src/libaten/material/refraction.o x64/Debug/libidaten/src/libaten/material/specular.o x64/Debug/libidaten/src/libidaten/kernel/LBVHBuilder.o x64/Debug/libidaten/src/libidaten/kernel/qbvh.o x64/Debug/libidaten/src/libidaten/kernel/RadixSort.o x64/Debug/libidaten/src/libidaten/kernel/sample_texture_impl.o x64/Debug/libidaten/src/libidaten/kernel/sbvh.o x64/Debug/libidaten/src/libidaten/kernel/Skinning.o x64/Debug/libidaten/src/libidaten/kernel/stackless_bvh.o x64/Debug/libidaten/src/libidaten/kernel/stackless_qbvh.o x64/Debug/libidaten/src/libidaten/svgf/svgf_atrous.o x64/Debug/libidaten/src/libidaten/svgf/svgf_debug.o x64/Debug/libidaten/src/libidaten/svgf/svgf_init.o x64/Debug/libidaten/src/libidaten/svgf/svgf_pt.o x64/Debug/libidaten/src/libidaten/svgf/svgf_ssrt.o x64/Debug/libidaten/src/libidaten/svgf/svgf_tile.o x64/Debug/libidaten/src/libidaten/svgf/svgf_tp.o x64/Debug/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o x64/Debug/libidaten/src/libidaten/kernel/bvh.o x64/Debug/libidaten/src/libidaten/kernel/compaction.o x64/Debug/libidaten/src/libidaten/kernel/intersect.o x64/Debug/libidaten/src/libidaten/kernel/light.o x64/Debug/libidaten/src/libidaten/kernel/material.o x64/Debug/libidaten/src/libidaten/kernel/pathtracing_impl.o x64/Debug/libidaten/gpuCode.o  $(Debug_Implicitly_Linked_Objects)

# Compiles file ../src/libidaten/cuda/cudaGLresource.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libidaten/cuda/cudaGLresource.d
x64/Debug/libidaten/src/libidaten/cuda/cudaGLresource.o: ../src/libidaten/cuda/cudaGLresource.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libidaten/cuda/cudaGLresource.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libidaten/cuda/cudaGLresource.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libidaten/cuda/cudaGLresource.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libidaten/cuda/cudaGLresource.d

# Compiles file ../src/libidaten/cuda/cudamemory.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libidaten/cuda/cudamemory.d
x64/Debug/libidaten/src/libidaten/cuda/cudamemory.o: ../src/libidaten/cuda/cudamemory.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libidaten/cuda/cudamemory.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libidaten/cuda/cudamemory.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libidaten/cuda/cudamemory.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libidaten/cuda/cudamemory.d

# Compiles file ../src/libidaten/kernel/pathtracing.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libidaten/kernel/pathtracing.d
x64/Debug/libidaten/src/libidaten/kernel/pathtracing.o: ../src/libidaten/kernel/pathtracing.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libidaten/kernel/pathtracing.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libidaten/kernel/pathtracing.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libidaten/kernel/pathtracing.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libidaten/kernel/pathtracing.d

# Compiles file ../src/libidaten/kernel/renderer.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libidaten/kernel/renderer.d
x64/Debug/libidaten/src/libidaten/kernel/renderer.o: ../src/libidaten/kernel/renderer.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libidaten/kernel/renderer.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libidaten/kernel/renderer.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libidaten/kernel/renderer.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libidaten/kernel/renderer.d

# Compiles file ../src/libaten/light/arealight.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libaten/light/arealight.d
x64/Debug/libidaten/src/libaten/light/arealight.o: ../src/libaten/light/arealight.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/light/arealight.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libaten/light/arealight.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/light/arealight.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libaten/light/arealight.d

# Compiles file ../src/libaten/light/ibl.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libaten/light/ibl.d
x64/Debug/libidaten/src/libaten/light/ibl.o: ../src/libaten/light/ibl.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/light/ibl.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libaten/light/ibl.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/light/ibl.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libaten/light/ibl.d

# Compiles file ../src/libaten/light/light.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libaten/light/light.d
x64/Debug/libidaten/src/libaten/light/light.o: ../src/libaten/light/light.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/light/light.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libaten/light/light.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/light/light.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libaten/light/light.d

# Compiles file ../src/libaten/material/layer.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libaten/material/layer.d
x64/Debug/libidaten/src/libaten/material/layer.o: ../src/libaten/material/layer.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/layer.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libaten/material/layer.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/layer.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libaten/material/layer.d

# Compiles file ../src/libaten/math/mat4.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libaten/math/mat4.d
x64/Debug/libidaten/src/libaten/math/mat4.o: ../src/libaten/math/mat4.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/math/mat4.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libaten/math/mat4.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/math/mat4.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libaten/math/mat4.d

# Compiles file ../src/libaten/sampler/sobol.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libaten/sampler/sobol.d
x64/Debug/libidaten/src/libaten/sampler/sobol.o: ../src/libaten/sampler/sobol.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/sampler/sobol.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libaten/sampler/sobol.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/sampler/sobol.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libaten/sampler/sobol.d

# Compiles file ../src/libaten/renderer/envmap.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libaten/renderer/envmap.d
x64/Debug/libidaten/src/libaten/renderer/envmap.o: ../src/libaten/renderer/envmap.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/envmap.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libaten/renderer/envmap.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/envmap.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libaten/renderer/envmap.d

# Compiles file ../src/libaten/os/linux/misc/timer_linux.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libaten/os/linux/misc/timer_linux.d
x64/Debug/libidaten/src/libaten/os/linux/misc/timer_linux.o: ../src/libaten/os/linux/misc/timer_linux.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/os/linux/misc/timer_linux.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libaten/os/linux/misc/timer_linux.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/os/linux/misc/timer_linux.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libaten/os/linux/misc/timer_linux.d

# Compiles file ../src/libaten/misc/color.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libaten/misc/color.d
x64/Debug/libidaten/src/libaten/misc/color.o: ../src/libaten/misc/color.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/misc/color.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libaten/misc/color.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/misc/color.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libaten/misc/color.d

# Compiles file ../src/libidaten/svgf/svgf.cpp for the Debug configuration...
-include x64/Debug/libidaten/src/libidaten/svgf/svgf.d
x64/Debug/libidaten/src/libidaten/svgf/svgf.o: ../src/libidaten/svgf/svgf.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libidaten/svgf/svgf.cpp $(Debug_Include_Path) -o x64/Debug/libidaten/src/libidaten/svgf/svgf.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libidaten/svgf/svgf.cpp $(Debug_Include_Path) > x64/Debug/libidaten/src/libidaten/svgf/svgf.d

# Link gpu code files.
x64/Debug/libidaten/gpuCode.o: x64/Debug/libidaten/src/libaten/material/carpaint.o x64/Debug/libidaten/src/libaten/material/FlakesNormal.o x64/Debug/libidaten/src/libaten/material/disney_brdf.o x64/Debug/libidaten/src/libaten/geometry/sphere.o x64/Debug/libidaten/src/libidaten/cuda/cudaTextureResource.o x64/Debug/libidaten/src/libaten/camera/pinhole.o x64/Debug/libidaten/src/libaten/material/beckman.o x64/Debug/libidaten/src/libaten/material/blinn.o x64/Debug/libidaten/src/libaten/material/ggx.o x64/Debug/libidaten/src/libaten/material/material.o x64/Debug/libidaten/src/libaten/material/oren_nayar.o x64/Debug/libidaten/src/libaten/material/refraction.o x64/Debug/libidaten/src/libaten/material/specular.o x64/Debug/libidaten/src/libidaten/kernel/LBVHBuilder.o x64/Debug/libidaten/src/libidaten/kernel/qbvh.o x64/Debug/libidaten/src/libidaten/kernel/RadixSort.o x64/Debug/libidaten/src/libidaten/kernel/sample_texture_impl.o x64/Debug/libidaten/src/libidaten/kernel/sbvh.o x64/Debug/libidaten/src/libidaten/kernel/Skinning.o x64/Debug/libidaten/src/libidaten/kernel/stackless_bvh.o x64/Debug/libidaten/src/libidaten/kernel/stackless_qbvh.o x64/Debug/libidaten/src/libidaten/svgf/svgf_atrous.o x64/Debug/libidaten/src/libidaten/svgf/svgf_debug.o x64/Debug/libidaten/src/libidaten/svgf/svgf_init.o x64/Debug/libidaten/src/libidaten/svgf/svgf_pt.o x64/Debug/libidaten/src/libidaten/svgf/svgf_ssrt.o x64/Debug/libidaten/src/libidaten/svgf/svgf_tile.o x64/Debug/libidaten/src/libidaten/svgf/svgf_tp.o x64/Debug/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o x64/Debug/libidaten/src/libidaten/kernel/bvh.o x64/Debug/libidaten/src/libidaten/kernel/compaction.o x64/Debug/libidaten/src/libidaten/kernel/intersect.o x64/Debug/libidaten/src/libidaten/kernel/light.o x64/Debug/libidaten/src/libidaten/kernel/material.o x64/Debug/libidaten/src/libidaten/kernel/pathtracing_impl.o 
	$(NVCC) -arch=sm_60 -dlink x64/Debug/libidaten/src/libaten/material/carpaint.o x64/Debug/libidaten/src/libaten/material/FlakesNormal.o x64/Debug/libidaten/src/libaten/material/disney_brdf.o x64/Debug/libidaten/src/libaten/geometry/sphere.o x64/Debug/libidaten/src/libidaten/cuda/cudaTextureResource.o x64/Debug/libidaten/src/libaten/camera/pinhole.o x64/Debug/libidaten/src/libaten/material/beckman.o x64/Debug/libidaten/src/libaten/material/blinn.o x64/Debug/libidaten/src/libaten/material/ggx.o x64/Debug/libidaten/src/libaten/material/material.o x64/Debug/libidaten/src/libaten/material/oren_nayar.o x64/Debug/libidaten/src/libaten/material/refraction.o x64/Debug/libidaten/src/libaten/material/specular.o x64/Debug/libidaten/src/libidaten/kernel/LBVHBuilder.o x64/Debug/libidaten/src/libidaten/kernel/qbvh.o x64/Debug/libidaten/src/libidaten/kernel/RadixSort.o x64/Debug/libidaten/src/libidaten/kernel/sample_texture_impl.o x64/Debug/libidaten/src/libidaten/kernel/sbvh.o x64/Debug/libidaten/src/libidaten/kernel/Skinning.o x64/Debug/libidaten/src/libidaten/kernel/stackless_bvh.o x64/Debug/libidaten/src/libidaten/kernel/stackless_qbvh.o x64/Debug/libidaten/src/libidaten/svgf/svgf_atrous.o x64/Debug/libidaten/src/libidaten/svgf/svgf_debug.o x64/Debug/libidaten/src/libidaten/svgf/svgf_init.o x64/Debug/libidaten/src/libidaten/svgf/svgf_pt.o x64/Debug/libidaten/src/libidaten/svgf/svgf_ssrt.o x64/Debug/libidaten/src/libidaten/svgf/svgf_tile.o x64/Debug/libidaten/src/libidaten/svgf/svgf_tp.o x64/Debug/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o x64/Debug/libidaten/src/libidaten/kernel/bvh.o x64/Debug/libidaten/src/libidaten/kernel/compaction.o x64/Debug/libidaten/src/libidaten/kernel/intersect.o x64/Debug/libidaten/src/libidaten/kernel/light.o x64/Debug/libidaten/src/libidaten/kernel/material.o x64/Debug/libidaten/src/libidaten/kernel/pathtracing_impl.o  -o x64/Debug/libidaten/gpuCode.o

# Compiles file ../src/libaten/material/carpaint.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/material/carpaint.o: ../src/libaten/material/carpaint.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/material/carpaint.cpp -o x64/Debug/libidaten/src/libaten/material/carpaint.o

# Compiles file ../src/libaten/material/FlakesNormal.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/material/FlakesNormal.o: ../src/libaten/material/FlakesNormal.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/material/FlakesNormal.cpp -o x64/Debug/libidaten/src/libaten/material/FlakesNormal.o

# Compiles file ../src/libaten/material/disney_brdf.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/material/disney_brdf.o: ../src/libaten/material/disney_brdf.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/material/disney_brdf.cpp -o x64/Debug/libidaten/src/libaten/material/disney_brdf.o

# Compiles file ../src/libaten/geometry/sphere.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/geometry/sphere.o: ../src/libaten/geometry/sphere.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/geometry/sphere.cpp -o x64/Debug/libidaten/src/libaten/geometry/sphere.o

# Compiles file ../src/libidaten/cuda/cudaTextureResource.cpp for the Debug configuration...
x64/Debug/libidaten/src/libidaten/cuda/cudaTextureResource.o: ../src/libidaten/cuda/cudaTextureResource.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/cuda/cudaTextureResource.cpp -o x64/Debug/libidaten/src/libidaten/cuda/cudaTextureResource.o

# Compiles file ../src/libaten/camera/pinhole.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/camera/pinhole.o: ../src/libaten/camera/pinhole.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/camera/pinhole.cpp -o x64/Debug/libidaten/src/libaten/camera/pinhole.o

# Compiles file ../src/libaten/material/beckman.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/material/beckman.o: ../src/libaten/material/beckman.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/material/beckman.cpp -o x64/Debug/libidaten/src/libaten/material/beckman.o

# Compiles file ../src/libaten/material/blinn.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/material/blinn.o: ../src/libaten/material/blinn.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/material/blinn.cpp -o x64/Debug/libidaten/src/libaten/material/blinn.o

# Compiles file ../src/libaten/material/ggx.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/material/ggx.o: ../src/libaten/material/ggx.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/material/ggx.cpp -o x64/Debug/libidaten/src/libaten/material/ggx.o

# Compiles file ../src/libaten/material/material.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/material/material.o: ../src/libaten/material/material.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/material/material.cpp -o x64/Debug/libidaten/src/libaten/material/material.o

# Compiles file ../src/libaten/material/oren_nayar.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/material/oren_nayar.o: ../src/libaten/material/oren_nayar.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/material/oren_nayar.cpp -o x64/Debug/libidaten/src/libaten/material/oren_nayar.o

# Compiles file ../src/libaten/material/refraction.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/material/refraction.o: ../src/libaten/material/refraction.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/material/refraction.cpp -o x64/Debug/libidaten/src/libaten/material/refraction.o

# Compiles file ../src/libaten/material/specular.cpp for the Debug configuration...
x64/Debug/libidaten/src/libaten/material/specular.o: ../src/libaten/material/specular.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libaten/material/specular.cpp -o x64/Debug/libidaten/src/libaten/material/specular.o

# Compiles file ../src/libidaten/kernel/LBVHBuilder.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/LBVHBuilder.o: ../src/libidaten/kernel/LBVHBuilder.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/LBVHBuilder.cu -o x64/Debug/libidaten/src/libidaten/kernel/LBVHBuilder.o

# Compiles file ../src/libidaten/kernel/qbvh.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/qbvh.o: ../src/libidaten/kernel/qbvh.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/qbvh.cu -o x64/Debug/libidaten/src/libidaten/kernel/qbvh.o

# Compiles file ../src/libidaten/kernel/RadixSort.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/RadixSort.o: ../src/libidaten/kernel/RadixSort.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/RadixSort.cu -o x64/Debug/libidaten/src/libidaten/kernel/RadixSort.o

# Compiles file ../src/libidaten/kernel/sample_texture_impl.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/sample_texture_impl.o: ../src/libidaten/kernel/sample_texture_impl.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/sample_texture_impl.cu -o x64/Debug/libidaten/src/libidaten/kernel/sample_texture_impl.o

# Compiles file ../src/libidaten/kernel/sbvh.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/sbvh.o: ../src/libidaten/kernel/sbvh.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/sbvh.cu -o x64/Debug/libidaten/src/libidaten/kernel/sbvh.o

# Compiles file ../src/libidaten/kernel/Skinning.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/Skinning.o: ../src/libidaten/kernel/Skinning.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/Skinning.cu -o x64/Debug/libidaten/src/libidaten/kernel/Skinning.o

# Compiles file ../src/libidaten/kernel/stackless_bvh.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/stackless_bvh.o: ../src/libidaten/kernel/stackless_bvh.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/stackless_bvh.cu -o x64/Debug/libidaten/src/libidaten/kernel/stackless_bvh.o

# Compiles file ../src/libidaten/kernel/stackless_qbvh.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/stackless_qbvh.o: ../src/libidaten/kernel/stackless_qbvh.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/stackless_qbvh.cu -o x64/Debug/libidaten/src/libidaten/kernel/stackless_qbvh.o

# Compiles file ../src/libidaten/svgf/svgf_atrous.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/svgf/svgf_atrous.o: ../src/libidaten/svgf/svgf_atrous.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_atrous.cu -o x64/Debug/libidaten/src/libidaten/svgf/svgf_atrous.o

# Compiles file ../src/libidaten/svgf/svgf_debug.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/svgf/svgf_debug.o: ../src/libidaten/svgf/svgf_debug.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_debug.cu -o x64/Debug/libidaten/src/libidaten/svgf/svgf_debug.o

# Compiles file ../src/libidaten/svgf/svgf_init.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/svgf/svgf_init.o: ../src/libidaten/svgf/svgf_init.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_init.cu -o x64/Debug/libidaten/src/libidaten/svgf/svgf_init.o

# Compiles file ../src/libidaten/svgf/svgf_pt.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/svgf/svgf_pt.o: ../src/libidaten/svgf/svgf_pt.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_pt.cu -o x64/Debug/libidaten/src/libidaten/svgf/svgf_pt.o

# Compiles file ../src/libidaten/svgf/svgf_ssrt.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/svgf/svgf_ssrt.o: ../src/libidaten/svgf/svgf_ssrt.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_ssrt.cu -o x64/Debug/libidaten/src/libidaten/svgf/svgf_ssrt.o

# Compiles file ../src/libidaten/svgf/svgf_tile.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/svgf/svgf_tile.o: ../src/libidaten/svgf/svgf_tile.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_tile.cu -o x64/Debug/libidaten/src/libidaten/svgf/svgf_tile.o

# Compiles file ../src/libidaten/svgf/svgf_tp.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/svgf/svgf_tp.o: ../src/libidaten/svgf/svgf_tp.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_tp.cu -o x64/Debug/libidaten/src/libidaten/svgf/svgf_tp.o

# Compiles file ../src/libidaten/svgf/svgf_VarianceEstimation.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o: ../src/libidaten/svgf/svgf_VarianceEstimation.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_VarianceEstimation.cu -o x64/Debug/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o

# Compiles file ../src/libidaten/kernel/bvh.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/bvh.o: ../src/libidaten/kernel/bvh.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/bvh.cu -o x64/Debug/libidaten/src/libidaten/kernel/bvh.o

# Compiles file ../src/libidaten/kernel/compaction.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/compaction.o: ../src/libidaten/kernel/compaction.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/compaction.cu -o x64/Debug/libidaten/src/libidaten/kernel/compaction.o

# Compiles file ../src/libidaten/kernel/intersect.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/intersect.o: ../src/libidaten/kernel/intersect.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/intersect.cu -o x64/Debug/libidaten/src/libidaten/kernel/intersect.o

# Compiles file ../src/libidaten/kernel/light.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/light.o: ../src/libidaten/kernel/light.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 --expt-extended-lambda  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/light.cu -o x64/Debug/libidaten/src/libidaten/kernel/light.o

# Compiles file ../src/libidaten/kernel/material.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/material.o: ../src/libidaten/kernel/material.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/material.cu -o x64/Debug/libidaten/src/libidaten/kernel/material.o

# Compiles file ../src/libidaten/kernel/pathtracing_impl.cu for the Debug configuration...
x64/Debug/libidaten/src/libidaten/kernel/pathtracing_impl.o: ../src/libidaten/kernel/pathtracing_impl.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 --expt-extended-lambda  -Xcompiler ,-O0 -g -std=c++11  $(Debug_Include_Path) $(Debug_Preprocessor_Definitions) -c ../src/libidaten/kernel/pathtracing_impl.cu -o x64/Debug/libidaten/src/libidaten/kernel/pathtracing_impl.o

# Builds the Release configuration...
.PHONY: Release
Release: create_folders x64/Release/libidaten/src/libidaten/cuda/cudaGLresource.o x64/Release/libidaten/src/libidaten/cuda/cudamemory.o x64/Release/libidaten/src/libidaten/kernel/pathtracing.o x64/Release/libidaten/src/libidaten/kernel/renderer.o x64/Release/libidaten/src/libaten/light/arealight.o x64/Release/libidaten/src/libaten/light/ibl.o x64/Release/libidaten/src/libaten/light/light.o x64/Release/libidaten/src/libaten/material/layer.o x64/Release/libidaten/src/libaten/math/mat4.o x64/Release/libidaten/src/libaten/sampler/sobol.o x64/Release/libidaten/src/libaten/renderer/envmap.o x64/Release/libidaten/src/libaten/os/linux/misc/timer_linux.o x64/Release/libidaten/src/libaten/misc/color.o x64/Release/libidaten/src/libidaten/svgf/svgf.o x64/Release/libidaten/src/libaten/material/carpaint.o x64/Release/libidaten/src/libaten/material/FlakesNormal.o x64/Release/libidaten/src/libaten/material/disney_brdf.o x64/Release/libidaten/src/libaten/geometry/sphere.o x64/Release/libidaten/src/libidaten/cuda/cudaTextureResource.o x64/Release/libidaten/src/libaten/camera/pinhole.o x64/Release/libidaten/src/libaten/material/beckman.o x64/Release/libidaten/src/libaten/material/blinn.o x64/Release/libidaten/src/libaten/material/ggx.o x64/Release/libidaten/src/libaten/material/material.o x64/Release/libidaten/src/libaten/material/oren_nayar.o x64/Release/libidaten/src/libaten/material/refraction.o x64/Release/libidaten/src/libaten/material/specular.o x64/Release/libidaten/src/libidaten/kernel/LBVHBuilder.o x64/Release/libidaten/src/libidaten/kernel/qbvh.o x64/Release/libidaten/src/libidaten/kernel/RadixSort.o x64/Release/libidaten/src/libidaten/kernel/sample_texture_impl.o x64/Release/libidaten/src/libidaten/kernel/sbvh.o x64/Release/libidaten/src/libidaten/kernel/Skinning.o x64/Release/libidaten/src/libidaten/kernel/stackless_bvh.o x64/Release/libidaten/src/libidaten/kernel/stackless_qbvh.o x64/Release/libidaten/src/libidaten/svgf/svgf_atrous.o x64/Release/libidaten/src/libidaten/svgf/svgf_debug.o x64/Release/libidaten/src/libidaten/svgf/svgf_init.o x64/Release/libidaten/src/libidaten/svgf/svgf_pt.o x64/Release/libidaten/src/libidaten/svgf/svgf_ssrt.o x64/Release/libidaten/src/libidaten/svgf/svgf_tile.o x64/Release/libidaten/src/libidaten/svgf/svgf_tp.o x64/Release/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o x64/Release/libidaten/src/libidaten/kernel/bvh.o x64/Release/libidaten/src/libidaten/kernel/compaction.o x64/Release/libidaten/src/libidaten/kernel/intersect.o x64/Release/libidaten/src/libidaten/kernel/light.o x64/Release/libidaten/src/libidaten/kernel/material.o x64/Release/libidaten/src/libidaten/kernel/pathtracing_impl.o x64/Release/libidaten/gpuCode.o 
	ar rcs x64/Release/libidaten.a x64/Release/libidaten/src/libidaten/cuda/cudaGLresource.o x64/Release/libidaten/src/libidaten/cuda/cudamemory.o x64/Release/libidaten/src/libidaten/kernel/pathtracing.o x64/Release/libidaten/src/libidaten/kernel/renderer.o x64/Release/libidaten/src/libaten/light/arealight.o x64/Release/libidaten/src/libaten/light/ibl.o x64/Release/libidaten/src/libaten/light/light.o x64/Release/libidaten/src/libaten/material/layer.o x64/Release/libidaten/src/libaten/math/mat4.o x64/Release/libidaten/src/libaten/sampler/sobol.o x64/Release/libidaten/src/libaten/renderer/envmap.o x64/Release/libidaten/src/libaten/os/linux/misc/timer_linux.o x64/Release/libidaten/src/libaten/misc/color.o x64/Release/libidaten/src/libidaten/svgf/svgf.o x64/Release/libidaten/src/libaten/material/carpaint.o x64/Release/libidaten/src/libaten/material/FlakesNormal.o x64/Release/libidaten/src/libaten/material/disney_brdf.o x64/Release/libidaten/src/libaten/geometry/sphere.o x64/Release/libidaten/src/libidaten/cuda/cudaTextureResource.o x64/Release/libidaten/src/libaten/camera/pinhole.o x64/Release/libidaten/src/libaten/material/beckman.o x64/Release/libidaten/src/libaten/material/blinn.o x64/Release/libidaten/src/libaten/material/ggx.o x64/Release/libidaten/src/libaten/material/material.o x64/Release/libidaten/src/libaten/material/oren_nayar.o x64/Release/libidaten/src/libaten/material/refraction.o x64/Release/libidaten/src/libaten/material/specular.o x64/Release/libidaten/src/libidaten/kernel/LBVHBuilder.o x64/Release/libidaten/src/libidaten/kernel/qbvh.o x64/Release/libidaten/src/libidaten/kernel/RadixSort.o x64/Release/libidaten/src/libidaten/kernel/sample_texture_impl.o x64/Release/libidaten/src/libidaten/kernel/sbvh.o x64/Release/libidaten/src/libidaten/kernel/Skinning.o x64/Release/libidaten/src/libidaten/kernel/stackless_bvh.o x64/Release/libidaten/src/libidaten/kernel/stackless_qbvh.o x64/Release/libidaten/src/libidaten/svgf/svgf_atrous.o x64/Release/libidaten/src/libidaten/svgf/svgf_debug.o x64/Release/libidaten/src/libidaten/svgf/svgf_init.o x64/Release/libidaten/src/libidaten/svgf/svgf_pt.o x64/Release/libidaten/src/libidaten/svgf/svgf_ssrt.o x64/Release/libidaten/src/libidaten/svgf/svgf_tile.o x64/Release/libidaten/src/libidaten/svgf/svgf_tp.o x64/Release/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o x64/Release/libidaten/src/libidaten/kernel/bvh.o x64/Release/libidaten/src/libidaten/kernel/compaction.o x64/Release/libidaten/src/libidaten/kernel/intersect.o x64/Release/libidaten/src/libidaten/kernel/light.o x64/Release/libidaten/src/libidaten/kernel/material.o x64/Release/libidaten/src/libidaten/kernel/pathtracing_impl.o x64/Release/libidaten/gpuCode.o  $(Release_Implicitly_Linked_Objects)

# Compiles file ../src/libidaten/cuda/cudaGLresource.cpp for the Release configuration...
-include x64/Release/libidaten/src/libidaten/cuda/cudaGLresource.d
x64/Release/libidaten/src/libidaten/cuda/cudaGLresource.o: ../src/libidaten/cuda/cudaGLresource.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libidaten/cuda/cudaGLresource.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libidaten/cuda/cudaGLresource.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libidaten/cuda/cudaGLresource.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libidaten/cuda/cudaGLresource.d

# Compiles file ../src/libidaten/cuda/cudamemory.cpp for the Release configuration...
-include x64/Release/libidaten/src/libidaten/cuda/cudamemory.d
x64/Release/libidaten/src/libidaten/cuda/cudamemory.o: ../src/libidaten/cuda/cudamemory.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libidaten/cuda/cudamemory.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libidaten/cuda/cudamemory.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libidaten/cuda/cudamemory.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libidaten/cuda/cudamemory.d

# Compiles file ../src/libidaten/kernel/pathtracing.cpp for the Release configuration...
-include x64/Release/libidaten/src/libidaten/kernel/pathtracing.d
x64/Release/libidaten/src/libidaten/kernel/pathtracing.o: ../src/libidaten/kernel/pathtracing.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libidaten/kernel/pathtracing.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libidaten/kernel/pathtracing.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libidaten/kernel/pathtracing.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libidaten/kernel/pathtracing.d

# Compiles file ../src/libidaten/kernel/renderer.cpp for the Release configuration...
-include x64/Release/libidaten/src/libidaten/kernel/renderer.d
x64/Release/libidaten/src/libidaten/kernel/renderer.o: ../src/libidaten/kernel/renderer.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libidaten/kernel/renderer.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libidaten/kernel/renderer.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libidaten/kernel/renderer.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libidaten/kernel/renderer.d

# Compiles file ../src/libaten/light/arealight.cpp for the Release configuration...
-include x64/Release/libidaten/src/libaten/light/arealight.d
x64/Release/libidaten/src/libaten/light/arealight.o: ../src/libaten/light/arealight.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/light/arealight.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libaten/light/arealight.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/light/arealight.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libaten/light/arealight.d

# Compiles file ../src/libaten/light/ibl.cpp for the Release configuration...
-include x64/Release/libidaten/src/libaten/light/ibl.d
x64/Release/libidaten/src/libaten/light/ibl.o: ../src/libaten/light/ibl.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/light/ibl.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libaten/light/ibl.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/light/ibl.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libaten/light/ibl.d

# Compiles file ../src/libaten/light/light.cpp for the Release configuration...
-include x64/Release/libidaten/src/libaten/light/light.d
x64/Release/libidaten/src/libaten/light/light.o: ../src/libaten/light/light.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/light/light.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libaten/light/light.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/light/light.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libaten/light/light.d

# Compiles file ../src/libaten/material/layer.cpp for the Release configuration...
-include x64/Release/libidaten/src/libaten/material/layer.d
x64/Release/libidaten/src/libaten/material/layer.o: ../src/libaten/material/layer.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/layer.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libaten/material/layer.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/layer.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libaten/material/layer.d

# Compiles file ../src/libaten/math/mat4.cpp for the Release configuration...
-include x64/Release/libidaten/src/libaten/math/mat4.d
x64/Release/libidaten/src/libaten/math/mat4.o: ../src/libaten/math/mat4.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/math/mat4.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libaten/math/mat4.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/math/mat4.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libaten/math/mat4.d

# Compiles file ../src/libaten/sampler/sobol.cpp for the Release configuration...
-include x64/Release/libidaten/src/libaten/sampler/sobol.d
x64/Release/libidaten/src/libaten/sampler/sobol.o: ../src/libaten/sampler/sobol.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/sampler/sobol.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libaten/sampler/sobol.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/sampler/sobol.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libaten/sampler/sobol.d

# Compiles file ../src/libaten/renderer/envmap.cpp for the Release configuration...
-include x64/Release/libidaten/src/libaten/renderer/envmap.d
x64/Release/libidaten/src/libaten/renderer/envmap.o: ../src/libaten/renderer/envmap.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/envmap.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libaten/renderer/envmap.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/envmap.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libaten/renderer/envmap.d

# Compiles file ../src/libaten/os/linux/misc/timer_linux.cpp for the Release configuration...
-include x64/Release/libidaten/src/libaten/os/linux/misc/timer_linux.d
x64/Release/libidaten/src/libaten/os/linux/misc/timer_linux.o: ../src/libaten/os/linux/misc/timer_linux.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/os/linux/misc/timer_linux.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libaten/os/linux/misc/timer_linux.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/os/linux/misc/timer_linux.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libaten/os/linux/misc/timer_linux.d

# Compiles file ../src/libaten/misc/color.cpp for the Release configuration...
-include x64/Release/libidaten/src/libaten/misc/color.d
x64/Release/libidaten/src/libaten/misc/color.o: ../src/libaten/misc/color.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/misc/color.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libaten/misc/color.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/misc/color.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libaten/misc/color.d

# Compiles file ../src/libidaten/svgf/svgf.cpp for the Release configuration...
-include x64/Release/libidaten/src/libidaten/svgf/svgf.d
x64/Release/libidaten/src/libidaten/svgf/svgf.o: ../src/libidaten/svgf/svgf.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libidaten/svgf/svgf.cpp $(Release_Include_Path) -o x64/Release/libidaten/src/libidaten/svgf/svgf.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libidaten/svgf/svgf.cpp $(Release_Include_Path) > x64/Release/libidaten/src/libidaten/svgf/svgf.d

# Link gpu code files.
x64/Release/libidaten/gpuCode.o: x64/Release/libidaten/src/libaten/material/carpaint.o x64/Release/libidaten/src/libaten/material/FlakesNormal.o x64/Release/libidaten/src/libaten/material/disney_brdf.o x64/Release/libidaten/src/libaten/geometry/sphere.o x64/Release/libidaten/src/libidaten/cuda/cudaTextureResource.o x64/Release/libidaten/src/libaten/camera/pinhole.o x64/Release/libidaten/src/libaten/material/beckman.o x64/Release/libidaten/src/libaten/material/blinn.o x64/Release/libidaten/src/libaten/material/ggx.o x64/Release/libidaten/src/libaten/material/material.o x64/Release/libidaten/src/libaten/material/oren_nayar.o x64/Release/libidaten/src/libaten/material/refraction.o x64/Release/libidaten/src/libaten/material/specular.o x64/Release/libidaten/src/libidaten/kernel/LBVHBuilder.o x64/Release/libidaten/src/libidaten/kernel/qbvh.o x64/Release/libidaten/src/libidaten/kernel/RadixSort.o x64/Release/libidaten/src/libidaten/kernel/sample_texture_impl.o x64/Release/libidaten/src/libidaten/kernel/sbvh.o x64/Release/libidaten/src/libidaten/kernel/Skinning.o x64/Release/libidaten/src/libidaten/kernel/stackless_bvh.o x64/Release/libidaten/src/libidaten/kernel/stackless_qbvh.o x64/Release/libidaten/src/libidaten/svgf/svgf_atrous.o x64/Release/libidaten/src/libidaten/svgf/svgf_debug.o x64/Release/libidaten/src/libidaten/svgf/svgf_init.o x64/Release/libidaten/src/libidaten/svgf/svgf_pt.o x64/Release/libidaten/src/libidaten/svgf/svgf_ssrt.o x64/Release/libidaten/src/libidaten/svgf/svgf_tile.o x64/Release/libidaten/src/libidaten/svgf/svgf_tp.o x64/Release/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o x64/Release/libidaten/src/libidaten/kernel/bvh.o x64/Release/libidaten/src/libidaten/kernel/compaction.o x64/Release/libidaten/src/libidaten/kernel/intersect.o x64/Release/libidaten/src/libidaten/kernel/light.o x64/Release/libidaten/src/libidaten/kernel/material.o x64/Release/libidaten/src/libidaten/kernel/pathtracing_impl.o 
	$(NVCC) -arch=sm_60 -dlink x64/Release/libidaten/src/libaten/material/carpaint.o x64/Release/libidaten/src/libaten/material/FlakesNormal.o x64/Release/libidaten/src/libaten/material/disney_brdf.o x64/Release/libidaten/src/libaten/geometry/sphere.o x64/Release/libidaten/src/libidaten/cuda/cudaTextureResource.o x64/Release/libidaten/src/libaten/camera/pinhole.o x64/Release/libidaten/src/libaten/material/beckman.o x64/Release/libidaten/src/libaten/material/blinn.o x64/Release/libidaten/src/libaten/material/ggx.o x64/Release/libidaten/src/libaten/material/material.o x64/Release/libidaten/src/libaten/material/oren_nayar.o x64/Release/libidaten/src/libaten/material/refraction.o x64/Release/libidaten/src/libaten/material/specular.o x64/Release/libidaten/src/libidaten/kernel/LBVHBuilder.o x64/Release/libidaten/src/libidaten/kernel/qbvh.o x64/Release/libidaten/src/libidaten/kernel/RadixSort.o x64/Release/libidaten/src/libidaten/kernel/sample_texture_impl.o x64/Release/libidaten/src/libidaten/kernel/sbvh.o x64/Release/libidaten/src/libidaten/kernel/Skinning.o x64/Release/libidaten/src/libidaten/kernel/stackless_bvh.o x64/Release/libidaten/src/libidaten/kernel/stackless_qbvh.o x64/Release/libidaten/src/libidaten/svgf/svgf_atrous.o x64/Release/libidaten/src/libidaten/svgf/svgf_debug.o x64/Release/libidaten/src/libidaten/svgf/svgf_init.o x64/Release/libidaten/src/libidaten/svgf/svgf_pt.o x64/Release/libidaten/src/libidaten/svgf/svgf_ssrt.o x64/Release/libidaten/src/libidaten/svgf/svgf_tile.o x64/Release/libidaten/src/libidaten/svgf/svgf_tp.o x64/Release/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o x64/Release/libidaten/src/libidaten/kernel/bvh.o x64/Release/libidaten/src/libidaten/kernel/compaction.o x64/Release/libidaten/src/libidaten/kernel/intersect.o x64/Release/libidaten/src/libidaten/kernel/light.o x64/Release/libidaten/src/libidaten/kernel/material.o x64/Release/libidaten/src/libidaten/kernel/pathtracing_impl.o  -o x64/Release/libidaten/gpuCode.o

# Compiles file ../src/libaten/material/carpaint.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/material/carpaint.o: ../src/libaten/material/carpaint.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/material/carpaint.cpp -o x64/Release/libidaten/src/libaten/material/carpaint.o

# Compiles file ../src/libaten/material/FlakesNormal.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/material/FlakesNormal.o: ../src/libaten/material/FlakesNormal.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/material/FlakesNormal.cpp -o x64/Release/libidaten/src/libaten/material/FlakesNormal.o

# Compiles file ../src/libaten/material/disney_brdf.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/material/disney_brdf.o: ../src/libaten/material/disney_brdf.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/material/disney_brdf.cpp -o x64/Release/libidaten/src/libaten/material/disney_brdf.o

# Compiles file ../src/libaten/geometry/sphere.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/geometry/sphere.o: ../src/libaten/geometry/sphere.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/geometry/sphere.cpp -o x64/Release/libidaten/src/libaten/geometry/sphere.o

# Compiles file ../src/libidaten/cuda/cudaTextureResource.cpp for the Release configuration...
x64/Release/libidaten/src/libidaten/cuda/cudaTextureResource.o: ../src/libidaten/cuda/cudaTextureResource.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/cuda/cudaTextureResource.cpp -o x64/Release/libidaten/src/libidaten/cuda/cudaTextureResource.o

# Compiles file ../src/libaten/camera/pinhole.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/camera/pinhole.o: ../src/libaten/camera/pinhole.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/camera/pinhole.cpp -o x64/Release/libidaten/src/libaten/camera/pinhole.o

# Compiles file ../src/libaten/material/beckman.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/material/beckman.o: ../src/libaten/material/beckman.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/material/beckman.cpp -o x64/Release/libidaten/src/libaten/material/beckman.o

# Compiles file ../src/libaten/material/blinn.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/material/blinn.o: ../src/libaten/material/blinn.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/material/blinn.cpp -o x64/Release/libidaten/src/libaten/material/blinn.o

# Compiles file ../src/libaten/material/ggx.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/material/ggx.o: ../src/libaten/material/ggx.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/material/ggx.cpp -o x64/Release/libidaten/src/libaten/material/ggx.o

# Compiles file ../src/libaten/material/material.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/material/material.o: ../src/libaten/material/material.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/material/material.cpp -o x64/Release/libidaten/src/libaten/material/material.o

# Compiles file ../src/libaten/material/oren_nayar.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/material/oren_nayar.o: ../src/libaten/material/oren_nayar.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/material/oren_nayar.cpp -o x64/Release/libidaten/src/libaten/material/oren_nayar.o

# Compiles file ../src/libaten/material/refraction.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/material/refraction.o: ../src/libaten/material/refraction.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/material/refraction.cpp -o x64/Release/libidaten/src/libaten/material/refraction.o

# Compiles file ../src/libaten/material/specular.cpp for the Release configuration...
x64/Release/libidaten/src/libaten/material/specular.o: ../src/libaten/material/specular.cpp
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 -x cu  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libaten/material/specular.cpp -o x64/Release/libidaten/src/libaten/material/specular.o

# Compiles file ../src/libidaten/kernel/LBVHBuilder.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/LBVHBuilder.o: ../src/libidaten/kernel/LBVHBuilder.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/LBVHBuilder.cu -o x64/Release/libidaten/src/libidaten/kernel/LBVHBuilder.o

# Compiles file ../src/libidaten/kernel/qbvh.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/qbvh.o: ../src/libidaten/kernel/qbvh.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/qbvh.cu -o x64/Release/libidaten/src/libidaten/kernel/qbvh.o

# Compiles file ../src/libidaten/kernel/RadixSort.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/RadixSort.o: ../src/libidaten/kernel/RadixSort.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/RadixSort.cu -o x64/Release/libidaten/src/libidaten/kernel/RadixSort.o

# Compiles file ../src/libidaten/kernel/sample_texture_impl.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/sample_texture_impl.o: ../src/libidaten/kernel/sample_texture_impl.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/sample_texture_impl.cu -o x64/Release/libidaten/src/libidaten/kernel/sample_texture_impl.o

# Compiles file ../src/libidaten/kernel/sbvh.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/sbvh.o: ../src/libidaten/kernel/sbvh.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/sbvh.cu -o x64/Release/libidaten/src/libidaten/kernel/sbvh.o

# Compiles file ../src/libidaten/kernel/Skinning.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/Skinning.o: ../src/libidaten/kernel/Skinning.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/Skinning.cu -o x64/Release/libidaten/src/libidaten/kernel/Skinning.o

# Compiles file ../src/libidaten/kernel/stackless_bvh.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/stackless_bvh.o: ../src/libidaten/kernel/stackless_bvh.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/stackless_bvh.cu -o x64/Release/libidaten/src/libidaten/kernel/stackless_bvh.o

# Compiles file ../src/libidaten/kernel/stackless_qbvh.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/stackless_qbvh.o: ../src/libidaten/kernel/stackless_qbvh.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/stackless_qbvh.cu -o x64/Release/libidaten/src/libidaten/kernel/stackless_qbvh.o

# Compiles file ../src/libidaten/svgf/svgf_atrous.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/svgf/svgf_atrous.o: ../src/libidaten/svgf/svgf_atrous.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_atrous.cu -o x64/Release/libidaten/src/libidaten/svgf/svgf_atrous.o

# Compiles file ../src/libidaten/svgf/svgf_debug.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/svgf/svgf_debug.o: ../src/libidaten/svgf/svgf_debug.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_debug.cu -o x64/Release/libidaten/src/libidaten/svgf/svgf_debug.o

# Compiles file ../src/libidaten/svgf/svgf_init.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/svgf/svgf_init.o: ../src/libidaten/svgf/svgf_init.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_init.cu -o x64/Release/libidaten/src/libidaten/svgf/svgf_init.o

# Compiles file ../src/libidaten/svgf/svgf_pt.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/svgf/svgf_pt.o: ../src/libidaten/svgf/svgf_pt.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_pt.cu -o x64/Release/libidaten/src/libidaten/svgf/svgf_pt.o

# Compiles file ../src/libidaten/svgf/svgf_ssrt.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/svgf/svgf_ssrt.o: ../src/libidaten/svgf/svgf_ssrt.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_ssrt.cu -o x64/Release/libidaten/src/libidaten/svgf/svgf_ssrt.o

# Compiles file ../src/libidaten/svgf/svgf_tile.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/svgf/svgf_tile.o: ../src/libidaten/svgf/svgf_tile.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_tile.cu -o x64/Release/libidaten/src/libidaten/svgf/svgf_tile.o

# Compiles file ../src/libidaten/svgf/svgf_tp.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/svgf/svgf_tp.o: ../src/libidaten/svgf/svgf_tp.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_tp.cu -o x64/Release/libidaten/src/libidaten/svgf/svgf_tp.o

# Compiles file ../src/libidaten/svgf/svgf_VarianceEstimation.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o: ../src/libidaten/svgf/svgf_VarianceEstimation.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/svgf/svgf_VarianceEstimation.cu -o x64/Release/libidaten/src/libidaten/svgf/svgf_VarianceEstimation.o

# Compiles file ../src/libidaten/kernel/bvh.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/bvh.o: ../src/libidaten/kernel/bvh.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/bvh.cu -o x64/Release/libidaten/src/libidaten/kernel/bvh.o

# Compiles file ../src/libidaten/kernel/compaction.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/compaction.o: ../src/libidaten/kernel/compaction.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/compaction.cu -o x64/Release/libidaten/src/libidaten/kernel/compaction.o

# Compiles file ../src/libidaten/kernel/intersect.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/intersect.o: ../src/libidaten/kernel/intersect.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/intersect.cu -o x64/Release/libidaten/src/libidaten/kernel/intersect.o

# Compiles file ../src/libidaten/kernel/light.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/light.o: ../src/libidaten/kernel/light.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 --expt-extended-lambda  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/light.cu -o x64/Release/libidaten/src/libidaten/kernel/light.o

# Compiles file ../src/libidaten/kernel/material.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/material.o: ../src/libidaten/kernel/material.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/material.cu -o x64/Release/libidaten/src/libidaten/kernel/material.o

# Compiles file ../src/libidaten/kernel/pathtracing_impl.cu for the Release configuration...
x64/Release/libidaten/src/libidaten/kernel/pathtracing_impl.o: ../src/libidaten/kernel/pathtracing_impl.cu
	$(NVCC)  -rdc=true --compile -cudart static --machine 64 -use_fast_math -gencode arch=compute_60,code=sm_60 --expt-extended-lambda  -Xcompiler ,-O2 -std=c++11  $(Release_Include_Path) $(Release_Preprocessor_Definitions) -c ../src/libidaten/kernel/pathtracing_impl.cu -o x64/Release/libidaten/src/libidaten/kernel/pathtracing_impl.o

# Creates the intermediate and output folders for each configuration...
.PHONY: create_folders
create_folders:
	mkdir -p x64/Debug/libidaten/src/libidaten/cuda
	mkdir -p x64/Debug/libidaten/src/libidaten/kernel
	mkdir -p x64/Debug/libidaten/src/libaten/light
	mkdir -p x64/Debug/libidaten/src/libaten/material
	mkdir -p x64/Debug/libidaten/src/libaten/math
	mkdir -p x64/Debug/libidaten/src/libaten/sampler
	mkdir -p x64/Debug/libidaten/src/libaten/renderer
	mkdir -p x64/Debug/libidaten/src/libaten/os/linux/misc
	mkdir -p x64/Debug/libidaten/src/libaten/misc
	mkdir -p x64/Debug/libidaten/src/libidaten/svgf
	mkdir -p x64/Debug/libidaten/src/libaten/geometry
	mkdir -p x64/Debug/libidaten/src/libaten/camera
	mkdir -p x64/Debug
	mkdir -p x64/Release/libidaten/src/libidaten/cuda
	mkdir -p x64/Release/libidaten/src/libidaten/kernel
	mkdir -p x64/Release/libidaten/src/libaten/light
	mkdir -p x64/Release/libidaten/src/libaten/material
	mkdir -p x64/Release/libidaten/src/libaten/math
	mkdir -p x64/Release/libidaten/src/libaten/sampler
	mkdir -p x64/Release/libidaten/src/libaten/renderer
	mkdir -p x64/Release/libidaten/src/libaten/os/linux/misc
	mkdir -p x64/Release/libidaten/src/libaten/misc
	mkdir -p x64/Release/libidaten/src/libidaten/svgf
	mkdir -p x64/Release/libidaten/src/libaten/geometry
	mkdir -p x64/Release/libidaten/src/libaten/camera
	mkdir -p x64/Release

# Cleans intermediate and output files (objects, libraries, executables)...
.PHONY: clean
clean:
	rm -f x64/Debug/libidaten/src/libidaten/cuda/*.o
	rm -f x64/Debug/libidaten/src/libidaten/cuda/*.d
	rm -f x64/Debug/libidaten/src/libidaten/kernel/*.o
	rm -f x64/Debug/libidaten/src/libidaten/kernel/*.d
	rm -f x64/Debug/libidaten/src/libaten/light/*.o
	rm -f x64/Debug/libidaten/src/libaten/light/*.d
	rm -f x64/Debug/libidaten/src/libaten/material/*.o
	rm -f x64/Debug/libidaten/src/libaten/material/*.d
	rm -f x64/Debug/libidaten/src/libaten/math/*.o
	rm -f x64/Debug/libidaten/src/libaten/math/*.d
	rm -f x64/Debug/libidaten/src/libaten/sampler/*.o
	rm -f x64/Debug/libidaten/src/libaten/sampler/*.d
	rm -f x64/Debug/libidaten/src/libaten/renderer/*.o
	rm -f x64/Debug/libidaten/src/libaten/renderer/*.d
	rm -f x64/Debug/libidaten/src/libaten/os/linux/misc/*.o
	rm -f x64/Debug/libidaten/src/libaten/os/linux/misc/*.d
	rm -f x64/Debug/libidaten/src/libaten/misc/*.o
	rm -f x64/Debug/libidaten/src/libaten/misc/*.d
	rm -f x64/Debug/libidaten/src/libidaten/svgf/*.o
	rm -f x64/Debug/libidaten/src/libidaten/svgf/*.d
	rm -f x64/Debug/libidaten/src/libaten/geometry/*.o
	rm -f x64/Debug/libidaten/src/libaten/geometry/*.d
	rm -f x64/Debug/libidaten/src/libaten/camera/*.o
	rm -f x64/Debug/libidaten/src/libaten/camera/*.d
	rm -f x64/Debug/libidaten.a
	rm -f x64/Release/libidaten/src/libidaten/cuda/*.o
	rm -f x64/Release/libidaten/src/libidaten/cuda/*.d
	rm -f x64/Release/libidaten/src/libidaten/kernel/*.o
	rm -f x64/Release/libidaten/src/libidaten/kernel/*.d
	rm -f x64/Release/libidaten/src/libaten/light/*.o
	rm -f x64/Release/libidaten/src/libaten/light/*.d
	rm -f x64/Release/libidaten/src/libaten/material/*.o
	rm -f x64/Release/libidaten/src/libaten/material/*.d
	rm -f x64/Release/libidaten/src/libaten/math/*.o
	rm -f x64/Release/libidaten/src/libaten/math/*.d
	rm -f x64/Release/libidaten/src/libaten/sampler/*.o
	rm -f x64/Release/libidaten/src/libaten/sampler/*.d
	rm -f x64/Release/libidaten/src/libaten/renderer/*.o
	rm -f x64/Release/libidaten/src/libaten/renderer/*.d
	rm -f x64/Release/libidaten/src/libaten/os/linux/misc/*.o
	rm -f x64/Release/libidaten/src/libaten/os/linux/misc/*.d
	rm -f x64/Release/libidaten/src/libaten/misc/*.o
	rm -f x64/Release/libidaten/src/libaten/misc/*.d
	rm -f x64/Release/libidaten/src/libidaten/svgf/*.o
	rm -f x64/Release/libidaten/src/libidaten/svgf/*.d
	rm -f x64/Release/libidaten/src/libaten/geometry/*.o
	rm -f x64/Release/libidaten/src/libaten/geometry/*.d
	rm -f x64/Release/libidaten/src/libaten/camera/*.o
	rm -f x64/Release/libidaten/src/libaten/camera/*.d
	rm -f x64/Release/libidaten.a

