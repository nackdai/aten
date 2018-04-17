# Compiler flags...
CPP_COMPILER = g++
C_COMPILER = gcc

# Include paths...
Debug_Include_Path=-I"../src/libaten" -I"../3rdparty/glew/include" -I"../3rdparty/glfw/include" -I"../3rdparty/stb" -I"../3rdparty/glm" -I"../3rdparty/imgui" 
Release_Include_Path=-I"../src/libaten" -I"../3rdparty/glew/include" -I"../3rdparty/glfw/include" -I"../3rdparty/stb" -I"../3rdparty/glm" -I"../3rdparty/imgui" 

# Library paths...
Debug_Library_Path=
Release_Library_Path=

# Additional libraries...
Debug_Libraries=-Wl,--no-as-needed -Wl,--start-group -l$(NOINHERIT)  -Wl,--end-group
Release_Libraries=-Wl,--no-as-needed -Wl,--start-group -l$(NOINHERIT)  -Wl,--end-group

# Preprocessor definitions...
Debug_Preprocessor_Definitions=-D __AT_DEBUG__ -D GCC_BUILD 
Release_Preprocessor_Definitions=-D GCC_BUILD 

# Implictly linked object files...
Debug_Implicitly_Linked_Objects=
Release_Implicitly_Linked_Objects=

# Compiler flags...
Debug_Compiler_Flags=-O0 -g -std=c++11 -fopenmp 
Release_Compiler_Flags=-O2 -g -std=c++11 -fopenmp 

# Builds all configurations for this project...
.PHONY: build_all_configurations
build_all_configurations: Debug Release 

# Builds the Debug configuration...
.PHONY: Debug
Debug: create_folders x64/Debug/libaten/src/libaten/misc/color.o x64/Debug/libaten/src/libaten/misc/omputil.o x64/Debug/libaten/src/libaten/misc/thread.o x64/Debug/libaten/src/libaten/misc/timeline.o x64/Debug/libaten/src/libaten/os/linux/misc/timer_linux.o x64/Debug/libaten/src/libaten/math/mat4.o x64/Debug/libaten/src/libaten/renderer/aov.o x64/Debug/libaten/src/libaten/renderer/bdpt.o x64/Debug/libaten/src/libaten/renderer/directlight.o x64/Debug/libaten/src/libaten/renderer/envmap.o x64/Debug/libaten/src/libaten/renderer/erpt.o x64/Debug/libaten/src/libaten/renderer/film.o x64/Debug/libaten/src/libaten/renderer/nonphotoreal.o x64/Debug/libaten/src/libaten/renderer/pathtracing.o x64/Debug/libaten/src/libaten/renderer/pssmlt.o x64/Debug/libaten/src/libaten/renderer/raytracing.o x64/Debug/libaten/src/libaten/renderer/sorted_pathtracing.o x64/Debug/libaten/src/libaten/material/beckman.o x64/Debug/libaten/src/libaten/material/blinn.o x64/Debug/libaten/src/libaten/material/carpaint.o x64/Debug/libaten/src/libaten/material/disney_brdf.o x64/Debug/libaten/src/libaten/material/FlakesNormal.o x64/Debug/libaten/src/libaten/material/ggx.o x64/Debug/libaten/src/libaten/material/layer.o x64/Debug/libaten/src/libaten/material/material.o x64/Debug/libaten/src/libaten/material/oren_nayar.o x64/Debug/libaten/src/libaten/material/refraction.o x64/Debug/libaten/src/libaten/material/specular.o x64/Debug/libaten/src/libaten/material/toon.o x64/Debug/libaten/src/libaten/sampler/halton.o x64/Debug/libaten/src/libaten/sampler/sampler.o x64/Debug/libaten/src/libaten/sampler/sobol.o x64/Debug/libaten/src/libaten/camera/CameraOperator.o x64/Debug/libaten/src/libaten/camera/equirect.o x64/Debug/libaten/src/libaten/camera/pinhole.o x64/Debug/libaten/src/libaten/camera/thinlens.o x64/Debug/libaten/src/libaten/scene/hitable.o x64/Debug/libaten/src/libaten/scene/scene.o x64/Debug/libaten/src/libaten/visualizer/blitter.o x64/Debug/libaten/src/libaten/visualizer/fbo.o x64/Debug/libaten/src/libaten/visualizer/GeomDataBuffer.o x64/Debug/libaten/src/libaten/visualizer/GLProfiler.o x64/Debug/libaten/src/libaten/visualizer/MultiPassPostProc.o x64/Debug/libaten/src/libaten/visualizer/RasterizeRenderer.o x64/Debug/libaten/src/libaten/visualizer/shader.o x64/Debug/libaten/src/libaten/visualizer/visualizer.o x64/Debug/libaten/src/libaten/visualizer/window.o x64/Debug/libaten/src/libaten/hdr/gamma.o x64/Debug/libaten/src/libaten/hdr/hdr.o x64/Debug/libaten/src/libaten/hdr/tonemap.o x64/Debug/libaten/src/libaten/texture/texture.o x64/Debug/libaten/src/libaten/light/arealight.o x64/Debug/libaten/src/libaten/light/ibl.o x64/Debug/libaten/src/libaten/light/light.o x64/Debug/libaten/src/libaten/posteffect/BloomEffect.o x64/Debug/libaten/src/libaten/filter/atrous.o x64/Debug/libaten/src/libaten/filter/bilateral.o x64/Debug/libaten/src/libaten/filter/nlm.o x64/Debug/libaten/src/libaten/filter/taa.o x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.o x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.o x64/Debug/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.o x64/Debug/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.o x64/Debug/libaten/src/libaten/proxy/DataCollector.o x64/Debug/libaten/src/libaten/accelerator/accelerator.o x64/Debug/libaten/src/libaten/accelerator/bvh.o x64/Debug/libaten/src/libaten/accelerator/bvh_frustum.o x64/Debug/libaten/src/libaten/accelerator/bvh_update.o x64/Debug/libaten/src/libaten/accelerator/CullingFrusta.o x64/Debug/libaten/src/libaten/accelerator/qbvh.o x64/Debug/libaten/src/libaten/accelerator/sbvh.o x64/Debug/libaten/src/libaten/accelerator/sbvh_voxel.o x64/Debug/libaten/src/libaten/accelerator/stackless_bvh.o x64/Debug/libaten/src/libaten/accelerator/stackless_qbvh.o x64/Debug/libaten/src/libaten/accelerator/ThreadedBvhFrustum.o x64/Debug/libaten/src/libaten/accelerator/threaded_bvh.o x64/Debug/libaten/src/libaten/ui/imgui_impl_glfw_gl3.o x64/Debug/libaten/3rdparty/imgui/imgui.o x64/Debug/libaten/3rdparty/imgui/imgui_draw.o x64/Debug/libaten/src/libaten/geometry/cube.o x64/Debug/libaten/src/libaten/geometry/face.o x64/Debug/libaten/src/libaten/geometry/geombase.o x64/Debug/libaten/src/libaten/geometry/object.o x64/Debug/libaten/src/libaten/geometry/objshape.o x64/Debug/libaten/src/libaten/geometry/sphere.o x64/Debug/libaten/src/libaten/geometry/tranformable.o x64/Debug/libaten/src/libaten/geometry/vertex.o x64/Debug/libaten/src/libaten/deformable/deformable.o x64/Debug/libaten/src/libaten/deformable/DeformMesh.o x64/Debug/libaten/src/libaten/deformable/DeformMeshGroup.o x64/Debug/libaten/src/libaten/deformable/DeformMeshSet.o x64/Debug/libaten/src/libaten/deformable/DeformPrimitives.o x64/Debug/libaten/src/libaten/deformable/Skeleton.o x64/Debug/libaten/src/libaten/deformable/DeformAnimation.o x64/Debug/libaten/src/libaten/deformable/DeformAnimationInterp.o x64/Debug/libaten/src/libaten/os/linux/system_linux.o 
	ar rcs x64/Debug/libaten.a x64/Debug/libaten/src/libaten/misc/color.o x64/Debug/libaten/src/libaten/misc/omputil.o x64/Debug/libaten/src/libaten/misc/thread.o x64/Debug/libaten/src/libaten/misc/timeline.o x64/Debug/libaten/src/libaten/os/linux/misc/timer_linux.o x64/Debug/libaten/src/libaten/math/mat4.o x64/Debug/libaten/src/libaten/renderer/aov.o x64/Debug/libaten/src/libaten/renderer/bdpt.o x64/Debug/libaten/src/libaten/renderer/directlight.o x64/Debug/libaten/src/libaten/renderer/envmap.o x64/Debug/libaten/src/libaten/renderer/erpt.o x64/Debug/libaten/src/libaten/renderer/film.o x64/Debug/libaten/src/libaten/renderer/nonphotoreal.o x64/Debug/libaten/src/libaten/renderer/pathtracing.o x64/Debug/libaten/src/libaten/renderer/pssmlt.o x64/Debug/libaten/src/libaten/renderer/raytracing.o x64/Debug/libaten/src/libaten/renderer/sorted_pathtracing.o x64/Debug/libaten/src/libaten/material/beckman.o x64/Debug/libaten/src/libaten/material/blinn.o x64/Debug/libaten/src/libaten/material/carpaint.o x64/Debug/libaten/src/libaten/material/disney_brdf.o x64/Debug/libaten/src/libaten/material/FlakesNormal.o x64/Debug/libaten/src/libaten/material/ggx.o x64/Debug/libaten/src/libaten/material/layer.o x64/Debug/libaten/src/libaten/material/material.o x64/Debug/libaten/src/libaten/material/oren_nayar.o x64/Debug/libaten/src/libaten/material/refraction.o x64/Debug/libaten/src/libaten/material/specular.o x64/Debug/libaten/src/libaten/material/toon.o x64/Debug/libaten/src/libaten/sampler/halton.o x64/Debug/libaten/src/libaten/sampler/sampler.o x64/Debug/libaten/src/libaten/sampler/sobol.o x64/Debug/libaten/src/libaten/camera/CameraOperator.o x64/Debug/libaten/src/libaten/camera/equirect.o x64/Debug/libaten/src/libaten/camera/pinhole.o x64/Debug/libaten/src/libaten/camera/thinlens.o x64/Debug/libaten/src/libaten/scene/hitable.o x64/Debug/libaten/src/libaten/scene/scene.o x64/Debug/libaten/src/libaten/visualizer/blitter.o x64/Debug/libaten/src/libaten/visualizer/fbo.o x64/Debug/libaten/src/libaten/visualizer/GeomDataBuffer.o x64/Debug/libaten/src/libaten/visualizer/GLProfiler.o x64/Debug/libaten/src/libaten/visualizer/MultiPassPostProc.o x64/Debug/libaten/src/libaten/visualizer/RasterizeRenderer.o x64/Debug/libaten/src/libaten/visualizer/shader.o x64/Debug/libaten/src/libaten/visualizer/visualizer.o x64/Debug/libaten/src/libaten/visualizer/window.o x64/Debug/libaten/src/libaten/hdr/gamma.o x64/Debug/libaten/src/libaten/hdr/hdr.o x64/Debug/libaten/src/libaten/hdr/tonemap.o x64/Debug/libaten/src/libaten/texture/texture.o x64/Debug/libaten/src/libaten/light/arealight.o x64/Debug/libaten/src/libaten/light/ibl.o x64/Debug/libaten/src/libaten/light/light.o x64/Debug/libaten/src/libaten/posteffect/BloomEffect.o x64/Debug/libaten/src/libaten/filter/atrous.o x64/Debug/libaten/src/libaten/filter/bilateral.o x64/Debug/libaten/src/libaten/filter/nlm.o x64/Debug/libaten/src/libaten/filter/taa.o x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.o x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.o x64/Debug/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.o x64/Debug/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.o x64/Debug/libaten/src/libaten/proxy/DataCollector.o x64/Debug/libaten/src/libaten/accelerator/accelerator.o x64/Debug/libaten/src/libaten/accelerator/bvh.o x64/Debug/libaten/src/libaten/accelerator/bvh_frustum.o x64/Debug/libaten/src/libaten/accelerator/bvh_update.o x64/Debug/libaten/src/libaten/accelerator/CullingFrusta.o x64/Debug/libaten/src/libaten/accelerator/qbvh.o x64/Debug/libaten/src/libaten/accelerator/sbvh.o x64/Debug/libaten/src/libaten/accelerator/sbvh_voxel.o x64/Debug/libaten/src/libaten/accelerator/stackless_bvh.o x64/Debug/libaten/src/libaten/accelerator/stackless_qbvh.o x64/Debug/libaten/src/libaten/accelerator/ThreadedBvhFrustum.o x64/Debug/libaten/src/libaten/accelerator/threaded_bvh.o x64/Debug/libaten/src/libaten/ui/imgui_impl_glfw_gl3.o x64/Debug/libaten/3rdparty/imgui/imgui.o x64/Debug/libaten/3rdparty/imgui/imgui_draw.o x64/Debug/libaten/src/libaten/geometry/cube.o x64/Debug/libaten/src/libaten/geometry/face.o x64/Debug/libaten/src/libaten/geometry/geombase.o x64/Debug/libaten/src/libaten/geometry/object.o x64/Debug/libaten/src/libaten/geometry/objshape.o x64/Debug/libaten/src/libaten/geometry/sphere.o x64/Debug/libaten/src/libaten/geometry/tranformable.o x64/Debug/libaten/src/libaten/geometry/vertex.o x64/Debug/libaten/src/libaten/deformable/deformable.o x64/Debug/libaten/src/libaten/deformable/DeformMesh.o x64/Debug/libaten/src/libaten/deformable/DeformMeshGroup.o x64/Debug/libaten/src/libaten/deformable/DeformMeshSet.o x64/Debug/libaten/src/libaten/deformable/DeformPrimitives.o x64/Debug/libaten/src/libaten/deformable/Skeleton.o x64/Debug/libaten/src/libaten/deformable/DeformAnimation.o x64/Debug/libaten/src/libaten/deformable/DeformAnimationInterp.o x64/Debug/libaten/src/libaten/os/linux/system_linux.o  $(Debug_Implicitly_Linked_Objects)

# Compiles file ../src/libaten/misc/color.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/misc/color.d
x64/Debug/libaten/src/libaten/misc/color.o: ../src/libaten/misc/color.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/misc/color.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/misc/color.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/misc/color.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/misc/color.d

# Compiles file ../src/libaten/misc/omputil.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/misc/omputil.d
x64/Debug/libaten/src/libaten/misc/omputil.o: ../src/libaten/misc/omputil.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/misc/omputil.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/misc/omputil.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/misc/omputil.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/misc/omputil.d

# Compiles file ../src/libaten/misc/thread.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/misc/thread.d
x64/Debug/libaten/src/libaten/misc/thread.o: ../src/libaten/misc/thread.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/misc/thread.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/misc/thread.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/misc/thread.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/misc/thread.d

# Compiles file ../src/libaten/misc/timeline.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/misc/timeline.d
x64/Debug/libaten/src/libaten/misc/timeline.o: ../src/libaten/misc/timeline.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/misc/timeline.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/misc/timeline.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/misc/timeline.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/misc/timeline.d

# Compiles file ../src/libaten/os/linux/misc/timer_linux.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/os/linux/misc/timer_linux.d
x64/Debug/libaten/src/libaten/os/linux/misc/timer_linux.o: ../src/libaten/os/linux/misc/timer_linux.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/os/linux/misc/timer_linux.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/os/linux/misc/timer_linux.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/os/linux/misc/timer_linux.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/os/linux/misc/timer_linux.d

# Compiles file ../src/libaten/math/mat4.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/math/mat4.d
x64/Debug/libaten/src/libaten/math/mat4.o: ../src/libaten/math/mat4.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/math/mat4.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/math/mat4.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/math/mat4.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/math/mat4.d

# Compiles file ../src/libaten/renderer/aov.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/aov.d
x64/Debug/libaten/src/libaten/renderer/aov.o: ../src/libaten/renderer/aov.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/aov.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/aov.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/aov.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/aov.d

# Compiles file ../src/libaten/renderer/bdpt.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/bdpt.d
x64/Debug/libaten/src/libaten/renderer/bdpt.o: ../src/libaten/renderer/bdpt.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/bdpt.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/bdpt.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/bdpt.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/bdpt.d

# Compiles file ../src/libaten/renderer/directlight.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/directlight.d
x64/Debug/libaten/src/libaten/renderer/directlight.o: ../src/libaten/renderer/directlight.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/directlight.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/directlight.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/directlight.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/directlight.d

# Compiles file ../src/libaten/renderer/envmap.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/envmap.d
x64/Debug/libaten/src/libaten/renderer/envmap.o: ../src/libaten/renderer/envmap.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/envmap.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/envmap.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/envmap.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/envmap.d

# Compiles file ../src/libaten/renderer/erpt.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/erpt.d
x64/Debug/libaten/src/libaten/renderer/erpt.o: ../src/libaten/renderer/erpt.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/erpt.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/erpt.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/erpt.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/erpt.d

# Compiles file ../src/libaten/renderer/film.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/film.d
x64/Debug/libaten/src/libaten/renderer/film.o: ../src/libaten/renderer/film.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/film.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/film.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/film.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/film.d

# Compiles file ../src/libaten/renderer/nonphotoreal.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/nonphotoreal.d
x64/Debug/libaten/src/libaten/renderer/nonphotoreal.o: ../src/libaten/renderer/nonphotoreal.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/nonphotoreal.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/nonphotoreal.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/nonphotoreal.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/nonphotoreal.d

# Compiles file ../src/libaten/renderer/pathtracing.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/pathtracing.d
x64/Debug/libaten/src/libaten/renderer/pathtracing.o: ../src/libaten/renderer/pathtracing.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/pathtracing.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/pathtracing.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/pathtracing.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/pathtracing.d

# Compiles file ../src/libaten/renderer/pssmlt.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/pssmlt.d
x64/Debug/libaten/src/libaten/renderer/pssmlt.o: ../src/libaten/renderer/pssmlt.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/pssmlt.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/pssmlt.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/pssmlt.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/pssmlt.d

# Compiles file ../src/libaten/renderer/raytracing.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/raytracing.d
x64/Debug/libaten/src/libaten/renderer/raytracing.o: ../src/libaten/renderer/raytracing.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/raytracing.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/raytracing.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/raytracing.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/raytracing.d

# Compiles file ../src/libaten/renderer/sorted_pathtracing.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/renderer/sorted_pathtracing.d
x64/Debug/libaten/src/libaten/renderer/sorted_pathtracing.o: ../src/libaten/renderer/sorted_pathtracing.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/renderer/sorted_pathtracing.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/renderer/sorted_pathtracing.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/renderer/sorted_pathtracing.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/renderer/sorted_pathtracing.d

# Compiles file ../src/libaten/material/beckman.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/beckman.d
x64/Debug/libaten/src/libaten/material/beckman.o: ../src/libaten/material/beckman.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/beckman.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/beckman.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/beckman.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/beckman.d

# Compiles file ../src/libaten/material/blinn.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/blinn.d
x64/Debug/libaten/src/libaten/material/blinn.o: ../src/libaten/material/blinn.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/blinn.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/blinn.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/blinn.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/blinn.d

# Compiles file ../src/libaten/material/carpaint.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/carpaint.d
x64/Debug/libaten/src/libaten/material/carpaint.o: ../src/libaten/material/carpaint.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/carpaint.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/carpaint.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/carpaint.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/carpaint.d

# Compiles file ../src/libaten/material/disney_brdf.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/disney_brdf.d
x64/Debug/libaten/src/libaten/material/disney_brdf.o: ../src/libaten/material/disney_brdf.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/disney_brdf.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/disney_brdf.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/disney_brdf.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/disney_brdf.d

# Compiles file ../src/libaten/material/FlakesNormal.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/FlakesNormal.d
x64/Debug/libaten/src/libaten/material/FlakesNormal.o: ../src/libaten/material/FlakesNormal.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/FlakesNormal.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/FlakesNormal.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/FlakesNormal.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/FlakesNormal.d

# Compiles file ../src/libaten/material/ggx.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/ggx.d
x64/Debug/libaten/src/libaten/material/ggx.o: ../src/libaten/material/ggx.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/ggx.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/ggx.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/ggx.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/ggx.d

# Compiles file ../src/libaten/material/layer.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/layer.d
x64/Debug/libaten/src/libaten/material/layer.o: ../src/libaten/material/layer.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/layer.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/layer.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/layer.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/layer.d

# Compiles file ../src/libaten/material/material.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/material.d
x64/Debug/libaten/src/libaten/material/material.o: ../src/libaten/material/material.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/material.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/material.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/material.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/material.d

# Compiles file ../src/libaten/material/oren_nayar.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/oren_nayar.d
x64/Debug/libaten/src/libaten/material/oren_nayar.o: ../src/libaten/material/oren_nayar.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/oren_nayar.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/oren_nayar.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/oren_nayar.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/oren_nayar.d

# Compiles file ../src/libaten/material/refraction.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/refraction.d
x64/Debug/libaten/src/libaten/material/refraction.o: ../src/libaten/material/refraction.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/refraction.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/refraction.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/refraction.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/refraction.d

# Compiles file ../src/libaten/material/specular.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/specular.d
x64/Debug/libaten/src/libaten/material/specular.o: ../src/libaten/material/specular.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/specular.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/specular.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/specular.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/specular.d

# Compiles file ../src/libaten/material/toon.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/material/toon.d
x64/Debug/libaten/src/libaten/material/toon.o: ../src/libaten/material/toon.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/material/toon.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/material/toon.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/material/toon.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/material/toon.d

# Compiles file ../src/libaten/sampler/halton.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/sampler/halton.d
x64/Debug/libaten/src/libaten/sampler/halton.o: ../src/libaten/sampler/halton.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/sampler/halton.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/sampler/halton.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/sampler/halton.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/sampler/halton.d

# Compiles file ../src/libaten/sampler/sampler.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/sampler/sampler.d
x64/Debug/libaten/src/libaten/sampler/sampler.o: ../src/libaten/sampler/sampler.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/sampler/sampler.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/sampler/sampler.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/sampler/sampler.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/sampler/sampler.d

# Compiles file ../src/libaten/sampler/sobol.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/sampler/sobol.d
x64/Debug/libaten/src/libaten/sampler/sobol.o: ../src/libaten/sampler/sobol.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/sampler/sobol.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/sampler/sobol.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/sampler/sobol.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/sampler/sobol.d

# Compiles file ../src/libaten/camera/CameraOperator.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/camera/CameraOperator.d
x64/Debug/libaten/src/libaten/camera/CameraOperator.o: ../src/libaten/camera/CameraOperator.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/camera/CameraOperator.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/camera/CameraOperator.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/camera/CameraOperator.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/camera/CameraOperator.d

# Compiles file ../src/libaten/camera/equirect.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/camera/equirect.d
x64/Debug/libaten/src/libaten/camera/equirect.o: ../src/libaten/camera/equirect.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/camera/equirect.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/camera/equirect.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/camera/equirect.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/camera/equirect.d

# Compiles file ../src/libaten/camera/pinhole.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/camera/pinhole.d
x64/Debug/libaten/src/libaten/camera/pinhole.o: ../src/libaten/camera/pinhole.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/camera/pinhole.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/camera/pinhole.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/camera/pinhole.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/camera/pinhole.d

# Compiles file ../src/libaten/camera/thinlens.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/camera/thinlens.d
x64/Debug/libaten/src/libaten/camera/thinlens.o: ../src/libaten/camera/thinlens.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/camera/thinlens.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/camera/thinlens.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/camera/thinlens.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/camera/thinlens.d

# Compiles file ../src/libaten/scene/hitable.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/scene/hitable.d
x64/Debug/libaten/src/libaten/scene/hitable.o: ../src/libaten/scene/hitable.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/scene/hitable.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/scene/hitable.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/scene/hitable.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/scene/hitable.d

# Compiles file ../src/libaten/scene/scene.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/scene/scene.d
x64/Debug/libaten/src/libaten/scene/scene.o: ../src/libaten/scene/scene.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/scene/scene.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/scene/scene.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/scene/scene.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/scene/scene.d

# Compiles file ../src/libaten/visualizer/blitter.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/visualizer/blitter.d
x64/Debug/libaten/src/libaten/visualizer/blitter.o: ../src/libaten/visualizer/blitter.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/visualizer/blitter.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/visualizer/blitter.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/visualizer/blitter.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/visualizer/blitter.d

# Compiles file ../src/libaten/visualizer/fbo.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/visualizer/fbo.d
x64/Debug/libaten/src/libaten/visualizer/fbo.o: ../src/libaten/visualizer/fbo.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/visualizer/fbo.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/visualizer/fbo.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/visualizer/fbo.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/visualizer/fbo.d

# Compiles file ../src/libaten/visualizer/GeomDataBuffer.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/visualizer/GeomDataBuffer.d
x64/Debug/libaten/src/libaten/visualizer/GeomDataBuffer.o: ../src/libaten/visualizer/GeomDataBuffer.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/visualizer/GeomDataBuffer.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/visualizer/GeomDataBuffer.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/visualizer/GeomDataBuffer.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/visualizer/GeomDataBuffer.d

# Compiles file ../src/libaten/visualizer/GLProfiler.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/visualizer/GLProfiler.d
x64/Debug/libaten/src/libaten/visualizer/GLProfiler.o: ../src/libaten/visualizer/GLProfiler.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/visualizer/GLProfiler.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/visualizer/GLProfiler.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/visualizer/GLProfiler.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/visualizer/GLProfiler.d

# Compiles file ../src/libaten/visualizer/MultiPassPostProc.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/visualizer/MultiPassPostProc.d
x64/Debug/libaten/src/libaten/visualizer/MultiPassPostProc.o: ../src/libaten/visualizer/MultiPassPostProc.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/visualizer/MultiPassPostProc.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/visualizer/MultiPassPostProc.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/visualizer/MultiPassPostProc.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/visualizer/MultiPassPostProc.d

# Compiles file ../src/libaten/visualizer/RasterizeRenderer.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/visualizer/RasterizeRenderer.d
x64/Debug/libaten/src/libaten/visualizer/RasterizeRenderer.o: ../src/libaten/visualizer/RasterizeRenderer.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/visualizer/RasterizeRenderer.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/visualizer/RasterizeRenderer.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/visualizer/RasterizeRenderer.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/visualizer/RasterizeRenderer.d

# Compiles file ../src/libaten/visualizer/shader.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/visualizer/shader.d
x64/Debug/libaten/src/libaten/visualizer/shader.o: ../src/libaten/visualizer/shader.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/visualizer/shader.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/visualizer/shader.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/visualizer/shader.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/visualizer/shader.d

# Compiles file ../src/libaten/visualizer/visualizer.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/visualizer/visualizer.d
x64/Debug/libaten/src/libaten/visualizer/visualizer.o: ../src/libaten/visualizer/visualizer.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/visualizer/visualizer.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/visualizer/visualizer.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/visualizer/visualizer.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/visualizer/visualizer.d

# Compiles file ../src/libaten/visualizer/window.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/visualizer/window.d
x64/Debug/libaten/src/libaten/visualizer/window.o: ../src/libaten/visualizer/window.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/visualizer/window.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/visualizer/window.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/visualizer/window.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/visualizer/window.d

# Compiles file ../src/libaten/hdr/gamma.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/hdr/gamma.d
x64/Debug/libaten/src/libaten/hdr/gamma.o: ../src/libaten/hdr/gamma.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/hdr/gamma.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/hdr/gamma.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/hdr/gamma.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/hdr/gamma.d

# Compiles file ../src/libaten/hdr/hdr.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/hdr/hdr.d
x64/Debug/libaten/src/libaten/hdr/hdr.o: ../src/libaten/hdr/hdr.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/hdr/hdr.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/hdr/hdr.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/hdr/hdr.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/hdr/hdr.d

# Compiles file ../src/libaten/hdr/tonemap.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/hdr/tonemap.d
x64/Debug/libaten/src/libaten/hdr/tonemap.o: ../src/libaten/hdr/tonemap.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/hdr/tonemap.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/hdr/tonemap.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/hdr/tonemap.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/hdr/tonemap.d

# Compiles file ../src/libaten/texture/texture.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/texture/texture.d
x64/Debug/libaten/src/libaten/texture/texture.o: ../src/libaten/texture/texture.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/texture/texture.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/texture/texture.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/texture/texture.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/texture/texture.d

# Compiles file ../src/libaten/light/arealight.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/light/arealight.d
x64/Debug/libaten/src/libaten/light/arealight.o: ../src/libaten/light/arealight.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/light/arealight.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/light/arealight.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/light/arealight.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/light/arealight.d

# Compiles file ../src/libaten/light/ibl.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/light/ibl.d
x64/Debug/libaten/src/libaten/light/ibl.o: ../src/libaten/light/ibl.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/light/ibl.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/light/ibl.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/light/ibl.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/light/ibl.d

# Compiles file ../src/libaten/light/light.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/light/light.d
x64/Debug/libaten/src/libaten/light/light.o: ../src/libaten/light/light.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/light/light.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/light/light.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/light/light.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/light/light.d

# Compiles file ../src/libaten/posteffect/BloomEffect.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/posteffect/BloomEffect.d
x64/Debug/libaten/src/libaten/posteffect/BloomEffect.o: ../src/libaten/posteffect/BloomEffect.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/posteffect/BloomEffect.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/posteffect/BloomEffect.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/posteffect/BloomEffect.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/posteffect/BloomEffect.d

# Compiles file ../src/libaten/filter/atrous.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/filter/atrous.d
x64/Debug/libaten/src/libaten/filter/atrous.o: ../src/libaten/filter/atrous.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/filter/atrous.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/filter/atrous.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/filter/atrous.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/filter/atrous.d

# Compiles file ../src/libaten/filter/bilateral.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/filter/bilateral.d
x64/Debug/libaten/src/libaten/filter/bilateral.o: ../src/libaten/filter/bilateral.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/filter/bilateral.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/filter/bilateral.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/filter/bilateral.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/filter/bilateral.d

# Compiles file ../src/libaten/filter/nlm.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/filter/nlm.d
x64/Debug/libaten/src/libaten/filter/nlm.o: ../src/libaten/filter/nlm.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/filter/nlm.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/filter/nlm.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/filter/nlm.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/filter/nlm.d

# Compiles file ../src/libaten/filter/taa.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/filter/taa.d
x64/Debug/libaten/src/libaten/filter/taa.o: ../src/libaten/filter/taa.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/filter/taa.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/filter/taa.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/filter/taa.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/filter/taa.d

# Compiles file ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.d
x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.o: ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.d

# Compiles file ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.d
x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.o: ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.d

# Compiles file ../src/libaten/filter/VirtualFlashImage/VirtualFlashImage.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.d
x64/Debug/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.o: ../src/libaten/filter/VirtualFlashImage/VirtualFlashImage.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/filter/VirtualFlashImage/VirtualFlashImage.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/filter/VirtualFlashImage/VirtualFlashImage.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.d

# Compiles file ../src/libaten/filter/GeometryRendering/GeometryRendering.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.d
x64/Debug/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.o: ../src/libaten/filter/GeometryRendering/GeometryRendering.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/filter/GeometryRendering/GeometryRendering.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/filter/GeometryRendering/GeometryRendering.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.d

# Compiles file ../src/libaten/proxy/DataCollector.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/proxy/DataCollector.d
x64/Debug/libaten/src/libaten/proxy/DataCollector.o: ../src/libaten/proxy/DataCollector.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/proxy/DataCollector.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/proxy/DataCollector.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/proxy/DataCollector.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/proxy/DataCollector.d

# Compiles file ../src/libaten/accelerator/accelerator.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/accelerator.d
x64/Debug/libaten/src/libaten/accelerator/accelerator.o: ../src/libaten/accelerator/accelerator.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/accelerator.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/accelerator.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/accelerator.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/accelerator.d

# Compiles file ../src/libaten/accelerator/bvh.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/bvh.d
x64/Debug/libaten/src/libaten/accelerator/bvh.o: ../src/libaten/accelerator/bvh.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/bvh.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/bvh.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/bvh.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/bvh.d

# Compiles file ../src/libaten/accelerator/bvh_frustum.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/bvh_frustum.d
x64/Debug/libaten/src/libaten/accelerator/bvh_frustum.o: ../src/libaten/accelerator/bvh_frustum.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/bvh_frustum.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/bvh_frustum.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/bvh_frustum.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/bvh_frustum.d

# Compiles file ../src/libaten/accelerator/bvh_update.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/bvh_update.d
x64/Debug/libaten/src/libaten/accelerator/bvh_update.o: ../src/libaten/accelerator/bvh_update.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/bvh_update.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/bvh_update.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/bvh_update.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/bvh_update.d

# Compiles file ../src/libaten/accelerator/CullingFrusta.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/CullingFrusta.d
x64/Debug/libaten/src/libaten/accelerator/CullingFrusta.o: ../src/libaten/accelerator/CullingFrusta.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/CullingFrusta.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/CullingFrusta.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/CullingFrusta.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/CullingFrusta.d

# Compiles file ../src/libaten/accelerator/qbvh.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/qbvh.d
x64/Debug/libaten/src/libaten/accelerator/qbvh.o: ../src/libaten/accelerator/qbvh.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/qbvh.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/qbvh.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/qbvh.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/qbvh.d

# Compiles file ../src/libaten/accelerator/sbvh.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/sbvh.d
x64/Debug/libaten/src/libaten/accelerator/sbvh.o: ../src/libaten/accelerator/sbvh.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/sbvh.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/sbvh.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/sbvh.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/sbvh.d

# Compiles file ../src/libaten/accelerator/sbvh_voxel.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/sbvh_voxel.d
x64/Debug/libaten/src/libaten/accelerator/sbvh_voxel.o: ../src/libaten/accelerator/sbvh_voxel.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/sbvh_voxel.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/sbvh_voxel.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/sbvh_voxel.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/sbvh_voxel.d

# Compiles file ../src/libaten/accelerator/stackless_bvh.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/stackless_bvh.d
x64/Debug/libaten/src/libaten/accelerator/stackless_bvh.o: ../src/libaten/accelerator/stackless_bvh.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/stackless_bvh.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/stackless_bvh.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/stackless_bvh.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/stackless_bvh.d

# Compiles file ../src/libaten/accelerator/stackless_qbvh.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/stackless_qbvh.d
x64/Debug/libaten/src/libaten/accelerator/stackless_qbvh.o: ../src/libaten/accelerator/stackless_qbvh.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/stackless_qbvh.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/stackless_qbvh.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/stackless_qbvh.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/stackless_qbvh.d

# Compiles file ../src/libaten/accelerator/ThreadedBvhFrustum.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/ThreadedBvhFrustum.d
x64/Debug/libaten/src/libaten/accelerator/ThreadedBvhFrustum.o: ../src/libaten/accelerator/ThreadedBvhFrustum.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/ThreadedBvhFrustum.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/ThreadedBvhFrustum.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/ThreadedBvhFrustum.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/ThreadedBvhFrustum.d

# Compiles file ../src/libaten/accelerator/threaded_bvh.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/accelerator/threaded_bvh.d
x64/Debug/libaten/src/libaten/accelerator/threaded_bvh.o: ../src/libaten/accelerator/threaded_bvh.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/accelerator/threaded_bvh.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/accelerator/threaded_bvh.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/accelerator/threaded_bvh.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/accelerator/threaded_bvh.d

# Compiles file ../src/libaten/ui/imgui_impl_glfw_gl3.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/ui/imgui_impl_glfw_gl3.d
x64/Debug/libaten/src/libaten/ui/imgui_impl_glfw_gl3.o: ../src/libaten/ui/imgui_impl_glfw_gl3.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/ui/imgui_impl_glfw_gl3.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/ui/imgui_impl_glfw_gl3.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/ui/imgui_impl_glfw_gl3.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/ui/imgui_impl_glfw_gl3.d

# Compiles file ../3rdparty/imgui/imgui.cpp for the Debug configuration...
-include x64/Debug/libaten/3rdparty/imgui/imgui.d
x64/Debug/libaten/3rdparty/imgui/imgui.o: ../3rdparty/imgui/imgui.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../3rdparty/imgui/imgui.cpp $(Debug_Include_Path) -o x64/Debug/libaten/3rdparty/imgui/imgui.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../3rdparty/imgui/imgui.cpp $(Debug_Include_Path) > x64/Debug/libaten/3rdparty/imgui/imgui.d

# Compiles file ../3rdparty/imgui/imgui_draw.cpp for the Debug configuration...
-include x64/Debug/libaten/3rdparty/imgui/imgui_draw.d
x64/Debug/libaten/3rdparty/imgui/imgui_draw.o: ../3rdparty/imgui/imgui_draw.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../3rdparty/imgui/imgui_draw.cpp $(Debug_Include_Path) -o x64/Debug/libaten/3rdparty/imgui/imgui_draw.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../3rdparty/imgui/imgui_draw.cpp $(Debug_Include_Path) > x64/Debug/libaten/3rdparty/imgui/imgui_draw.d

# Compiles file ../src/libaten/geometry/cube.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/geometry/cube.d
x64/Debug/libaten/src/libaten/geometry/cube.o: ../src/libaten/geometry/cube.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/geometry/cube.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/geometry/cube.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/geometry/cube.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/geometry/cube.d

# Compiles file ../src/libaten/geometry/face.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/geometry/face.d
x64/Debug/libaten/src/libaten/geometry/face.o: ../src/libaten/geometry/face.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/geometry/face.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/geometry/face.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/geometry/face.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/geometry/face.d

# Compiles file ../src/libaten/geometry/geombase.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/geometry/geombase.d
x64/Debug/libaten/src/libaten/geometry/geombase.o: ../src/libaten/geometry/geombase.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/geometry/geombase.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/geometry/geombase.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/geometry/geombase.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/geometry/geombase.d

# Compiles file ../src/libaten/geometry/object.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/geometry/object.d
x64/Debug/libaten/src/libaten/geometry/object.o: ../src/libaten/geometry/object.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/geometry/object.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/geometry/object.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/geometry/object.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/geometry/object.d

# Compiles file ../src/libaten/geometry/objshape.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/geometry/objshape.d
x64/Debug/libaten/src/libaten/geometry/objshape.o: ../src/libaten/geometry/objshape.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/geometry/objshape.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/geometry/objshape.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/geometry/objshape.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/geometry/objshape.d

# Compiles file ../src/libaten/geometry/sphere.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/geometry/sphere.d
x64/Debug/libaten/src/libaten/geometry/sphere.o: ../src/libaten/geometry/sphere.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/geometry/sphere.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/geometry/sphere.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/geometry/sphere.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/geometry/sphere.d

# Compiles file ../src/libaten/geometry/tranformable.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/geometry/tranformable.d
x64/Debug/libaten/src/libaten/geometry/tranformable.o: ../src/libaten/geometry/tranformable.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/geometry/tranformable.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/geometry/tranformable.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/geometry/tranformable.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/geometry/tranformable.d

# Compiles file ../src/libaten/geometry/vertex.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/geometry/vertex.d
x64/Debug/libaten/src/libaten/geometry/vertex.o: ../src/libaten/geometry/vertex.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/geometry/vertex.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/geometry/vertex.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/geometry/vertex.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/geometry/vertex.d

# Compiles file ../src/libaten/deformable/deformable.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/deformable/deformable.d
x64/Debug/libaten/src/libaten/deformable/deformable.o: ../src/libaten/deformable/deformable.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/deformable/deformable.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/deformable/deformable.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/deformable/deformable.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/deformable/deformable.d

# Compiles file ../src/libaten/deformable/DeformMesh.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/deformable/DeformMesh.d
x64/Debug/libaten/src/libaten/deformable/DeformMesh.o: ../src/libaten/deformable/DeformMesh.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/deformable/DeformMesh.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/deformable/DeformMesh.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/deformable/DeformMesh.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/deformable/DeformMesh.d

# Compiles file ../src/libaten/deformable/DeformMeshGroup.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/deformable/DeformMeshGroup.d
x64/Debug/libaten/src/libaten/deformable/DeformMeshGroup.o: ../src/libaten/deformable/DeformMeshGroup.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/deformable/DeformMeshGroup.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/deformable/DeformMeshGroup.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/deformable/DeformMeshGroup.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/deformable/DeformMeshGroup.d

# Compiles file ../src/libaten/deformable/DeformMeshSet.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/deformable/DeformMeshSet.d
x64/Debug/libaten/src/libaten/deformable/DeformMeshSet.o: ../src/libaten/deformable/DeformMeshSet.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/deformable/DeformMeshSet.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/deformable/DeformMeshSet.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/deformable/DeformMeshSet.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/deformable/DeformMeshSet.d

# Compiles file ../src/libaten/deformable/DeformPrimitives.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/deformable/DeformPrimitives.d
x64/Debug/libaten/src/libaten/deformable/DeformPrimitives.o: ../src/libaten/deformable/DeformPrimitives.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/deformable/DeformPrimitives.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/deformable/DeformPrimitives.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/deformable/DeformPrimitives.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/deformable/DeformPrimitives.d

# Compiles file ../src/libaten/deformable/Skeleton.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/deformable/Skeleton.d
x64/Debug/libaten/src/libaten/deformable/Skeleton.o: ../src/libaten/deformable/Skeleton.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/deformable/Skeleton.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/deformable/Skeleton.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/deformable/Skeleton.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/deformable/Skeleton.d

# Compiles file ../src/libaten/deformable/DeformAnimation.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/deformable/DeformAnimation.d
x64/Debug/libaten/src/libaten/deformable/DeformAnimation.o: ../src/libaten/deformable/DeformAnimation.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/deformable/DeformAnimation.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/deformable/DeformAnimation.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/deformable/DeformAnimation.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/deformable/DeformAnimation.d

# Compiles file ../src/libaten/deformable/DeformAnimationInterp.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/deformable/DeformAnimationInterp.d
x64/Debug/libaten/src/libaten/deformable/DeformAnimationInterp.o: ../src/libaten/deformable/DeformAnimationInterp.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/deformable/DeformAnimationInterp.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/deformable/DeformAnimationInterp.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/deformable/DeformAnimationInterp.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/deformable/DeformAnimationInterp.d

# Compiles file ../src/libaten/os/linux/system_linux.cpp for the Debug configuration...
-include x64/Debug/libaten/src/libaten/os/linux/system_linux.d
x64/Debug/libaten/src/libaten/os/linux/system_linux.o: ../src/libaten/os/linux/system_linux.cpp
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -c ../src/libaten/os/linux/system_linux.cpp $(Debug_Include_Path) -o x64/Debug/libaten/src/libaten/os/linux/system_linux.o
	$(CPP_COMPILER) $(Debug_Preprocessor_Definitions) $(Debug_Compiler_Flags) -MM ../src/libaten/os/linux/system_linux.cpp $(Debug_Include_Path) > x64/Debug/libaten/src/libaten/os/linux/system_linux.d

# Builds the Release configuration...
.PHONY: Release
Release: create_folders x64/Release/libaten/src/libaten/misc/color.o x64/Release/libaten/src/libaten/misc/omputil.o x64/Release/libaten/src/libaten/misc/thread.o x64/Release/libaten/src/libaten/misc/timeline.o x64/Release/libaten/src/libaten/os/linux/misc/timer_linux.o x64/Release/libaten/src/libaten/math/mat4.o x64/Release/libaten/src/libaten/renderer/aov.o x64/Release/libaten/src/libaten/renderer/bdpt.o x64/Release/libaten/src/libaten/renderer/directlight.o x64/Release/libaten/src/libaten/renderer/envmap.o x64/Release/libaten/src/libaten/renderer/erpt.o x64/Release/libaten/src/libaten/renderer/film.o x64/Release/libaten/src/libaten/renderer/nonphotoreal.o x64/Release/libaten/src/libaten/renderer/pathtracing.o x64/Release/libaten/src/libaten/renderer/pssmlt.o x64/Release/libaten/src/libaten/renderer/raytracing.o x64/Release/libaten/src/libaten/renderer/sorted_pathtracing.o x64/Release/libaten/src/libaten/material/beckman.o x64/Release/libaten/src/libaten/material/blinn.o x64/Release/libaten/src/libaten/material/carpaint.o x64/Release/libaten/src/libaten/material/disney_brdf.o x64/Release/libaten/src/libaten/material/FlakesNormal.o x64/Release/libaten/src/libaten/material/ggx.o x64/Release/libaten/src/libaten/material/layer.o x64/Release/libaten/src/libaten/material/material.o x64/Release/libaten/src/libaten/material/oren_nayar.o x64/Release/libaten/src/libaten/material/refraction.o x64/Release/libaten/src/libaten/material/specular.o x64/Release/libaten/src/libaten/material/toon.o x64/Release/libaten/src/libaten/sampler/halton.o x64/Release/libaten/src/libaten/sampler/sampler.o x64/Release/libaten/src/libaten/sampler/sobol.o x64/Release/libaten/src/libaten/camera/CameraOperator.o x64/Release/libaten/src/libaten/camera/equirect.o x64/Release/libaten/src/libaten/camera/pinhole.o x64/Release/libaten/src/libaten/camera/thinlens.o x64/Release/libaten/src/libaten/scene/hitable.o x64/Release/libaten/src/libaten/scene/scene.o x64/Release/libaten/src/libaten/visualizer/blitter.o x64/Release/libaten/src/libaten/visualizer/fbo.o x64/Release/libaten/src/libaten/visualizer/GeomDataBuffer.o x64/Release/libaten/src/libaten/visualizer/GLProfiler.o x64/Release/libaten/src/libaten/visualizer/MultiPassPostProc.o x64/Release/libaten/src/libaten/visualizer/RasterizeRenderer.o x64/Release/libaten/src/libaten/visualizer/shader.o x64/Release/libaten/src/libaten/visualizer/visualizer.o x64/Release/libaten/src/libaten/visualizer/window.o x64/Release/libaten/src/libaten/hdr/gamma.o x64/Release/libaten/src/libaten/hdr/hdr.o x64/Release/libaten/src/libaten/hdr/tonemap.o x64/Release/libaten/src/libaten/texture/texture.o x64/Release/libaten/src/libaten/light/arealight.o x64/Release/libaten/src/libaten/light/ibl.o x64/Release/libaten/src/libaten/light/light.o x64/Release/libaten/src/libaten/posteffect/BloomEffect.o x64/Release/libaten/src/libaten/filter/atrous.o x64/Release/libaten/src/libaten/filter/bilateral.o x64/Release/libaten/src/libaten/filter/nlm.o x64/Release/libaten/src/libaten/filter/taa.o x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.o x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.o x64/Release/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.o x64/Release/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.o x64/Release/libaten/src/libaten/proxy/DataCollector.o x64/Release/libaten/src/libaten/accelerator/accelerator.o x64/Release/libaten/src/libaten/accelerator/bvh.o x64/Release/libaten/src/libaten/accelerator/bvh_frustum.o x64/Release/libaten/src/libaten/accelerator/bvh_update.o x64/Release/libaten/src/libaten/accelerator/CullingFrusta.o x64/Release/libaten/src/libaten/accelerator/qbvh.o x64/Release/libaten/src/libaten/accelerator/sbvh.o x64/Release/libaten/src/libaten/accelerator/sbvh_voxel.o x64/Release/libaten/src/libaten/accelerator/stackless_bvh.o x64/Release/libaten/src/libaten/accelerator/stackless_qbvh.o x64/Release/libaten/src/libaten/accelerator/ThreadedBvhFrustum.o x64/Release/libaten/src/libaten/accelerator/threaded_bvh.o x64/Release/libaten/src/libaten/ui/imgui_impl_glfw_gl3.o x64/Release/libaten/3rdparty/imgui/imgui.o x64/Release/libaten/3rdparty/imgui/imgui_draw.o x64/Release/libaten/src/libaten/geometry/cube.o x64/Release/libaten/src/libaten/geometry/face.o x64/Release/libaten/src/libaten/geometry/geombase.o x64/Release/libaten/src/libaten/geometry/object.o x64/Release/libaten/src/libaten/geometry/objshape.o x64/Release/libaten/src/libaten/geometry/sphere.o x64/Release/libaten/src/libaten/geometry/tranformable.o x64/Release/libaten/src/libaten/geometry/vertex.o x64/Release/libaten/src/libaten/deformable/deformable.o x64/Release/libaten/src/libaten/deformable/DeformMesh.o x64/Release/libaten/src/libaten/deformable/DeformMeshGroup.o x64/Release/libaten/src/libaten/deformable/DeformMeshSet.o x64/Release/libaten/src/libaten/deformable/DeformPrimitives.o x64/Release/libaten/src/libaten/deformable/Skeleton.o x64/Release/libaten/src/libaten/deformable/DeformAnimation.o x64/Release/libaten/src/libaten/deformable/DeformAnimationInterp.o x64/Release/libaten/src/libaten/os/linux/system_linux.o 
	ar rcs x64/Release/libaten.a x64/Release/libaten/src/libaten/misc/color.o x64/Release/libaten/src/libaten/misc/omputil.o x64/Release/libaten/src/libaten/misc/thread.o x64/Release/libaten/src/libaten/misc/timeline.o x64/Release/libaten/src/libaten/os/linux/misc/timer_linux.o x64/Release/libaten/src/libaten/math/mat4.o x64/Release/libaten/src/libaten/renderer/aov.o x64/Release/libaten/src/libaten/renderer/bdpt.o x64/Release/libaten/src/libaten/renderer/directlight.o x64/Release/libaten/src/libaten/renderer/envmap.o x64/Release/libaten/src/libaten/renderer/erpt.o x64/Release/libaten/src/libaten/renderer/film.o x64/Release/libaten/src/libaten/renderer/nonphotoreal.o x64/Release/libaten/src/libaten/renderer/pathtracing.o x64/Release/libaten/src/libaten/renderer/pssmlt.o x64/Release/libaten/src/libaten/renderer/raytracing.o x64/Release/libaten/src/libaten/renderer/sorted_pathtracing.o x64/Release/libaten/src/libaten/material/beckman.o x64/Release/libaten/src/libaten/material/blinn.o x64/Release/libaten/src/libaten/material/carpaint.o x64/Release/libaten/src/libaten/material/disney_brdf.o x64/Release/libaten/src/libaten/material/FlakesNormal.o x64/Release/libaten/src/libaten/material/ggx.o x64/Release/libaten/src/libaten/material/layer.o x64/Release/libaten/src/libaten/material/material.o x64/Release/libaten/src/libaten/material/oren_nayar.o x64/Release/libaten/src/libaten/material/refraction.o x64/Release/libaten/src/libaten/material/specular.o x64/Release/libaten/src/libaten/material/toon.o x64/Release/libaten/src/libaten/sampler/halton.o x64/Release/libaten/src/libaten/sampler/sampler.o x64/Release/libaten/src/libaten/sampler/sobol.o x64/Release/libaten/src/libaten/camera/CameraOperator.o x64/Release/libaten/src/libaten/camera/equirect.o x64/Release/libaten/src/libaten/camera/pinhole.o x64/Release/libaten/src/libaten/camera/thinlens.o x64/Release/libaten/src/libaten/scene/hitable.o x64/Release/libaten/src/libaten/scene/scene.o x64/Release/libaten/src/libaten/visualizer/blitter.o x64/Release/libaten/src/libaten/visualizer/fbo.o x64/Release/libaten/src/libaten/visualizer/GeomDataBuffer.o x64/Release/libaten/src/libaten/visualizer/GLProfiler.o x64/Release/libaten/src/libaten/visualizer/MultiPassPostProc.o x64/Release/libaten/src/libaten/visualizer/RasterizeRenderer.o x64/Release/libaten/src/libaten/visualizer/shader.o x64/Release/libaten/src/libaten/visualizer/visualizer.o x64/Release/libaten/src/libaten/visualizer/window.o x64/Release/libaten/src/libaten/hdr/gamma.o x64/Release/libaten/src/libaten/hdr/hdr.o x64/Release/libaten/src/libaten/hdr/tonemap.o x64/Release/libaten/src/libaten/texture/texture.o x64/Release/libaten/src/libaten/light/arealight.o x64/Release/libaten/src/libaten/light/ibl.o x64/Release/libaten/src/libaten/light/light.o x64/Release/libaten/src/libaten/posteffect/BloomEffect.o x64/Release/libaten/src/libaten/filter/atrous.o x64/Release/libaten/src/libaten/filter/bilateral.o x64/Release/libaten/src/libaten/filter/nlm.o x64/Release/libaten/src/libaten/filter/taa.o x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.o x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.o x64/Release/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.o x64/Release/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.o x64/Release/libaten/src/libaten/proxy/DataCollector.o x64/Release/libaten/src/libaten/accelerator/accelerator.o x64/Release/libaten/src/libaten/accelerator/bvh.o x64/Release/libaten/src/libaten/accelerator/bvh_frustum.o x64/Release/libaten/src/libaten/accelerator/bvh_update.o x64/Release/libaten/src/libaten/accelerator/CullingFrusta.o x64/Release/libaten/src/libaten/accelerator/qbvh.o x64/Release/libaten/src/libaten/accelerator/sbvh.o x64/Release/libaten/src/libaten/accelerator/sbvh_voxel.o x64/Release/libaten/src/libaten/accelerator/stackless_bvh.o x64/Release/libaten/src/libaten/accelerator/stackless_qbvh.o x64/Release/libaten/src/libaten/accelerator/ThreadedBvhFrustum.o x64/Release/libaten/src/libaten/accelerator/threaded_bvh.o x64/Release/libaten/src/libaten/ui/imgui_impl_glfw_gl3.o x64/Release/libaten/3rdparty/imgui/imgui.o x64/Release/libaten/3rdparty/imgui/imgui_draw.o x64/Release/libaten/src/libaten/geometry/cube.o x64/Release/libaten/src/libaten/geometry/face.o x64/Release/libaten/src/libaten/geometry/geombase.o x64/Release/libaten/src/libaten/geometry/object.o x64/Release/libaten/src/libaten/geometry/objshape.o x64/Release/libaten/src/libaten/geometry/sphere.o x64/Release/libaten/src/libaten/geometry/tranformable.o x64/Release/libaten/src/libaten/geometry/vertex.o x64/Release/libaten/src/libaten/deformable/deformable.o x64/Release/libaten/src/libaten/deformable/DeformMesh.o x64/Release/libaten/src/libaten/deformable/DeformMeshGroup.o x64/Release/libaten/src/libaten/deformable/DeformMeshSet.o x64/Release/libaten/src/libaten/deformable/DeformPrimitives.o x64/Release/libaten/src/libaten/deformable/Skeleton.o x64/Release/libaten/src/libaten/deformable/DeformAnimation.o x64/Release/libaten/src/libaten/deformable/DeformAnimationInterp.o x64/Release/libaten/src/libaten/os/linux/system_linux.o  $(Release_Implicitly_Linked_Objects)

# Compiles file ../src/libaten/misc/color.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/misc/color.d
x64/Release/libaten/src/libaten/misc/color.o: ../src/libaten/misc/color.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/misc/color.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/misc/color.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/misc/color.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/misc/color.d

# Compiles file ../src/libaten/misc/omputil.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/misc/omputil.d
x64/Release/libaten/src/libaten/misc/omputil.o: ../src/libaten/misc/omputil.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/misc/omputil.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/misc/omputil.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/misc/omputil.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/misc/omputil.d

# Compiles file ../src/libaten/misc/thread.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/misc/thread.d
x64/Release/libaten/src/libaten/misc/thread.o: ../src/libaten/misc/thread.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/misc/thread.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/misc/thread.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/misc/thread.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/misc/thread.d

# Compiles file ../src/libaten/misc/timeline.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/misc/timeline.d
x64/Release/libaten/src/libaten/misc/timeline.o: ../src/libaten/misc/timeline.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/misc/timeline.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/misc/timeline.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/misc/timeline.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/misc/timeline.d

# Compiles file ../src/libaten/os/linux/misc/timer_linux.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/os/linux/misc/timer_linux.d
x64/Release/libaten/src/libaten/os/linux/misc/timer_linux.o: ../src/libaten/os/linux/misc/timer_linux.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/os/linux/misc/timer_linux.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/os/linux/misc/timer_linux.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/os/linux/misc/timer_linux.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/os/linux/misc/timer_linux.d

# Compiles file ../src/libaten/math/mat4.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/math/mat4.d
x64/Release/libaten/src/libaten/math/mat4.o: ../src/libaten/math/mat4.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/math/mat4.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/math/mat4.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/math/mat4.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/math/mat4.d

# Compiles file ../src/libaten/renderer/aov.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/aov.d
x64/Release/libaten/src/libaten/renderer/aov.o: ../src/libaten/renderer/aov.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/aov.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/aov.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/aov.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/aov.d

# Compiles file ../src/libaten/renderer/bdpt.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/bdpt.d
x64/Release/libaten/src/libaten/renderer/bdpt.o: ../src/libaten/renderer/bdpt.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/bdpt.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/bdpt.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/bdpt.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/bdpt.d

# Compiles file ../src/libaten/renderer/directlight.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/directlight.d
x64/Release/libaten/src/libaten/renderer/directlight.o: ../src/libaten/renderer/directlight.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/directlight.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/directlight.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/directlight.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/directlight.d

# Compiles file ../src/libaten/renderer/envmap.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/envmap.d
x64/Release/libaten/src/libaten/renderer/envmap.o: ../src/libaten/renderer/envmap.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/envmap.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/envmap.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/envmap.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/envmap.d

# Compiles file ../src/libaten/renderer/erpt.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/erpt.d
x64/Release/libaten/src/libaten/renderer/erpt.o: ../src/libaten/renderer/erpt.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/erpt.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/erpt.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/erpt.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/erpt.d

# Compiles file ../src/libaten/renderer/film.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/film.d
x64/Release/libaten/src/libaten/renderer/film.o: ../src/libaten/renderer/film.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/film.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/film.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/film.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/film.d

# Compiles file ../src/libaten/renderer/nonphotoreal.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/nonphotoreal.d
x64/Release/libaten/src/libaten/renderer/nonphotoreal.o: ../src/libaten/renderer/nonphotoreal.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/nonphotoreal.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/nonphotoreal.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/nonphotoreal.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/nonphotoreal.d

# Compiles file ../src/libaten/renderer/pathtracing.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/pathtracing.d
x64/Release/libaten/src/libaten/renderer/pathtracing.o: ../src/libaten/renderer/pathtracing.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/pathtracing.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/pathtracing.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/pathtracing.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/pathtracing.d

# Compiles file ../src/libaten/renderer/pssmlt.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/pssmlt.d
x64/Release/libaten/src/libaten/renderer/pssmlt.o: ../src/libaten/renderer/pssmlt.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/pssmlt.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/pssmlt.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/pssmlt.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/pssmlt.d

# Compiles file ../src/libaten/renderer/raytracing.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/raytracing.d
x64/Release/libaten/src/libaten/renderer/raytracing.o: ../src/libaten/renderer/raytracing.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/raytracing.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/raytracing.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/raytracing.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/raytracing.d

# Compiles file ../src/libaten/renderer/sorted_pathtracing.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/renderer/sorted_pathtracing.d
x64/Release/libaten/src/libaten/renderer/sorted_pathtracing.o: ../src/libaten/renderer/sorted_pathtracing.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/renderer/sorted_pathtracing.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/renderer/sorted_pathtracing.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/renderer/sorted_pathtracing.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/renderer/sorted_pathtracing.d

# Compiles file ../src/libaten/material/beckman.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/beckman.d
x64/Release/libaten/src/libaten/material/beckman.o: ../src/libaten/material/beckman.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/beckman.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/beckman.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/beckman.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/beckman.d

# Compiles file ../src/libaten/material/blinn.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/blinn.d
x64/Release/libaten/src/libaten/material/blinn.o: ../src/libaten/material/blinn.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/blinn.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/blinn.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/blinn.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/blinn.d

# Compiles file ../src/libaten/material/carpaint.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/carpaint.d
x64/Release/libaten/src/libaten/material/carpaint.o: ../src/libaten/material/carpaint.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/carpaint.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/carpaint.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/carpaint.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/carpaint.d

# Compiles file ../src/libaten/material/disney_brdf.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/disney_brdf.d
x64/Release/libaten/src/libaten/material/disney_brdf.o: ../src/libaten/material/disney_brdf.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/disney_brdf.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/disney_brdf.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/disney_brdf.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/disney_brdf.d

# Compiles file ../src/libaten/material/FlakesNormal.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/FlakesNormal.d
x64/Release/libaten/src/libaten/material/FlakesNormal.o: ../src/libaten/material/FlakesNormal.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/FlakesNormal.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/FlakesNormal.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/FlakesNormal.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/FlakesNormal.d

# Compiles file ../src/libaten/material/ggx.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/ggx.d
x64/Release/libaten/src/libaten/material/ggx.o: ../src/libaten/material/ggx.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/ggx.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/ggx.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/ggx.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/ggx.d

# Compiles file ../src/libaten/material/layer.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/layer.d
x64/Release/libaten/src/libaten/material/layer.o: ../src/libaten/material/layer.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/layer.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/layer.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/layer.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/layer.d

# Compiles file ../src/libaten/material/material.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/material.d
x64/Release/libaten/src/libaten/material/material.o: ../src/libaten/material/material.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/material.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/material.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/material.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/material.d

# Compiles file ../src/libaten/material/oren_nayar.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/oren_nayar.d
x64/Release/libaten/src/libaten/material/oren_nayar.o: ../src/libaten/material/oren_nayar.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/oren_nayar.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/oren_nayar.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/oren_nayar.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/oren_nayar.d

# Compiles file ../src/libaten/material/refraction.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/refraction.d
x64/Release/libaten/src/libaten/material/refraction.o: ../src/libaten/material/refraction.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/refraction.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/refraction.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/refraction.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/refraction.d

# Compiles file ../src/libaten/material/specular.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/specular.d
x64/Release/libaten/src/libaten/material/specular.o: ../src/libaten/material/specular.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/specular.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/specular.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/specular.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/specular.d

# Compiles file ../src/libaten/material/toon.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/material/toon.d
x64/Release/libaten/src/libaten/material/toon.o: ../src/libaten/material/toon.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/material/toon.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/material/toon.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/material/toon.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/material/toon.d

# Compiles file ../src/libaten/sampler/halton.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/sampler/halton.d
x64/Release/libaten/src/libaten/sampler/halton.o: ../src/libaten/sampler/halton.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/sampler/halton.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/sampler/halton.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/sampler/halton.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/sampler/halton.d

# Compiles file ../src/libaten/sampler/sampler.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/sampler/sampler.d
x64/Release/libaten/src/libaten/sampler/sampler.o: ../src/libaten/sampler/sampler.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/sampler/sampler.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/sampler/sampler.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/sampler/sampler.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/sampler/sampler.d

# Compiles file ../src/libaten/sampler/sobol.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/sampler/sobol.d
x64/Release/libaten/src/libaten/sampler/sobol.o: ../src/libaten/sampler/sobol.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/sampler/sobol.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/sampler/sobol.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/sampler/sobol.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/sampler/sobol.d

# Compiles file ../src/libaten/camera/CameraOperator.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/camera/CameraOperator.d
x64/Release/libaten/src/libaten/camera/CameraOperator.o: ../src/libaten/camera/CameraOperator.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/camera/CameraOperator.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/camera/CameraOperator.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/camera/CameraOperator.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/camera/CameraOperator.d

# Compiles file ../src/libaten/camera/equirect.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/camera/equirect.d
x64/Release/libaten/src/libaten/camera/equirect.o: ../src/libaten/camera/equirect.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/camera/equirect.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/camera/equirect.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/camera/equirect.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/camera/equirect.d

# Compiles file ../src/libaten/camera/pinhole.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/camera/pinhole.d
x64/Release/libaten/src/libaten/camera/pinhole.o: ../src/libaten/camera/pinhole.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/camera/pinhole.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/camera/pinhole.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/camera/pinhole.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/camera/pinhole.d

# Compiles file ../src/libaten/camera/thinlens.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/camera/thinlens.d
x64/Release/libaten/src/libaten/camera/thinlens.o: ../src/libaten/camera/thinlens.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/camera/thinlens.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/camera/thinlens.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/camera/thinlens.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/camera/thinlens.d

# Compiles file ../src/libaten/scene/hitable.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/scene/hitable.d
x64/Release/libaten/src/libaten/scene/hitable.o: ../src/libaten/scene/hitable.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/scene/hitable.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/scene/hitable.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/scene/hitable.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/scene/hitable.d

# Compiles file ../src/libaten/scene/scene.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/scene/scene.d
x64/Release/libaten/src/libaten/scene/scene.o: ../src/libaten/scene/scene.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/scene/scene.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/scene/scene.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/scene/scene.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/scene/scene.d

# Compiles file ../src/libaten/visualizer/blitter.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/visualizer/blitter.d
x64/Release/libaten/src/libaten/visualizer/blitter.o: ../src/libaten/visualizer/blitter.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/visualizer/blitter.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/visualizer/blitter.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/visualizer/blitter.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/visualizer/blitter.d

# Compiles file ../src/libaten/visualizer/fbo.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/visualizer/fbo.d
x64/Release/libaten/src/libaten/visualizer/fbo.o: ../src/libaten/visualizer/fbo.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/visualizer/fbo.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/visualizer/fbo.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/visualizer/fbo.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/visualizer/fbo.d

# Compiles file ../src/libaten/visualizer/GeomDataBuffer.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/visualizer/GeomDataBuffer.d
x64/Release/libaten/src/libaten/visualizer/GeomDataBuffer.o: ../src/libaten/visualizer/GeomDataBuffer.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/visualizer/GeomDataBuffer.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/visualizer/GeomDataBuffer.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/visualizer/GeomDataBuffer.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/visualizer/GeomDataBuffer.d

# Compiles file ../src/libaten/visualizer/GLProfiler.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/visualizer/GLProfiler.d
x64/Release/libaten/src/libaten/visualizer/GLProfiler.o: ../src/libaten/visualizer/GLProfiler.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/visualizer/GLProfiler.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/visualizer/GLProfiler.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/visualizer/GLProfiler.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/visualizer/GLProfiler.d

# Compiles file ../src/libaten/visualizer/MultiPassPostProc.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/visualizer/MultiPassPostProc.d
x64/Release/libaten/src/libaten/visualizer/MultiPassPostProc.o: ../src/libaten/visualizer/MultiPassPostProc.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/visualizer/MultiPassPostProc.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/visualizer/MultiPassPostProc.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/visualizer/MultiPassPostProc.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/visualizer/MultiPassPostProc.d

# Compiles file ../src/libaten/visualizer/RasterizeRenderer.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/visualizer/RasterizeRenderer.d
x64/Release/libaten/src/libaten/visualizer/RasterizeRenderer.o: ../src/libaten/visualizer/RasterizeRenderer.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/visualizer/RasterizeRenderer.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/visualizer/RasterizeRenderer.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/visualizer/RasterizeRenderer.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/visualizer/RasterizeRenderer.d

# Compiles file ../src/libaten/visualizer/shader.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/visualizer/shader.d
x64/Release/libaten/src/libaten/visualizer/shader.o: ../src/libaten/visualizer/shader.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/visualizer/shader.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/visualizer/shader.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/visualizer/shader.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/visualizer/shader.d

# Compiles file ../src/libaten/visualizer/visualizer.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/visualizer/visualizer.d
x64/Release/libaten/src/libaten/visualizer/visualizer.o: ../src/libaten/visualizer/visualizer.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/visualizer/visualizer.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/visualizer/visualizer.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/visualizer/visualizer.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/visualizer/visualizer.d

# Compiles file ../src/libaten/visualizer/window.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/visualizer/window.d
x64/Release/libaten/src/libaten/visualizer/window.o: ../src/libaten/visualizer/window.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/visualizer/window.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/visualizer/window.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/visualizer/window.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/visualizer/window.d

# Compiles file ../src/libaten/hdr/gamma.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/hdr/gamma.d
x64/Release/libaten/src/libaten/hdr/gamma.o: ../src/libaten/hdr/gamma.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/hdr/gamma.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/hdr/gamma.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/hdr/gamma.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/hdr/gamma.d

# Compiles file ../src/libaten/hdr/hdr.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/hdr/hdr.d
x64/Release/libaten/src/libaten/hdr/hdr.o: ../src/libaten/hdr/hdr.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/hdr/hdr.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/hdr/hdr.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/hdr/hdr.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/hdr/hdr.d

# Compiles file ../src/libaten/hdr/tonemap.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/hdr/tonemap.d
x64/Release/libaten/src/libaten/hdr/tonemap.o: ../src/libaten/hdr/tonemap.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/hdr/tonemap.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/hdr/tonemap.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/hdr/tonemap.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/hdr/tonemap.d

# Compiles file ../src/libaten/texture/texture.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/texture/texture.d
x64/Release/libaten/src/libaten/texture/texture.o: ../src/libaten/texture/texture.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/texture/texture.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/texture/texture.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/texture/texture.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/texture/texture.d

# Compiles file ../src/libaten/light/arealight.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/light/arealight.d
x64/Release/libaten/src/libaten/light/arealight.o: ../src/libaten/light/arealight.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/light/arealight.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/light/arealight.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/light/arealight.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/light/arealight.d

# Compiles file ../src/libaten/light/ibl.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/light/ibl.d
x64/Release/libaten/src/libaten/light/ibl.o: ../src/libaten/light/ibl.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/light/ibl.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/light/ibl.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/light/ibl.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/light/ibl.d

# Compiles file ../src/libaten/light/light.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/light/light.d
x64/Release/libaten/src/libaten/light/light.o: ../src/libaten/light/light.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/light/light.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/light/light.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/light/light.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/light/light.d

# Compiles file ../src/libaten/posteffect/BloomEffect.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/posteffect/BloomEffect.d
x64/Release/libaten/src/libaten/posteffect/BloomEffect.o: ../src/libaten/posteffect/BloomEffect.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/posteffect/BloomEffect.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/posteffect/BloomEffect.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/posteffect/BloomEffect.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/posteffect/BloomEffect.d

# Compiles file ../src/libaten/filter/atrous.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/filter/atrous.d
x64/Release/libaten/src/libaten/filter/atrous.o: ../src/libaten/filter/atrous.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/filter/atrous.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/filter/atrous.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/filter/atrous.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/filter/atrous.d

# Compiles file ../src/libaten/filter/bilateral.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/filter/bilateral.d
x64/Release/libaten/src/libaten/filter/bilateral.o: ../src/libaten/filter/bilateral.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/filter/bilateral.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/filter/bilateral.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/filter/bilateral.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/filter/bilateral.d

# Compiles file ../src/libaten/filter/nlm.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/filter/nlm.d
x64/Release/libaten/src/libaten/filter/nlm.o: ../src/libaten/filter/nlm.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/filter/nlm.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/filter/nlm.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/filter/nlm.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/filter/nlm.d

# Compiles file ../src/libaten/filter/taa.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/filter/taa.d
x64/Release/libaten/src/libaten/filter/taa.o: ../src/libaten/filter/taa.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/filter/taa.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/filter/taa.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/filter/taa.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/filter/taa.d

# Compiles file ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.d
x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.o: ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReduction.d

# Compiles file ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.d
x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.o: ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/PracticalNoiseReductionBilateral.d

# Compiles file ../src/libaten/filter/VirtualFlashImage/VirtualFlashImage.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.d
x64/Release/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.o: ../src/libaten/filter/VirtualFlashImage/VirtualFlashImage.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/filter/VirtualFlashImage/VirtualFlashImage.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/filter/VirtualFlashImage/VirtualFlashImage.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/filter/VirtualFlashImage/VirtualFlashImage.d

# Compiles file ../src/libaten/filter/GeometryRendering/GeometryRendering.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.d
x64/Release/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.o: ../src/libaten/filter/GeometryRendering/GeometryRendering.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/filter/GeometryRendering/GeometryRendering.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/filter/GeometryRendering/GeometryRendering.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/filter/GeometryRendering/GeometryRendering.d

# Compiles file ../src/libaten/proxy/DataCollector.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/proxy/DataCollector.d
x64/Release/libaten/src/libaten/proxy/DataCollector.o: ../src/libaten/proxy/DataCollector.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/proxy/DataCollector.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/proxy/DataCollector.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/proxy/DataCollector.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/proxy/DataCollector.d

# Compiles file ../src/libaten/accelerator/accelerator.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/accelerator.d
x64/Release/libaten/src/libaten/accelerator/accelerator.o: ../src/libaten/accelerator/accelerator.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/accelerator.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/accelerator.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/accelerator.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/accelerator.d

# Compiles file ../src/libaten/accelerator/bvh.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/bvh.d
x64/Release/libaten/src/libaten/accelerator/bvh.o: ../src/libaten/accelerator/bvh.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/bvh.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/bvh.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/bvh.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/bvh.d

# Compiles file ../src/libaten/accelerator/bvh_frustum.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/bvh_frustum.d
x64/Release/libaten/src/libaten/accelerator/bvh_frustum.o: ../src/libaten/accelerator/bvh_frustum.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/bvh_frustum.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/bvh_frustum.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/bvh_frustum.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/bvh_frustum.d

# Compiles file ../src/libaten/accelerator/bvh_update.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/bvh_update.d
x64/Release/libaten/src/libaten/accelerator/bvh_update.o: ../src/libaten/accelerator/bvh_update.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/bvh_update.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/bvh_update.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/bvh_update.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/bvh_update.d

# Compiles file ../src/libaten/accelerator/CullingFrusta.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/CullingFrusta.d
x64/Release/libaten/src/libaten/accelerator/CullingFrusta.o: ../src/libaten/accelerator/CullingFrusta.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/CullingFrusta.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/CullingFrusta.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/CullingFrusta.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/CullingFrusta.d

# Compiles file ../src/libaten/accelerator/qbvh.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/qbvh.d
x64/Release/libaten/src/libaten/accelerator/qbvh.o: ../src/libaten/accelerator/qbvh.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/qbvh.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/qbvh.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/qbvh.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/qbvh.d

# Compiles file ../src/libaten/accelerator/sbvh.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/sbvh.d
x64/Release/libaten/src/libaten/accelerator/sbvh.o: ../src/libaten/accelerator/sbvh.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/sbvh.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/sbvh.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/sbvh.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/sbvh.d

# Compiles file ../src/libaten/accelerator/sbvh_voxel.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/sbvh_voxel.d
x64/Release/libaten/src/libaten/accelerator/sbvh_voxel.o: ../src/libaten/accelerator/sbvh_voxel.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/sbvh_voxel.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/sbvh_voxel.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/sbvh_voxel.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/sbvh_voxel.d

# Compiles file ../src/libaten/accelerator/stackless_bvh.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/stackless_bvh.d
x64/Release/libaten/src/libaten/accelerator/stackless_bvh.o: ../src/libaten/accelerator/stackless_bvh.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/stackless_bvh.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/stackless_bvh.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/stackless_bvh.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/stackless_bvh.d

# Compiles file ../src/libaten/accelerator/stackless_qbvh.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/stackless_qbvh.d
x64/Release/libaten/src/libaten/accelerator/stackless_qbvh.o: ../src/libaten/accelerator/stackless_qbvh.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/stackless_qbvh.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/stackless_qbvh.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/stackless_qbvh.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/stackless_qbvh.d

# Compiles file ../src/libaten/accelerator/ThreadedBvhFrustum.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/ThreadedBvhFrustum.d
x64/Release/libaten/src/libaten/accelerator/ThreadedBvhFrustum.o: ../src/libaten/accelerator/ThreadedBvhFrustum.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/ThreadedBvhFrustum.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/ThreadedBvhFrustum.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/ThreadedBvhFrustum.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/ThreadedBvhFrustum.d

# Compiles file ../src/libaten/accelerator/threaded_bvh.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/accelerator/threaded_bvh.d
x64/Release/libaten/src/libaten/accelerator/threaded_bvh.o: ../src/libaten/accelerator/threaded_bvh.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/accelerator/threaded_bvh.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/accelerator/threaded_bvh.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/accelerator/threaded_bvh.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/accelerator/threaded_bvh.d

# Compiles file ../src/libaten/ui/imgui_impl_glfw_gl3.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/ui/imgui_impl_glfw_gl3.d
x64/Release/libaten/src/libaten/ui/imgui_impl_glfw_gl3.o: ../src/libaten/ui/imgui_impl_glfw_gl3.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/ui/imgui_impl_glfw_gl3.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/ui/imgui_impl_glfw_gl3.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/ui/imgui_impl_glfw_gl3.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/ui/imgui_impl_glfw_gl3.d

# Compiles file ../3rdparty/imgui/imgui.cpp for the Release configuration...
-include x64/Release/libaten/3rdparty/imgui/imgui.d
x64/Release/libaten/3rdparty/imgui/imgui.o: ../3rdparty/imgui/imgui.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../3rdparty/imgui/imgui.cpp $(Release_Include_Path) -o x64/Release/libaten/3rdparty/imgui/imgui.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../3rdparty/imgui/imgui.cpp $(Release_Include_Path) > x64/Release/libaten/3rdparty/imgui/imgui.d

# Compiles file ../3rdparty/imgui/imgui_draw.cpp for the Release configuration...
-include x64/Release/libaten/3rdparty/imgui/imgui_draw.d
x64/Release/libaten/3rdparty/imgui/imgui_draw.o: ../3rdparty/imgui/imgui_draw.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../3rdparty/imgui/imgui_draw.cpp $(Release_Include_Path) -o x64/Release/libaten/3rdparty/imgui/imgui_draw.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../3rdparty/imgui/imgui_draw.cpp $(Release_Include_Path) > x64/Release/libaten/3rdparty/imgui/imgui_draw.d

# Compiles file ../src/libaten/geometry/cube.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/geometry/cube.d
x64/Release/libaten/src/libaten/geometry/cube.o: ../src/libaten/geometry/cube.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/geometry/cube.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/geometry/cube.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/geometry/cube.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/geometry/cube.d

# Compiles file ../src/libaten/geometry/face.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/geometry/face.d
x64/Release/libaten/src/libaten/geometry/face.o: ../src/libaten/geometry/face.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/geometry/face.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/geometry/face.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/geometry/face.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/geometry/face.d

# Compiles file ../src/libaten/geometry/geombase.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/geometry/geombase.d
x64/Release/libaten/src/libaten/geometry/geombase.o: ../src/libaten/geometry/geombase.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/geometry/geombase.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/geometry/geombase.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/geometry/geombase.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/geometry/geombase.d

# Compiles file ../src/libaten/geometry/object.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/geometry/object.d
x64/Release/libaten/src/libaten/geometry/object.o: ../src/libaten/geometry/object.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/geometry/object.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/geometry/object.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/geometry/object.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/geometry/object.d

# Compiles file ../src/libaten/geometry/objshape.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/geometry/objshape.d
x64/Release/libaten/src/libaten/geometry/objshape.o: ../src/libaten/geometry/objshape.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/geometry/objshape.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/geometry/objshape.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/geometry/objshape.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/geometry/objshape.d

# Compiles file ../src/libaten/geometry/sphere.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/geometry/sphere.d
x64/Release/libaten/src/libaten/geometry/sphere.o: ../src/libaten/geometry/sphere.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/geometry/sphere.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/geometry/sphere.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/geometry/sphere.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/geometry/sphere.d

# Compiles file ../src/libaten/geometry/tranformable.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/geometry/tranformable.d
x64/Release/libaten/src/libaten/geometry/tranformable.o: ../src/libaten/geometry/tranformable.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/geometry/tranformable.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/geometry/tranformable.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/geometry/tranformable.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/geometry/tranformable.d

# Compiles file ../src/libaten/geometry/vertex.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/geometry/vertex.d
x64/Release/libaten/src/libaten/geometry/vertex.o: ../src/libaten/geometry/vertex.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/geometry/vertex.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/geometry/vertex.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/geometry/vertex.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/geometry/vertex.d

# Compiles file ../src/libaten/deformable/deformable.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/deformable/deformable.d
x64/Release/libaten/src/libaten/deformable/deformable.o: ../src/libaten/deformable/deformable.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/deformable/deformable.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/deformable/deformable.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/deformable/deformable.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/deformable/deformable.d

# Compiles file ../src/libaten/deformable/DeformMesh.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/deformable/DeformMesh.d
x64/Release/libaten/src/libaten/deformable/DeformMesh.o: ../src/libaten/deformable/DeformMesh.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/deformable/DeformMesh.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/deformable/DeformMesh.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/deformable/DeformMesh.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/deformable/DeformMesh.d

# Compiles file ../src/libaten/deformable/DeformMeshGroup.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/deformable/DeformMeshGroup.d
x64/Release/libaten/src/libaten/deformable/DeformMeshGroup.o: ../src/libaten/deformable/DeformMeshGroup.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/deformable/DeformMeshGroup.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/deformable/DeformMeshGroup.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/deformable/DeformMeshGroup.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/deformable/DeformMeshGroup.d

# Compiles file ../src/libaten/deformable/DeformMeshSet.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/deformable/DeformMeshSet.d
x64/Release/libaten/src/libaten/deformable/DeformMeshSet.o: ../src/libaten/deformable/DeformMeshSet.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/deformable/DeformMeshSet.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/deformable/DeformMeshSet.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/deformable/DeformMeshSet.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/deformable/DeformMeshSet.d

# Compiles file ../src/libaten/deformable/DeformPrimitives.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/deformable/DeformPrimitives.d
x64/Release/libaten/src/libaten/deformable/DeformPrimitives.o: ../src/libaten/deformable/DeformPrimitives.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/deformable/DeformPrimitives.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/deformable/DeformPrimitives.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/deformable/DeformPrimitives.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/deformable/DeformPrimitives.d

# Compiles file ../src/libaten/deformable/Skeleton.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/deformable/Skeleton.d
x64/Release/libaten/src/libaten/deformable/Skeleton.o: ../src/libaten/deformable/Skeleton.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/deformable/Skeleton.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/deformable/Skeleton.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/deformable/Skeleton.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/deformable/Skeleton.d

# Compiles file ../src/libaten/deformable/DeformAnimation.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/deformable/DeformAnimation.d
x64/Release/libaten/src/libaten/deformable/DeformAnimation.o: ../src/libaten/deformable/DeformAnimation.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/deformable/DeformAnimation.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/deformable/DeformAnimation.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/deformable/DeformAnimation.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/deformable/DeformAnimation.d

# Compiles file ../src/libaten/deformable/DeformAnimationInterp.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/deformable/DeformAnimationInterp.d
x64/Release/libaten/src/libaten/deformable/DeformAnimationInterp.o: ../src/libaten/deformable/DeformAnimationInterp.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/deformable/DeformAnimationInterp.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/deformable/DeformAnimationInterp.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/deformable/DeformAnimationInterp.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/deformable/DeformAnimationInterp.d

# Compiles file ../src/libaten/os/linux/system_linux.cpp for the Release configuration...
-include x64/Release/libaten/src/libaten/os/linux/system_linux.d
x64/Release/libaten/src/libaten/os/linux/system_linux.o: ../src/libaten/os/linux/system_linux.cpp
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -c ../src/libaten/os/linux/system_linux.cpp $(Release_Include_Path) -o x64/Release/libaten/src/libaten/os/linux/system_linux.o
	$(CPP_COMPILER) $(Release_Preprocessor_Definitions) $(Release_Compiler_Flags) -MM ../src/libaten/os/linux/system_linux.cpp $(Release_Include_Path) > x64/Release/libaten/src/libaten/os/linux/system_linux.d

# Creates the intermediate and output folders for each configuration...
.PHONY: create_folders
create_folders:
	mkdir -p x64/Debug/libaten/src/libaten/misc
	mkdir -p x64/Debug/libaten/src/libaten/os/linux/misc
	mkdir -p x64/Debug/libaten/src/libaten/math
	mkdir -p x64/Debug/libaten/src/libaten/renderer
	mkdir -p x64/Debug/libaten/src/libaten/material
	mkdir -p x64/Debug/libaten/src/libaten/sampler
	mkdir -p x64/Debug/libaten/src/libaten/camera
	mkdir -p x64/Debug/libaten/src/libaten/scene
	mkdir -p x64/Debug/libaten/src/libaten/visualizer
	mkdir -p x64/Debug/libaten/src/libaten/hdr
	mkdir -p x64/Debug/libaten/src/libaten/texture
	mkdir -p x64/Debug/libaten/src/libaten/light
	mkdir -p x64/Debug/libaten/src/libaten/posteffect
	mkdir -p x64/Debug/libaten/src/libaten/filter
	mkdir -p x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction
	mkdir -p x64/Debug/libaten/src/libaten/filter/VirtualFlashImage
	mkdir -p x64/Debug/libaten/src/libaten/filter/GeometryRendering
	mkdir -p x64/Debug/libaten/src/libaten/proxy
	mkdir -p x64/Debug/libaten/src/libaten/accelerator
	mkdir -p x64/Debug/libaten/src/libaten/ui
	mkdir -p x64/Debug/libaten/3rdparty/imgui
	mkdir -p x64/Debug/libaten/src/libaten/geometry
	mkdir -p x64/Debug/libaten/src/libaten/deformable
	mkdir -p x64/Debug/libaten/src/libaten/os/linux
	mkdir -p x64/Debug
	mkdir -p x64/Release/libaten/src/libaten/misc
	mkdir -p x64/Release/libaten/src/libaten/os/linux/misc
	mkdir -p x64/Release/libaten/src/libaten/math
	mkdir -p x64/Release/libaten/src/libaten/renderer
	mkdir -p x64/Release/libaten/src/libaten/material
	mkdir -p x64/Release/libaten/src/libaten/sampler
	mkdir -p x64/Release/libaten/src/libaten/camera
	mkdir -p x64/Release/libaten/src/libaten/scene
	mkdir -p x64/Release/libaten/src/libaten/visualizer
	mkdir -p x64/Release/libaten/src/libaten/hdr
	mkdir -p x64/Release/libaten/src/libaten/texture
	mkdir -p x64/Release/libaten/src/libaten/light
	mkdir -p x64/Release/libaten/src/libaten/posteffect
	mkdir -p x64/Release/libaten/src/libaten/filter
	mkdir -p x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction
	mkdir -p x64/Release/libaten/src/libaten/filter/VirtualFlashImage
	mkdir -p x64/Release/libaten/src/libaten/filter/GeometryRendering
	mkdir -p x64/Release/libaten/src/libaten/proxy
	mkdir -p x64/Release/libaten/src/libaten/accelerator
	mkdir -p x64/Release/libaten/src/libaten/ui
	mkdir -p x64/Release/libaten/3rdparty/imgui
	mkdir -p x64/Release/libaten/src/libaten/geometry
	mkdir -p x64/Release/libaten/src/libaten/deformable
	mkdir -p x64/Release/libaten/src/libaten/os/linux
	mkdir -p x64/Release

# Cleans intermediate and output files (objects, libraries, executables)...
.PHONY: clean
clean:
	rm -f x64/Debug/libaten/src/libaten/misc/*.o
	rm -f x64/Debug/libaten/src/libaten/misc/*.d
	rm -f x64/Debug/libaten/src/libaten/os/linux/misc/*.o
	rm -f x64/Debug/libaten/src/libaten/os/linux/misc/*.d
	rm -f x64/Debug/libaten/src/libaten/math/*.o
	rm -f x64/Debug/libaten/src/libaten/math/*.d
	rm -f x64/Debug/libaten/src/libaten/renderer/*.o
	rm -f x64/Debug/libaten/src/libaten/renderer/*.d
	rm -f x64/Debug/libaten/src/libaten/material/*.o
	rm -f x64/Debug/libaten/src/libaten/material/*.d
	rm -f x64/Debug/libaten/src/libaten/sampler/*.o
	rm -f x64/Debug/libaten/src/libaten/sampler/*.d
	rm -f x64/Debug/libaten/src/libaten/camera/*.o
	rm -f x64/Debug/libaten/src/libaten/camera/*.d
	rm -f x64/Debug/libaten/src/libaten/scene/*.o
	rm -f x64/Debug/libaten/src/libaten/scene/*.d
	rm -f x64/Debug/libaten/src/libaten/visualizer/*.o
	rm -f x64/Debug/libaten/src/libaten/visualizer/*.d
	rm -f x64/Debug/libaten/src/libaten/hdr/*.o
	rm -f x64/Debug/libaten/src/libaten/hdr/*.d
	rm -f x64/Debug/libaten/src/libaten/texture/*.o
	rm -f x64/Debug/libaten/src/libaten/texture/*.d
	rm -f x64/Debug/libaten/src/libaten/light/*.o
	rm -f x64/Debug/libaten/src/libaten/light/*.d
	rm -f x64/Debug/libaten/src/libaten/posteffect/*.o
	rm -f x64/Debug/libaten/src/libaten/posteffect/*.d
	rm -f x64/Debug/libaten/src/libaten/filter/*.o
	rm -f x64/Debug/libaten/src/libaten/filter/*.d
	rm -f x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/*.o
	rm -f x64/Debug/libaten/src/libaten/filter/PracticalNoiseReduction/*.d
	rm -f x64/Debug/libaten/src/libaten/filter/VirtualFlashImage/*.o
	rm -f x64/Debug/libaten/src/libaten/filter/VirtualFlashImage/*.d
	rm -f x64/Debug/libaten/src/libaten/filter/GeometryRendering/*.o
	rm -f x64/Debug/libaten/src/libaten/filter/GeometryRendering/*.d
	rm -f x64/Debug/libaten/src/libaten/proxy/*.o
	rm -f x64/Debug/libaten/src/libaten/proxy/*.d
	rm -f x64/Debug/libaten/src/libaten/accelerator/*.o
	rm -f x64/Debug/libaten/src/libaten/accelerator/*.d
	rm -f x64/Debug/libaten/src/libaten/ui/*.o
	rm -f x64/Debug/libaten/src/libaten/ui/*.d
	rm -f x64/Debug/libaten/3rdparty/imgui/*.o
	rm -f x64/Debug/libaten/3rdparty/imgui/*.d
	rm -f x64/Debug/libaten/src/libaten/geometry/*.o
	rm -f x64/Debug/libaten/src/libaten/geometry/*.d
	rm -f x64/Debug/libaten/src/libaten/deformable/*.o
	rm -f x64/Debug/libaten/src/libaten/deformable/*.d
	rm -f x64/Debug/libaten/src/libaten/os/linux/*.o
	rm -f x64/Debug/libaten/src/libaten/os/linux/*.d
	rm -f x64/Debug/libaten.a
	rm -f x64/Release/libaten/src/libaten/misc/*.o
	rm -f x64/Release/libaten/src/libaten/misc/*.d
	rm -f x64/Release/libaten/src/libaten/os/linux/misc/*.o
	rm -f x64/Release/libaten/src/libaten/os/linux/misc/*.d
	rm -f x64/Release/libaten/src/libaten/math/*.o
	rm -f x64/Release/libaten/src/libaten/math/*.d
	rm -f x64/Release/libaten/src/libaten/renderer/*.o
	rm -f x64/Release/libaten/src/libaten/renderer/*.d
	rm -f x64/Release/libaten/src/libaten/material/*.o
	rm -f x64/Release/libaten/src/libaten/material/*.d
	rm -f x64/Release/libaten/src/libaten/sampler/*.o
	rm -f x64/Release/libaten/src/libaten/sampler/*.d
	rm -f x64/Release/libaten/src/libaten/camera/*.o
	rm -f x64/Release/libaten/src/libaten/camera/*.d
	rm -f x64/Release/libaten/src/libaten/scene/*.o
	rm -f x64/Release/libaten/src/libaten/scene/*.d
	rm -f x64/Release/libaten/src/libaten/visualizer/*.o
	rm -f x64/Release/libaten/src/libaten/visualizer/*.d
	rm -f x64/Release/libaten/src/libaten/hdr/*.o
	rm -f x64/Release/libaten/src/libaten/hdr/*.d
	rm -f x64/Release/libaten/src/libaten/texture/*.o
	rm -f x64/Release/libaten/src/libaten/texture/*.d
	rm -f x64/Release/libaten/src/libaten/light/*.o
	rm -f x64/Release/libaten/src/libaten/light/*.d
	rm -f x64/Release/libaten/src/libaten/posteffect/*.o
	rm -f x64/Release/libaten/src/libaten/posteffect/*.d
	rm -f x64/Release/libaten/src/libaten/filter/*.o
	rm -f x64/Release/libaten/src/libaten/filter/*.d
	rm -f x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/*.o
	rm -f x64/Release/libaten/src/libaten/filter/PracticalNoiseReduction/*.d
	rm -f x64/Release/libaten/src/libaten/filter/VirtualFlashImage/*.o
	rm -f x64/Release/libaten/src/libaten/filter/VirtualFlashImage/*.d
	rm -f x64/Release/libaten/src/libaten/filter/GeometryRendering/*.o
	rm -f x64/Release/libaten/src/libaten/filter/GeometryRendering/*.d
	rm -f x64/Release/libaten/src/libaten/proxy/*.o
	rm -f x64/Release/libaten/src/libaten/proxy/*.d
	rm -f x64/Release/libaten/src/libaten/accelerator/*.o
	rm -f x64/Release/libaten/src/libaten/accelerator/*.d
	rm -f x64/Release/libaten/src/libaten/ui/*.o
	rm -f x64/Release/libaten/src/libaten/ui/*.d
	rm -f x64/Release/libaten/3rdparty/imgui/*.o
	rm -f x64/Release/libaten/3rdparty/imgui/*.d
	rm -f x64/Release/libaten/src/libaten/geometry/*.o
	rm -f x64/Release/libaten/src/libaten/geometry/*.d
	rm -f x64/Release/libaten/src/libaten/deformable/*.o
	rm -f x64/Release/libaten/src/libaten/deformable/*.d
	rm -f x64/Release/libaten/src/libaten/os/linux/*.o
	rm -f x64/Release/libaten/src/libaten/os/linux/*.d
	rm -f x64/Release/libaten.a

