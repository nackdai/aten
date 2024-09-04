set(PROJECT_NAME idaten)

project(${PROJECT_NAME})

find_package(CUDA REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_VERBOSE_BUILD ON)

message("-- CUDA --")
message("Version: " ${CUDA_VERSION})
message("Library: " ${CUDA_CUDA_LIBRARY})
message("Runtime: " ${CUDA_CUDART_LIBRARY})

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -rdc=true
    -use_fast_math
    --expt-relaxed-constexpr
    --extended-lambda
    -std=c++17
    -gencode
    arch=compute_${CUDA_TARGET_COMPUTE_CAPABILITY},code=sm_${CUDA_TARGET_COMPUTE_CAPABILITY}
    -Xcompiler=-std=c++17)

message("nvcc_flags " ${CUDA_NVCC_FLAGS})

add_definitions(-D__AT_CUDA__)

set_source_files_properties(
  ../libaten/camera/pinhole.cpp
  ../libaten/geometry/sphere.cpp
  ../libaten/material/FlakesNormal.cpp
  ../libaten/material/beckman.cpp
  ../libaten/material/car_paint.cpp
  ../libaten/material/disney_brdf.cpp
  ../libaten/material/ggx.cpp
  ../libaten/material/material.cpp
  ../libaten/material/microfacet_refraction.cpp
  ../libaten/material/oren_nayar.cpp
  ../libaten/material/refraction.cpp
  ../libaten/material/retroreflective.cpp
  ../libaten/material/specular.cpp
  ../libaten/material/velvet.cpp
  ../libaten/volume/grid.cpp
  ../libaten/volume/medium.cpp
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# Add library to build.
cuda_add_library(
  ${PROJECT_NAME}
  STATIC
  ../libaten/camera/camera.h
  ../libaten/camera/pinhole.cpp
  ../libaten/camera/pinhole.h
  ../libaten/defs.h
  ../libaten/geometry/EvaluateHitResult.h
  ../libaten/geometry/PolygonObject.h
  ../libaten/geometry/sphere.cpp
  ../libaten/geometry/sphere.h
  ../libaten/geometry/triangle.h
  ../libaten/light/arealight.h
  ../libaten/light/directionallight.h
  ../libaten/light/ibl.h
  ../libaten/light/light.h
  ../libaten/light/light_impl.h
  ../libaten/light/pointlight.h
  ../libaten/light/spotlight.h
  ../libaten/material/FlakesNormal.cpp
  ../libaten/material/FlakesNormal.h
  ../libaten/material/beckman.cpp
  ../libaten/material/beckman.h
  ../libaten/material/car_paint.cpp
  ../libaten/material/car_paint.h
  ../libaten/material/disney_brdf.cpp
  ../libaten/material/disney_brdf.h
  ../libaten/material/emissive.h
  ../libaten/material/ggx.cpp
  ../libaten/material/ggx.h
  ../libaten/material/lambert.h
  ../libaten/material/material.cpp
  ../libaten/material/material.h
  ../libaten/material/material_impl.h
  ../libaten/material/microfacet_refraction.cpp
  ../libaten/material/microfacet_refraction.h
  ../libaten/material/oren_nayar.cpp
  ../libaten/material/oren_nayar.h
  ../libaten/material/refraction.cpp
  ../libaten/material/refraction.h
  ../libaten/material/retroreflective.cpp
  ../libaten/material/retroreflective.h
  ../libaten/material/sample_texture.h
  ../libaten/material/specular.cpp
  ../libaten/material/specular.h
  ../libaten/material/velvet.cpp
  ../libaten/material/velvet.h
  ../libaten/math/aabb.h
  ../libaten/math/intersect.h
  ../libaten/math/mat4.cpp
  ../libaten/math/mat4.h
  ../libaten/math/math.h
  ../libaten/math/ray.h
  ../libaten/math/vec3.h
  ../libaten/math/vec4.h
  ../libaten/misc/color.cpp
  ../libaten/misc/color.h
  ../libaten/misc/timer.h
  ../libaten/misc/tuple.h
  ../libaten/os/linux/misc/timer_linux.cpp
  ../libaten/renderer/ao/aorenderer_impl.h
  ../libaten/renderer/aov.h
  ../libaten/renderer/npr/feature_line.h
  ../libaten/renderer/npr/npr_impl.h
  ../libaten/renderer/pathtracing/pathtracing_impl.h
  ../libaten/renderer/pathtracing/pt_params.h
  ../libaten/renderer/restir/restir_impl.h
  ../libaten/renderer/restir/restir_types.h
  ../libaten/renderer/svgf/svgf_impl.h
  ../libaten/renderer/svgf/svgf_types.h
  ../libaten/sampler/cmj.h
  ../libaten/sampler/sampler.h
  ../libaten/sampler/wanghash.h
  ../libaten/scene/hitable.h
  ../libaten/scene/scene.h
  ../libaten/types.h
  ../libaten/volume/grid.cpp
  ../libaten/volume/grid_util.h
  ../libaten/volume/medium.cpp
  ../libaten/volume/medium.h
  ../libaten/volume/phase_function.h
  ao/ao.cpp
  ao/ao.cu
  ao/ao.h
  aten4idaten.h
  cuda/cudaGLresource.cpp
  cuda/cudaGLresource.h
  cuda/cudaTextureResource.cpp
  cuda/cudaTextureResource.h
  cuda/cudadefs.h
  cuda/cudainit.h
  cuda/cudamemory.cpp
  cuda/cudamemory.h
  cuda/cudautil.h
  cuda/helper_math.h
  idaten.h
  idaten_namespace.h
  kernel/GenMipMaps.cu
  kernel/GenMipMaps.h
  kernel/LBVHBuilder.cu
  kernel/LBVHBuilder.h
  kernel/MortonCode.cuh
  kernel/RadixSort.cu
  kernel/RadixSort.h
  kernel/Skinning.cu
  kernel/Skinning.h
  kernel/StreamCompaction.cu
  kernel/StreamCompaction.h
  kernel/accelerator.cuh
  kernel/bluenoiseSampler.cuh
  kernel/bvh.cu
  kernel/bvh.cuh
  kernel/bvh_impl.cuh
  kernel/device_scene_context.cpp
  kernel/device_scene_context.cuh
  kernel/intersect.cu
  kernel/intersect.cuh
  kernel/intersect_impl.cuh
  kernel/intersecttype.h
  kernel/pathtracing.cpp
  kernel/pathtracing.h
  kernel/pathtracing_impl.cu
  kernel/persistent_thread.h
  kernel/pt_common.h
  kernel/pt_standard_impl.cu
  kernel/pt_standard_impl.h
  kernel/renderer.cpp
  kernel/renderer.h
  kernel/sample_texture_impl.cu
  kernel/sample_texture_impl.cuh
  kernel/sbvh.cu
  kernel/sbvh.cuh
  kernel/sbvh_impl.cuh
  kernel/stackless_bvh.cu
  kernel/stackless_bvh.cuh
  kernel/stackless_bvh_impl.cuh
  kernel/tex_viewer.cu
  npr/npr_pathtracing.cu
  npr/npr_pathtracing.h
  restir/restir.cpp
  restir/restir.cu
  restir/restir.h
  restir/restir_debug.cu
  restir/restir_reuse.cu
  svgf/svgf.cpp
  svgf/svgf.h
  svgf/svgf_VarianceEstimation.cu
  svgf/svgf_atrous.cu
  svgf/svgf_debug.cu
  svgf/svgf_pt.cu
  svgf/svgf_tp.cu
  volume/grid_loader_device.cu
  volume/grid_loader_device.h
  volume/simple_grid_rendering.cu
  volume/volume_grid.h
  volume/volume_rendering.cpp
  volume/volume_rendering.cu
  volume/volume_rendering.h)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PUBLIC ${nanovdb_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} aten glm)

# Defines outputs , depending Debug or Release.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  install(
    TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}" # Static libraries
    LIBRARY DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}" # Non-DLL shared
                                                              # libraries
    RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}") # Exectutable or
                                                               # DLL
else()
  install(
    TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_RELEASE}" # Static
                                                                # libraries
    LIBRARY DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_RELEASE}" # Non-DLL shared
                                                                # libraries
    RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_RELEASE}") # Exectutable
                                                                 # or DLL
endif()
