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
    -gencode
    arch=compute_${CUDA_TARGET_COMPUTE_CAPABILITY},code=sm_${CUDA_TARGET_COMPUTE_CAPABILITY}
    -std=c++14)

message("nvcc_flags " ${CUDA_NVCC_FLAGS})

add_definitions(-D__AT_CUDA__)

set_source_files_properties(
  ../libaten/material/material.cpp
  ../libaten/material/beckman.cpp
  ../libaten/material/blinn.cpp
  ../libaten/material/ggx.cpp
  ../libaten/material/oren_nayar.cpp
  ../libaten/material/refraction.cpp
  ../libaten/material/specular.cpp
  ../libaten/material/FlakesNormal.cpp
  ../libaten/material/velvet.cpp
  ../libaten/material/lambert_refraction.cpp
  ../libaten/material/microfacet_refraction.cpp
  ../libaten/material/disney_brdf.cpp
  ../libaten/geometry/sphere.cpp
  ../libaten/camera/pinhole.cpp
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# Add library to build.
cuda_add_library(
  ${PROJECT_NAME}
  STATIC
  ../libaten/os/linux/misc/timer_linux.cpp
  ../libaten/geometry/sphere.cpp
  ../libaten/misc/color.cpp
  ../libaten/renderer/envmap.cpp
  ../libaten/sampler/sobol.cpp
  ../libaten/camera/pinhole.cpp
  ../libaten/light/arealight.cpp
  ../libaten/light/ibl.cpp
  ../libaten/material/beckman.cpp
  ../libaten/material/blinn.cpp
  ../libaten/material/ggx.cpp
  ../libaten/material/material.cpp
  ../libaten/material/oren_nayar.cpp
  ../libaten/material/refraction.cpp
  ../libaten/material/specular.cpp
  ../libaten/material/FlakesNormal.cpp
  ../libaten/material/velvet.cpp
  ../libaten/material/lambert_refraction.cpp
  ../libaten/material/microfacet_refraction.cpp
  ../libaten/material/layer.cpp
  ../libaten/material/disney_brdf.cpp
  ../libaten/math/mat4.cpp
  idaten.h
  idaten_namespace.h
  aten4idaten.h
  ao/ao.cpp
  ao/ao.cu
  ao/ao.h
  cuda/cudaGLresource.cpp
  cuda/cudaGLresource.h
  cuda/cudaLayeredTextureResource.cpp
  cuda/cudaTextureResource.cpp
  cuda/cudaTextureResource.h
  cuda/cudadefs.h
  cuda/cudainit.h
  cuda/cudamemory.cpp
  cuda/cudamemory.h
  cuda/cudautil.h
  cuda/helper_math.h
  cuda/multigpu.h
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
  kernel/context.cuh
  kernel/idatendefs.cuh
  kernel/intersect.cu
  kernel/intersect.cuh
  kernel/intersect_impl.cuh
  kernel/intersecttype.h
  kernel/light.cu
  kernel/light.cuh
  kernel/light_impl.cuh
  kernel/material.cu
  kernel/material.cuh
  kernel/material_impl.cuh
  kernel/pathtracing.cpp
  kernel/pathtracing.h
  kernel/pathtracing_impl.cu
  kernel/pt_common.h
  kernel/qbvh.cu
  kernel/qbvh.cuh
  kernel/qbvh_impl.cuh
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
  kernel/stackless_qbvh.cu
  kernel/stackless_qbvh.cuh
  kernel/stackless_qbvh_impl.cuh
  kernel/tex_viewer.cu
  multigpu/pt_multigpu.cpp
  multigpu/pt_multigpu.h
  multigpu/renderer_multigpu.h
  multigpu/svgf_multigpu.cpp
  multigpu/svgf_multigpu.h
  restir/reservior.h
  restir/restir.cu
  restir/restir_debug.cu
  restir/restir_init.cu
  restir/restir_reuse.cu
  restir/restir_ssrt.cu
  restir/restir.cpp
  restir/restir.h
  restir/restir_info.h
  restir/restir_pt.cu
  restir/restir_sample_light.cuh
  svgf/svgf.cpp
  svgf/svgf.h
  svgf/svgf_VarianceEstimation.cu
  svgf/svgf_atrous.cu
  svgf/svgf_debug.cu
  svgf/svgf_init.cu
  svgf/svgf_pt.cu
  svgf/svgf_ssrt.cu
  svgf/svgf_tile.cu
  svgf/svgf_tp.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

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
