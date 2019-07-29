set(PROJECT_NAME idaten)

project(${PROJECT_NAME})

find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

message("-- CUDA --")
message("Version: " ${CUDA_VERSION})
message("Library: " ${CUDA_CUDA_LIBRARY})
message("Runtime: " ${CUDA_CUDART_LIBRARY})
message("Host Compiler: " ${CUDA_HOST_COMPILER})

# https://gist.github.com/erikzenker/713c4ff76949058d5d5d

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    "-ccbin " ${CMAKE_CXX_COMPILER}
    #" -rdc=true --compile -cudart static --machine 64 -G -g -gencode arch=compute_60,code=sm_60 -x cu -Xcompiler ,-O0 -g -std=c++14 -D__CORRECT_ISO_CPP11_MATH_H_PROTO")
    " -rdc=true -gencode arch=compute_60,code=sm_60 -Xcompiler ,-O0 -g -std=c++14 -D__CORRECT_ISO_CPP11_MATH_H_PROTO")

message(${CUDA_NVCC_FLAGS})

add_definitions(-D__AT_CUDA__)

# Add library to build.
cuda_add_library(${PROJECT_NAME} STATIC
  idaten.h
  idaten_namespace.h
  aten4idaten.h
  asvgf/asvgf.cpp
  asvgf/asvgf.h
  asvgf/asvgf_AtrousGradient.cu
  asvgf/asvgf_CreateGradient.cu
  asvgf/asvgf_ForwardProjection.cu
  asvgf/asvgf_debug.cu
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
  multigpu/pt_multigpu.cpp
  multigpu/pt_multigpu.h
  multigpu/renderer_multigpu.h
  multigpu/svgf_multigpu.cpp
  multigpu/svgf_multigpu.h
  svgf/svgf.cpp
  svgf/svgf.h
  svgf/svgf_VarianceEstimation.cu
  svgf/svgf_atrous.cu
  svgf/svgf_debug.cu
  svgf/svgf_init.cu
  svgf/svgf_pt.cu
  svgf/svgf_ssrt.cu
  svgf/svgf_taa.cu
  svgf/svgf_tile.cu
  svgf/svgf_tp.cu)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(${PROJECT_NAME} 
    aten
    glm)

set_target_properties(${PROJECT_NAME}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Defines outputs , depending Debug or Release.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}"   # Static libraries
    LIBRARY DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}"   # Non-DLL shared libraries
    RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_DEBUG}")  # Exectutable or DLL
else()
  install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_RELEASE}"   # Static libraries
    LIBRARY DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_RELEASE}"   # Non-DLL shared libraries
    RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/${OUTPUT_RELEASE}")  # Exectutable or DLL
endif()
