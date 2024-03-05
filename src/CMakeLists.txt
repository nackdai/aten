# Output Variables
set(OUTPUT_DEBUG Debug)
set(OUTPUT_RELEASE Release)

# TODO https://qiita.com/shohirose/items/637f4b712893764a7ec1

# Defines Flags for Windows and Linux.
if(MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} /W3 /MD /MDd /Od /EHsc /openmp")
  set(CMAKE_CXX_FLAGS_RELEASE
      "${CMAKE_CXX_FLAGS_RELEASE} /W3 /GL /O2 /Oi /Gy /EHsc /openmp")
endif(MSVC)
if(NOT MSVC)
  # NOTE: It seems that libc++ for clang-9 doesn't support part of c++17 spec.
  # https://github.com/conda-forge/libcxx-feedstock/issues/44
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fopenmp -std=c++17")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -std=c++17")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -stdlib=libc++") set
    # (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -stdlib=libc++")
  endif()
endif(NOT MSVC)

# Definition of Macros
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-D__AT_DEBUG__)
endif()

message("CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message("CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_CXX_FLAGS_DEBUG})
message("CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader DESTINATION ${CMAKE_BINARY_DIR})

add_subdirectory(libaten)
add_subdirectory(libatenscene)
add_subdirectory(appaten)
add_subdirectory(libidaten)
add_subdirectory(device_renderer)
add_subdirectory(svgf_renderer)
add_subdirectory(ao_renderer)
add_subdirectory(restir_renderer)
add_subdirectory(aten_unittest)
