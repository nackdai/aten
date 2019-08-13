# Output Variables
set(OUTPUT_DEBUG Debug)
set(OUTPUT_RELEASE Release)

# Defines Flags for Windows and Linux. 
if(MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W3 /MD /MDd /Od /EHsc /openmp")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /W3 /GL /O2 /Oi /Gy /EHsc /openmp")
endif(MSVC)
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fopenmp")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    #set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  endif()
endif(NOT MSVC)

# Definition of Macros
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-D__AT_DEBUG__ )
endif()

message("CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shader DESTINATION ${CMAKE_BINARY_DIR})

add_subdirectory(libaten)
add_subdirectory(libatenscene)
add_subdirectory(appaten)
add_subdirectory(libidaten)
add_subdirectory(svgftest)
