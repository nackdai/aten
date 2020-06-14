set(PROJECT_NAME aorenderer)

project(${PROJECT_NAME})

find_package(CUDA REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v -lcudart -lcudadevrt")

message("cudalibs " ${CUDA_cudadevrt_LIBRARY})

add_executable(${PROJECT_NAME} main.cpp ../common/scenedefs.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS}
                                                  ${imgui_INCLUDE_DIRECTORIES})
target_link_directories(${PROJECT_NAME} PUBLIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
target_link_libraries(${PROJECT_NAME} aten atenscene idaten glm)
