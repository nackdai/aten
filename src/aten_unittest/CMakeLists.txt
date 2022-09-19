set(PROJECT_NAME aten_unittest)

project(${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

add_executable(${PROJECT_NAME} aov_host_buffer.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC aten glm gtest gtest_main)
