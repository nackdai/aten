set(PROJECT_NAME appaten)

project(${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

add_executable(${PROJECT_NAME} main_test.cpp ../common/scenedefs.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC aten atenscene glm)
