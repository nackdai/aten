set(PROJECT_NAME ImageViewer)

project(${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${cmdline_INCLUDE_DIRECTORIES}
                                                  ${imgui_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PUBLIC aten atenscene glm)
