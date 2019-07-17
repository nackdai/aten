set(PROJECT_NAME atenscene)

project(${PROJECT_NAME})

# Add library to build.
add_library(${PROJECT_NAME} STATIC
  AssetManager.cpp
  AssetManager.h
  ImageLoader.cpp
  ImageLoader.h
  MaterialExporter.cpp
  MaterialExporter.h
  MaterialLoader.cpp
  MaterialLoader.h
  ObjLoader.cpp
  ObjLoader.h
  ObjWriter.cpp
  ObjWriter.h
  SceneLoader.cpp
  SceneLoader.h
  atenscene.h
  utility.h)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ../libaten
    ${stb_INCLUDE_DIRECTORIES}
    ${imgui_INCLUDE_DIRECTORIES})

if(MSVC)
  target_link_libraries(${PROJECT_NAME} PRIVATE glfw3.lib glew32d.lib OpenGL32.lib Shlwapi.lib )
endif(MSVC)
if(NOT MSVC)
  target_link_libraries(${PROJECT_NAME} PRIVATE glfw glew glm) 
endif(NOT MSVC)

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
