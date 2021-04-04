set(PROJECT_NAME atenscene)

project(${PROJECT_NAME})

# Add library to build.
add_library(
  ${PROJECT_NAME} STATIC
  ../../3rdparty/tinyxml2/tinyxml2.cpp
  AssetManager.cpp
  AssetManager.h
  AssimpImporter.cpp
  AssimpImporter.h
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

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE ${stb_INCLUDE_DIRECTORIES} ${imgui_INCLUDE_DIRECTORIES}
          ${tinyobjloader_INCLUDE_DIRECTORIES} ${tinyxml2_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} PUBLIC aten glm tinyobjloader assimp)

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
