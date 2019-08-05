# Header only libralies
set(tinyobjloader_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader_v09 CACHE PATH "tinyobjloader path")
set(stb_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/stb CACHE PATH "stb path")
set(cmdline_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/cmdline CACHE PATH "cmdline path")
set(imgui_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/imgui CACHE PATH "imgui path")
set(tinyxml2_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/tinyxml2 CACHE PATH "tinyxml2 path")

add_subdirectory(glew/build/cmake)
add_subdirectory(glfw)
add_subdirectory(glm)
add_subdirectory(tinyobjloader_v09)
