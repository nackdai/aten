# Header only libralies
set(tinyobjloader_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader_v09 CACHE PATH "tinyobjloader path")
set(stb_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/stb CACHE PATH "stb path")
set(cmdline_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/cmdline CACHE PATH "cmdline path")
set(imgui_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/imgui CACHE PATH "imgui path")


add_subdirectory(glew/build/cmake)
add_subdirectory(glfw)
add_subdirectory(glm)
