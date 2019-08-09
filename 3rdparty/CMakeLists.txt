# Header only libralies
set(tinyobjloader_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader CACHE PATH "tinyobjloader path")
set(stb_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/stb CACHE PATH "stb path")
set(cmdline_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/cmdline CACHE PATH "cmdline path")
set(imgui_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/imgui CACHE PATH "imgui path")
set(tinyxml2_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/tinyxml2 CACHE PATH "tinyxml2 path")
set(glew_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/glew/include CACHE PATH "glew path")

add_subdirectory(glew/build/cmake)
# https://www.glfw.org/docs/latest/compile_guide.html#compile_deps_x11
# Need to install xorg-dev
add_subdirectory(glfw)
add_subdirectory(glm)
add_subdirectory(tinyobjloader)
