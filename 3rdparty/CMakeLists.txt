cmake_minimum_required(VERSION 3.21.0)

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
set(nanovdb_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/nanovdb CACHE PATH "nanovdb path")

set(BUILD_UTILS OFF CACHE BOOL "glew build utils" FORCE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF CACHE BOOL "Build install rpath" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM "3.5" CACHE STRING "workaround for old CMakeLists" FORCE)
add_subdirectory(glew/build/cmake)
unset(CMAKE_POLICY_VERSION_MINIMUM)

add_subdirectory(glm)

set(CMAKE_POLICY_VERSION_MINIMUM "3.5" CACHE STRING "workaround for old CMakeLists" FORCE)
add_subdirectory(tinyobjloader)
unset(CMAKE_POLICY_VERSION_MINIMUM)

add_subdirectory(googletest)

set(ASSIMP_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(ASSIMP_USE_SYSTEM_ZLIB OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL_PDB OFF CACHE BOOL "" FORCE)
set(ASSIMP_NO_EXPORT OFF CACHE BOOL "" FORCE)

# Disable all importers/exporters in assimp.
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)

# OBJ, FBX, GLTF importers are allowed.
set(ASSIMP_BUILD_OBJ_IMPORTER ON)
set(ASSIMP_BUILD_FBX_IMPORTER ON)
set(ASSIMP_BUILD_GLTF_IMPORTER ON)

add_subdirectory(assimp)

if(MSVC)
  target_compile_options(assimp PRIVATE /wd4834 /wd4819 /utf-8 /EHsc)
else()
  # Ignore the case that all warnings are treated as the error in assimp.
  target_compile_options(assimp PRIVATE -Wno-error)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# NOTE:
# Adding glfw before assimp disables to build assimp's shared library.
# So, it has to done after assimp.

# https://www.glfw.org/docs/latest/compile_guide.html#compile_deps_x11
# Need to install xorg-dev
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Disable docs" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
add_subdirectory(glfw)

file(GLOB IMGUI_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_glfw.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_draw.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_tables.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_widgets.cpp"
)
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/imgui>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends>
)
target_link_libraries(imgui PRIVATE glfw)

set(BUILD_SHARED_LIBS OFF)
set(MY_EXTRA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    "${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES};${MY_EXTRA_INCLUDE}"
)
add_subdirectory(imgui_gradient)

if(MSVC)
  foreach(cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${cfg} CFG)
    set_target_properties(glew PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/glew/lib/${cfg}
        LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/glew/lib/${cfg}
        RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/glew/bin/${cfg}
    )
    set_target_properties(gtest PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/googletest/lib/${cfg}
        LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/googletest/lib/${cfg}
        RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/googletest/bin/${cfg}
    )
    set_target_properties(gtest_main PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/googletest/lib/${cfg}
        LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/googletest/lib/${cfg}
        RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/googletest/bin/${cfg}
    )
    set_target_properties(gmock PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/googletest/lib/${cfg}
        LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/googletest/lib/${cfg}
        RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/googletest/bin/${cfg}
    )
    set_target_properties(imgui PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/imgui/lib/${cfg}
        LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/imgui/lib/${cfg}
        RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/imgui/bin/${cfg}
    )
  endforeach()
endif()
