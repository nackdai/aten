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

add_subdirectory(glew/build/cmake)
# https://www.glfw.org/docs/latest/compile_guide.html#compile_deps_x11
# Need to install xorg-dev
add_subdirectory(glfw)
add_subdirectory(glm)
add_subdirectory(tinyobjloader)
add_subdirectory(googletest)

# options for assimp
option(ASSIMP_BUILD_TESTS "" OFF)
option(ASSIMP_INSTALL "" OFF)
option(ASSIMP_BUILD_ASSIMP_TOOLS "" OFF)
option(ASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT "" OFF)
option(ASSIMP_NO_EXPORT "" ON)
option(ASSIMP_BUILD_ALL_INPORTERS_BY_DEFAULT "" OFF)
option(ASSIMP_BUILD_OBJ_IMPORTER "" ON)
option(ASSIMP_BUILD_FBX_IMPORTER "" ON)
option(ASSIMP_BUILD_GLTF_IMPORTER "" ON)
# Disable Importer
option(ASSIMP_BUILD_AMF_IMPORTER "" OFF)
option(ASSIMP_BUILD_3DS_IMPORTER "" OFF)
option(ASSIMP_BUILD_AC_IMPORTER "" OFF)
option(ASSIMP_BUILD_ASE_IMPORTER "" OFF)
option(ASSIMP_BUILD_ASSBIN_IMPORTER "" OFF)
option(ASSIMP_BUILD_B3D_IMPORTER "" OFF)
option(ASSIMP_BUILD_BVH_IMPORTER "" OFF)
option(ASSIMP_BUILD_COLLADA_IMPORTER "" OFF)
option(ASSIMP_BUILD_DXF_IMPORTER "" OFF)
option(ASSIMP_BUILD_CSM_IMPORTER "" OFF)
option(ASSIMP_BUILD_HMP_IMPORTER "" OFF)
option(ASSIMP_BUILD_IRRMESH_IMPORTER "" OFF)
option(ASSIMP_BUILD_IRR_IMPORTER "" OFF)
option(ASSIMP_BUILD_LWO_IMPORTER "" OFF)
option(ASSIMP_BUILD_LWS_IMPORTER "" OFF)
option(ASSIMP_BUILD_M3D_IMPORTER "" OFF)
option(ASSIMP_BUILD_MD2_IMPORTER "" OFF)
option(ASSIMP_BUILD_MD3_IMPORTER "" OFF)
option(ASSIMP_BUILD_MD5_IMPORTER "" OFF)
option(ASSIMP_BUILD_MDC_IMPORTER "" OFF)
option(ASSIMP_BUILD_MDL_IMPORTER "" OFF)
option(ASSIMP_BUILD_NFF_IMPORTER "" OFF)
option(ASSIMP_BUILD_NDO_IMPORTER "" OFF)
option(ASSIMP_BUILD_OFF_IMPORTER "" OFF)
option(ASSIMP_BUILD_OGRE_IMPORTER "" OFF)
option(ASSIMP_BUILD_OPENGEX_IMPORTER "" OFF)
option(ASSIMP_BUILD_PLY_IMPORTER "" OFF)
option(ASSIMP_BUILD_MS3D_IMPORTER "" OFF)
option(ASSIMP_BUILD_COB_IMPORTER "" OFF)
option(ASSIMP_BUILD_BLEND_IMPORTER "" OFF)
option(ASSIMP_BUILD_IFC_IMPORTER "" OFF)
option(ASSIMP_BUILD_XGL_IMPORTER "" OFF)
option(ASSIMP_BUILD_Q3D_IMPORTER "" OFF)
option(ASSIMP_BUILD_Q3BSP_IMPORTER "" OFF)
option(ASSIMP_BUILD_RAW_IMPORTER "" OFF)
option(ASSIMP_BUILD_SIB_IMPORTER "" OFF)
option(ASSIMP_BUILD_SMD_IMPORTER "" OFF)
option(ASSIMP_BUILD_STL_IMPORTER "" OFF)
option(ASSIMP_BUILD_TERRAGEN_IMPORTER "" OFF)
option(ASSIMP_BUILD_3D_IMPORTER "" OFF)
option(ASSIMP_BUILD_X_IMPORTER "" OFF)
option(ASSIMP_BUILD_X3D_IMPORTER "" OFF)
option(ASSIMP_BUILD_3MF_IMPORTER "" OFF)
option(ASSIMP_BUILD_MMD_IMPORTER "" OFF)

# NOTE:
# clang newer than 13 deals with warnings "ordered-compare-function-pointers" "unused-but-set-variable" as error.
# In order to disable it, specify -Wno-error=ordered-compare-function-pointers as compiler option.
add_subdirectory(assimp)
