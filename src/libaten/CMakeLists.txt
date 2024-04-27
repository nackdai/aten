set(PROJECT_NAME aten)

project(${PROJECT_NAME})

# Add library to build.
add_library(
  ${PROJECT_NAME} STATIC
  ../../3rdparty/imgui/backends/imgui_impl_glfw.cpp
  ../../3rdparty/imgui/backends/imgui_impl_glfw.h
  ../../3rdparty/imgui/backends/imgui_impl_opengl3.cpp
  ../../3rdparty/imgui/backends/imgui_impl_opengl3.h
  ../../3rdparty/imgui/imconfig.h
  ../../3rdparty/imgui/imgui.cpp
  ../../3rdparty/imgui/imgui.h
  ../../3rdparty/imgui/imgui_draw.cpp
  ../../3rdparty/imgui/imgui_internal.h
  ../../3rdparty/imgui/imgui_tables.cpp
  ../../3rdparty/imgui/imgui_widgets.cpp
  accelerator/GpuPayloadDefs.h
  accelerator/accelerator.cpp
  accelerator/accelerator.h
  accelerator/bvh.cpp
  accelerator/bvh.h
  accelerator/bvh_node.cpp
  accelerator/bvh_node.h
  accelerator/bvh_update.cpp
  accelerator/bvh_util.h
  accelerator/sbvh.cpp
  accelerator/sbvh.h
  accelerator/sbvh_voxel.cpp
  accelerator/stackless_bvh.cpp
  accelerator/stackless_bvh.h
  accelerator/threaded_bvh.cpp
  accelerator/threaded_bvh.h
  aten.h
  aten_namespace.h
  camera/CameraOperator.cpp
  camera/CameraOperator.h
  camera/camera.h
  camera/equirect.cpp
  camera/equirect.h
  camera/pinhole.cpp
  camera/pinhole.h
  camera/thinlens.cpp
  camera/thinlens.h
  deformable/ANMFormat.h
  deformable/DeformAnimation.cpp
  deformable/DeformAnimation.h
  deformable/DeformAnimationInterp.cpp
  deformable/DeformAnimationInterp.h
  deformable/DeformMesh.cpp
  deformable/DeformMesh.h
  deformable/DeformMeshGroup.cpp
  deformable/DeformMeshGroup.h
  deformable/DeformMeshSet.cpp
  deformable/DeformMeshSet.h
  deformable/DeformPrimitives.cpp
  deformable/DeformPrimitives.h
  deformable/MDLFormat.h
  deformable/MSHFormat.h
  deformable/SKLFormat.h
  deformable/Skeleton.cpp
  deformable/Skeleton.h
  deformable/SkinningVertex.h
  deformable/deformable.cpp
  deformable/deformable.h
  defs.h
  filter/atrous.cpp
  filter/atrous.h
  filter/magnifier.cpp
  filter/magnifier.h
  filter/taa.cpp
  filter/taa.h
  geometry/EvaluateHitResult.h
  geometry/NoHitableMesh.cpp
  geometry/NoHitableMesh.h
  geometry/PolygonObject.cpp
  geometry/PolygonObject.h
  geometry/TriangleGroupMesh.cpp
  geometry/TriangleGroupMesh.h
  geometry/geomparam.h
  geometry/instance.h
  geometry/sphere.cpp
  geometry/sphere.h
  geometry/transformable.h
  geometry/transformable_factory.h
  geometry/triangle.cpp
  geometry/triangle.h
  geometry/vertex.h
  hdr/gamma.cpp
  hdr/gamma.h
  hdr/hdr.cpp
  hdr/hdr.h
  hdr/tonemap.cpp
  hdr/tonemap.h
  light/arealight.cpp
  light/arealight.h
  light/directionallight.h
  light/ibl.cpp
  light/ibl.h
  light/light.h
  light/light_impl.h
  light/light_instantiate_from_value.cpp
  light/light_parameter.h
  light/pointlight.h
  light/spotlight.h
  material/FlakesNormal.cpp
  material/FlakesNormal.h
  material/beckman.cpp
  material/beckman.h
  material/car_paint.cpp
  material/car_paint.h
  material/disney_brdf.cpp
  material/disney_brdf.h
  material/emissive.h
  material/ggx.cpp
  material/ggx.h
  material/lambert.h
  material/material.cpp
  material/material.h
  material/material_impl.h
  material/material_instantiate_from_value.cpp
  material/microfacet_refraction.cpp
  material/microfacet_refraction.h
  material/oren_nayar.cpp
  material/oren_nayar.h
  material/refraction.cpp
  material/refraction.h
  material/retroreflective.cpp
  material/retroreflective.h
  material/sample_texture.h
  material/specular.cpp
  material/specular.h
  material/velvet.cpp
  material/velvet.h
  math/aabb.h
  math/frustum.h
  math/intersect.h
  math/mat4.cpp
  math/mat4.h
  math/math.h
  math/quaternion.h
  math/ray.h
  math/vec2.h
  math/vec3.h
  math/vec4.h
  misc/bitflag.h
  misc/color.cpp
  misc/color.h
  misc/const_span.h
  misc/key.h
  misc/misc.h
  misc/omputil.cpp
  misc/omputil.h
  misc/span.h
  misc/stream.h
  misc/thread.cpp
  misc/thread.h
  misc/timeline.cpp
  misc/timeline.h
  misc/timer.h
  misc/tuple.h
  misc/type_traits.h
  misc/value.h
  os/linux/misc/timer_linux.cpp
  os/linux/system_linux.cpp
  os/system.h
  proxy/DataCollector.cpp
  proxy/DataCollector.h
  renderer/ao/aorenderer.cpp
  renderer/ao/aorenderer.h
  renderer/ao/aorenderer_impl.h
  renderer/aov.h
  renderer/background.h
  renderer/film.cpp
  renderer/film.h
  renderer/npr/feature_line.cpp
  renderer/npr/feature_line.h
  renderer/npr/npr_impl.h
  renderer/pathtracing/pathtracing.cpp
  renderer/pathtracing/pathtracing.h
  renderer/pathtracing/pathtracing_impl.h
  renderer/pathtracing/pt_params.h
  renderer/renderer.h
  renderer/restir/restir.cpp
  renderer/restir/restir.h
  renderer/restir/restir_impl.h
  renderer/restir/restir_types.h
  renderer/svgf/svgf.cpp
  renderer/svgf/svgf.h
  renderer/svgf/svgf_impl.h
  renderer/svgf/svgf_types.h
  sampler/cmj.h
  sampler/sampler.cpp
  sampler/sampler.h
  sampler/samplerinterface.h
  sampler/wanghash.h
  scene/AcceleratedScene.h
  scene/hit_parameter.h
  scene/hitable.cpp
  scene/hitable.h
  scene/host_scene_context.cpp
  scene/host_scene_context.h
  scene/scene.cpp
  scene/scene.h
  texture/texture.cpp
  texture/texture.h
  types.h
  visualizer/GLProfiler.cpp
  visualizer/GLProfiler.h
  visualizer/GeomDataBuffer.cpp
  visualizer/GeomDataBuffer.h
  visualizer/MultiPassPostProc.cpp
  visualizer/MultiPassPostProc.h
  visualizer/RasterizeRenderer.cpp
  visualizer/RasterizeRenderer.h
  visualizer/atengl.h
  visualizer/blitter.cpp
  visualizer/blitter.h
  visualizer/fbo.cpp
  visualizer/fbo.h
  visualizer/pixelformat.h
  visualizer/shader.cpp
  visualizer/shader.h
  visualizer/visualizer.cpp
  visualizer/visualizer.h
  visualizer/window.cpp
  visualizer/window.h)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PRIVATE ${glew_INCLUDE_DIRECTORIES} ${stb_INCLUDE_DIRECTORIES}
          ${imgui_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} PRIVATE glfw glew glm)

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
