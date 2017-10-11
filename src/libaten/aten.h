#pragma once

#include "defs.h"
#include "types.h"

#include "camera/pinhole.h"
#include "camera/thinlens.h"
#include "camera/equirect.h"
#include "camera/CameraOperator.h"

#include "filter/nlm.h"
#include "filter/bilateral.h"
#include "filter/atrous.h"
#include "filter/taa.h"

#include "filter/PracticalNoiseReduction/PracticalNoiseReduction.h"
#include "filter/VirtualFlashImage/VirtualFlashImage.h"
#include "filter/GeometryRendering/GeometryRendering.h"

#include "material/emissive.h"
#include "material/lambert.h"
#include "material/specular.h"
#include "material/refraction.h"
#include "material/blinn.h"
#include "material/ggx.h"
#include "material/beckman.h"
#include "material/disney_brdf.h"
#include "material/oren_nayar.h"
#include "material/layer.h"
#include "material/toon.h"

#include "math/math.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "math/ray.h"
#include "math/mat4.h"
#include "math/aabb.h"

#include "misc/color.h"
#include "misc/timer.h"
#include "misc/thread.h"
#include "misc/value.h"
#include "misc/stream.h"

#include "light/light.h"
#include "light/pointlight.h"
#include "light/directionallight.h"
#include "light/spotlight.h"
#include "light/arealight.h"
#include "light/ibl.h"

#include "object/object.h"
#include "object/vertex.h"

#include "proxy/DataCollector.h"

#include "texture/texture.h"

#include "hdr/hdr.h"
#include "hdr/tonemap.h"
#include "hdr/gamma.h"

#include "visualizer/visualizer.h"
#include "visualizer/window.h"
#include "visualizer/shader.h"
#include "visualizer/blitter.h"

#include "scene/scene.h"
#include "scene/instance.h"

#include "accelerator/accelerator.h"
#include "accelerator/bvh.h"
#include "accelerator/gpu_bvh.h"

#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/wanghash.h"

#include "shape/sphere.h"
#include "shape/cube.h"
#include "shape/tranformable.h"

#include "renderer/renderer.h"
#include "renderer/film.h"
#include "renderer/background.h"
#include "renderer/envmap.h"
#include "renderer/raytracing.h"
#include "renderer/pathtracing.h"
#include "renderer/erpt.h"
#include "renderer/pssmlt.h"
#include "renderer/aov.h"
#include "renderer/sorted_pathtracing.h"
#include "renderer/bdpt.h"
#include "renderer/directlight.h"

#include "posteffect/BloomEffect.h"
