#pragma once

#include "defs.h"
#include "types.h"

#include "accelerator/accelerator.h"

#include "camera/camera.h"
#include "camera/pinhole.h"

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

#include "scene/scene.h"

#include "shape/sphere.h"
#include "shape/cube.h"

#include "object/vertex.h"
#include "object/object.h"

#include "sampler/sampler.h"
#include "sampler/wanghash.h"