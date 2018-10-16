#pragma once

#include "scene/scene.h"
#include "scene/context.h"
#include "camera/camera.h"
#include "material/material.h"
#include "scene/hitable.h"
#include "sampler/sampler.h"

namespace aten
{
    vec3 shadeNPR(
        const context& ctxt,
        const material* mtrl,
        const vec3& p,
        const vec3& normal,
        real u, real v,
        scene* scene,
        sampler* sampler);
}
