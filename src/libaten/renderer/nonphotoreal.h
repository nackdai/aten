#pragma once

#include "scene/scene.h"
#include "camera/camera.h"
#include "material/material.h"
#include "scene/hitable.h"
#include "sampler/sampler.h"

namespace aten
{
    vec3 shadeNPR(
        material* mtrl,
        const vec3& p,
        const vec3& normal,
        real u, real v,
        scene* scene,
        sampler* sampler);
}
