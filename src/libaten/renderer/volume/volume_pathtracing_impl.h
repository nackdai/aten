#pragma once

#include "defs.h"
#include "material/material.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "renderer/pathtracing/pt_params.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif


namespace AT_NAME
{
    inline AT_DEVICE_API void UpdateMedium(
        const ray& ray,
        const vec3& wo,
        const vec3& surface_normal,
        const MaterialParameter& mtrl,
        PathThroughput& throughput)
    {
        const auto wi = -ray.dir;
        const auto is_trasmitted = dot(wo, wi) < 0;
        const auto is_enter = dot(wi, surface_normal) > 0;

        if (is_trasmitted) {
            if (is_enter) {
                if (mtrl.is_medium) {
                    throughput.mediums.push(mtrl.id);
                }
            }
            else {
                //std::ignore = throughput.mediums.safe_pop();
                if (throughput.mediums.size() > 0) {
                    throughput.mediums.pop();
                }
            }
        }
    }

    inline AT_DEVICE_API int32_t GetCurrentMediumIdx(const PathThroughput& throughput)
    {
        return throughput.mediums.top();
    }

    inline AT_DEVICE_API bool HasMedium(const PathThroughput& throughput)
    {
        return !throughput.mediums.empty();
    }
}
