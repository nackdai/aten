#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "image/texture.h"

#ifdef __AT_CUDA__

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"

#include "kernel/device_scene_context.cuh"

namespace AT_NAME {
    AT_DEVICE_API aten::vec4 sampleTexture(
        const AT_NAME::context& ctxt,
        const int32_t texid,
        float u, float v,
        const aten::vec4& defaultValue,
        int32_t lod = 0);

#ifndef __AT_DEBUG__
#include "kernel/sample_texture_impl.cuh"
#endif
}
#else

#include "scene/host_scene_context.h"

namespace AT_NAME {
    inline AT_DEVICE_API aten::vec4 sampleTexture(
        const AT_NAME::context& ctxt,
        const int32_t texid,
        float u, float v,
        const aten::vec4& defaultValue,
        int32_t lod = 0)
    {
        aten::vec4 ret = defaultValue;

        // TODO
        if (texid >= 0) {
            auto tex = ctxt.GetTexture(texid);
            if (tex) {
                ret = tex->at(u, v);
            }
        }

        return ret;
    }
}
#endif

namespace AT_NAME {
    inline AT_DEVICE_API void applyNormalMap(
        const AT_NAME::context& ctxt,
        const int32_t normalMapIdx,
        const aten::vec3& orgNml,
        aten::vec3& newNml,
        float u, float v)
    {
        if (normalMapIdx >= 0) {
            auto nml = aten::vec3(sampleTexture(ctxt, normalMapIdx, u, v, aten::vec4(float(0))));
            nml = float(2) * nml - aten::vec3(1);    // [0, 1] -> [-1, 1].
            nml = normalize(nml);

            aten::vec3 n = normalize(orgNml);

            aten::vec3 t, b;
            aten::tie(t, b) = aten::GetTangentCoordinate(n);

            newNml = nml.z * n + nml.x * t + nml.y * b;
            newNml = normalize(newNml);
        }
        else {
            newNml = normalize(orgNml);
        }
    }
}
