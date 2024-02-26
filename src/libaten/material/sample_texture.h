#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "texture/texture.h"

#ifdef __AT_CUDA__

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"

namespace AT_NAME {
    AT_DEVICE_API aten::vec4 sampleTexture(const int32_t texid, real u, real v, const aten::vec4& defaultValue, int32_t lod = 0);

#ifndef __AT_DEBUG__
#include "kernel/sample_texture_impl.cuh"
#endif
}
#else

#include "scene/host_scene_context.h"

namespace AT_NAME {
    inline AT_DEVICE_API aten::vec4 sampleTexture(const int32_t texid, real u, real v, const aten::vec4& defaultValue, int32_t lod = 0)
    {
        aten::vec4 ret = defaultValue;

        // TODO
        if (texid >= 0) {
            const auto ctxt = aten::context::getPinnedContext();
            auto tex = ctxt->GtTexture(texid);
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
        const int32_t normalMapIdx,
        const aten::vec3& orgNml,
        aten::vec3& newNml,
        real u, real v)
    {
        if (normalMapIdx >= 0) {
            auto nml = aten::vec3(sampleTexture(normalMapIdx, u, v, aten::vec4(real(0))));
            nml = real(2) * nml - aten::vec3(1);    // [0, 1] -> [-1, 1].
            nml = normalize(nml);

            aten::vec3 n = normalize(orgNml);
            aten::vec3 t = aten::getOrthoVector(n);
            aten::vec3 b = cross(n, t);

            newNml = nml.z * n + nml.x * t + nml.y * b;
            newNml = normalize(newNml);
        }
        else {
            newNml = normalize(orgNml);
        }
    }
}
