#pragma once

#include "types.h"
#include "math/vec3.h"
#include "texture/texture.h"

#ifdef __AT_CUDA__

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"

namespace AT_NAME {
    AT_DEVICE_MTRL_API aten::vec3 sampleTexture(const int texid, real u, real v, const aten::vec3& defaultValue, int lod = 0);

#ifndef __AT_DEBUG__
#include "kernel/sample_texture_impl.cuh"
#endif
}
#else

#include "scene/context.h"

namespace AT_NAME {
    inline AT_DEVICE_MTRL_API aten::vec3 sampleTexture(const int texid, real u, real v, const aten::vec3& defaultValue, int lod = 0)
    {
        aten::vec3 ret = defaultValue;

        // TODO
        if (texid >= 0) {
            const auto ctxt = aten::context::getPinnedContext();
            auto tex = ctxt->getTexture(texid);
            if (tex) {
                ret = tex->at(u, v);
            }
        }

        return std::move(ret);
    }
}
#endif

namespace AT_NAME {
    inline AT_DEVICE_MTRL_API void applyNormalMap(
        const int normalMapIdx,
        const aten::vec3& orgNml,
        aten::vec3& newNml,
        real u, real v)
    {
        if (normalMapIdx >= 0) {
            auto nml = sampleTexture(normalMapIdx, u, v, aten::vec3(real(0)));
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