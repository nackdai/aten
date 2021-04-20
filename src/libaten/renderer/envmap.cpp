#include "renderer/envmap.h"

namespace AT_NAME
{
    aten::vec3 envmap::sample(const aten::ray& inRay) const
    {
        AT_ASSERT(m_envmap);

        // Translate cartesian coordinates to spherical system.
        const aten::vec3& dir = inRay.dir;

        auto uv = convertDirectionToUV(dir);
        auto u = uv.x;
        auto v = uv.y;

        auto ret = m_envmap->at(u, v);

        return ret;
    }

    aten::vec3 envmap::sample(real u, real v) const
    {
        auto ret = m_envmap->at(u, v);

        return ret;
    }
}
