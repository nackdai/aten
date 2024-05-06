#include "camera/equirect.h"
#include "scene/scene.h"

namespace aten {
    CameraSampleResult EquirectCamera::sample(
        float s, float t,
        sampler* sampler) const
    {
        AT_ASSERT(0 <= s && s <= 1);
        AT_ASSERT(0 <= t && t <= 1);

        s = aten::clamp<float>(s, 0, 1);
        t = aten::clamp<float>(t, 0, 1);

        auto dir = AT_NAME::Background::ConvertUVToDirection(s, t);

        CameraSampleResult result;
        result.posOnLens = m_origin + dir;
        result.nmlOnLens = dir;
        result.posOnImageSensor = m_origin;
        result.r = ray(m_origin, dir);

        return result;
    }
}
