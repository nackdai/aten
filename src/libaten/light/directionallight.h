#pragma once

#include "light/light.h"

namespace aten {
    class Values;
}

namespace AT_NAME {
    class DirectionalLight : public Light {
    public:
        DirectionalLight()
            : Light(aten::LightType::Direction, aten::LightAttributeDirectional)
        {}
        DirectionalLight(
            const aten::vec3& dir,
            const aten::vec3& le)
            : Light(aten::LightType::Direction, aten::LightAttributeDirectional)
        {
            m_param.dir = normalize(dir);
            m_param.le = le;
        }

        DirectionalLight(aten::Values& val);

        virtual ~DirectionalLight() = default;

    public:
        static AT_DEVICE_API void sample(
            const aten::LightParameter* param,
            const aten::vec3& org,
            aten::sampler* sampler,
            aten::LightSampleResult* result)
        {
            // PDF to sample area.
            result->pdf = real(1);

            result->dir = -normalize(param->dir);
            result->nml = normalize(param->dir);

            // TODO
            // シーンのAABBを覆う球上に配置されるようにするべき.
            result->pos = org + real(100000) * real(0.5) * result->dir;

            result->le = param->le;
            result->finalColor = param->le;
        }
    };
}
