#pragma once

#include "light/light.h"

namespace aten {
    class Values;
}

namespace AT_NAME {
    class DirectionalLight : public Light {
    public:
        DirectionalLight()
            : Light(aten::LightType::Direction, LightAttributeDirectional)
        {}
        DirectionalLight(
            const aten::vec3& dir,
            const aten::vec3& le)
            : Light(aten::LightType::Direction, LightAttributeDirectional)
        {
            m_param.dir = normalize(dir);
            m_param.le = le;
        }

        DirectionalLight(aten::Values& val);

        virtual ~DirectionalLight() {}

    public:
        virtual aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            aten::sampler* sampler) const override final
        {
            aten::LightSampleResult result;
            sample(&m_param, org, sampler, &result);
            return result;
        }

        static AT_DEVICE_API void sample(
            const aten::LightParameter* param,
            const aten::vec3& org,
            aten::sampler* sampler,
            aten::LightSampleResult* result)
        {
            // PDF to sample area.
            result->pdf = real(1);

            result->dir = -normalize(param->dir);
            result->nml = aten::vec3();    // Not used...

            // TODO
            // シーンのAABBを覆う球上に配置されるようにするべき.
            result->pos = org + real(100000) * real(0.5) * result->dir;

            result->le = param->le;
            result->intensity = real(1);
            result->finalColor = param->le;
        }
    };
}
