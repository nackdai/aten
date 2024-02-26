#pragma once

#include "light/light.h"
#include "light/spotlight.h"
#include "light/pointlight.h"
#include "light/directionallight.h"
#include "light/arealight.h"
#include "light/ibl.h"

namespace AT_NAME
{
    template <class CONTEXT>
    inline AT_DEVICE_API void Light::sample(
        aten::LightSampleResult& result,
        const aten::LightParameter& param,
        const CONTEXT& ctxt,
        const aten::vec3& org,
        const aten::vec3& nml,
        aten::sampler* sampler,
        uint32_t lod)
    {
        switch (param.type) {
        case aten::LightType::Area:
            AT_NAME::AreaLight::sample(result, param, ctxt, org, sampler);
            break;
        case aten::LightType::IBL:
            AT_NAME::ImageBasedLight::sample(result, param, ctxt, org, nml, sampler, lod);
            break;
        case aten::LightType::Point:
            AT_NAME::PointLight::sample(&param, org, sampler, &result);
            break;
        case aten::LightType::Spot:
            AT_NAME::SpotLight::sample(&param, org, sampler, &result);
            break;
        case aten::LightType::Direction:
            AT_NAME::DirectionalLight::sample(&param, org, sampler, &result);
            break;
        default:
            AT_ASSERT(false);
            break;
        }
    }
}
