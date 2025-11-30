#include "light/arealight.h"
#include "light/directionallight.h"
#include "light/ibl.h"
#include "light/light.h"
#include "light/pointlight.h"
#include "light/spotlight.h"
#include "misc/value.h"

namespace AT_NAME {
    Light::Light(aten::LightType type, const aten::LightAttribute& attrib, const aten::Values& val)
        : m_param(type, attrib)
    {
    }

    AreaLight::AreaLight(const aten::Values& val)
        : Light(aten::LightType::Area, LightAttributeArea, val)
    {
    }

    DirectionalLight::DirectionalLight(aten::Values& val)
        : Light(aten::LightType::Direction, LightAttributeDirectional, val)
    {}

    PointLight::PointLight(aten::Values& val)
        : Light(aten::LightType::Point, LightAttributeSingluar, val)
    {
    }

    SpotLight::SpotLight(aten::Values& val)
        : Light(aten::LightType::Spot, LightAttributeSingluar, val)
    {
    }
}
