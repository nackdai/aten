#include "light/arealight.h"
#include "geometry/transformable.h"

namespace AT_NAME {
    AreaLight::AreaLight(
        const std::shared_ptr<aten::transformable>& obj,
        const aten::vec3& light_color,
        const float intensity,
        const float scale/*= 1.0F*/)
        : Light(aten::LightType::Area, aten::LightAttributeArea)
    {
        m_obj = obj;
        m_param.arealight_objid = obj->id();
        m_param.light_color = light_color;
        m_param.intensity = intensity;
        m_param.scale = scale;
    }
}
