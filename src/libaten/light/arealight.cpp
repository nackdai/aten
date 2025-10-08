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
        param_.arealight_objid = obj->id();
        param_.light_color = light_color;
        param_.intensity = intensity;
        param_.scale = scale;
    }
}
