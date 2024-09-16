#include "light/arealight.h"
#include "geometry/transformable.h"

namespace AT_NAME {
    AreaLight::AreaLight(
        const std::shared_ptr<aten::transformable>& obj,
        const aten::vec3& light_color,
        float lux)
        : Light(aten::LightType::Area, aten::LightAttributeArea)
    {
        m_obj = obj;

        m_param.arealight_objid = obj->id();
        m_param.light_color = light_color;

        // Convert lux[W] to intensity[W/sr].
        // Diffuse light and sample hemisphere uniformliy.
        m_param.intensity = lux / (2.0f * AT_MATH_PI_2);
    }
}
