#include "light/arealight.h"
#include "geometry/transformable.h"

namespace AT_NAME {
    AreaLight::AreaLight(const std::shared_ptr<aten::transformable>& obj, const aten::vec3& le)
        : Light(aten::LightType::Area, aten::LightAttributeArea)
    {
        m_obj = obj;

        m_param.objid = obj->id();
        m_param.le = le;
    }
}
