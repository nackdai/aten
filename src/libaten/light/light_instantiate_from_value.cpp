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
        m_param.pos = val.get("pos", m_param.pos);
        m_param.dir = val.get("dir", m_param.dir);
        m_param.le = val.get("le", m_param.le);
    }

    AreaLight::AreaLight(const aten::Values& val)
        : Light(aten::LightType::Area, LightAttributeArea, val)
    {
        m_obj = static_cast<std::shared_ptr<aten::transformable>>(val.get("object", m_obj));

        m_param.objid = (m_obj ? m_obj->id() : -1);
        m_param.le = val.get("color", m_param.le);
    }

    DirectionalLight::DirectionalLight(aten::Values& val)
        : Light(aten::LightType::Direction, LightAttributeDirectional, val)
    {}

    ImageBasedLight::ImageBasedLight(aten::Values& val)
        : Light(aten::LightType::IBL, LightAttributeIBL, val)
    {
        auto tex = val.get<texture>("envmap");
        AT_ASSERT(tex);

        auto bg = std::make_shared<AT_NAME::envmap>();
        bg->init(tex);

        setEnvMap(bg);
    }

    PointLight::PointLight(aten::Values& val)
        : Light(aten::LightType::Point, LightAttributeSingluar, val)
    {
        m_param.constAttn = val.get("constAttn", m_param.constAttn);
        m_param.linearAttn = val.get("linearAttn", m_param.linearAttn);
        m_param.expAttn = val.get("expAttn", m_param.expAttn);

        setAttenuation(m_param.constAttn, m_param.linearAttn, m_param.expAttn);
    }

    SpotLight::SpotLight(aten::Values& val)
        : Light(aten::LightType::Spot, LightAttributeSingluar, val)
    {
        m_param.constAttn = val.get("constAttn", m_param.constAttn);
        m_param.linearAttn = val.get("linearAttn", m_param.linearAttn);
        m_param.expAttn = val.get("expAttn", m_param.expAttn);

        setAttenuation(m_param.constAttn, m_param.linearAttn, m_param.expAttn);

        m_param.innerAngle = val.get("innerAngle", m_param.innerAngle);
        m_param.outerAngle = val.get("outerAngle", m_param.outerAngle);
        m_param.falloff = val.get("falloff", m_param.falloff);

        setSpotlightFactor(m_param.innerAngle, m_param.outerAngle, m_param.falloff);
    }
}
