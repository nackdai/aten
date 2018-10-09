#include "light/light.h"

namespace AT_NAME {
    Light::Light(aten::LightType type, const aten::LightAttribute& attrib)
        : m_param(type, attrib)
    {
    }

    Light::Light(aten::LightType type, const aten::LightAttribute& attrib, aten::Values& val)
        : m_param(type, attrib)
    {
        m_param.pos = val.get("pos", m_param.pos);
        m_param.dir = val.get("dir", m_param.dir);
        m_param.le = val.get("le", m_param.le);
    }

    Light::~Light()
    {
    }
}