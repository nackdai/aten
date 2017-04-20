#include "light/light.h"

namespace aten {
	std::vector<Light*> Light::g_lights;

	uint32_t Light::getLightNum()
	{
		return (uint32_t)g_lights.size();
	}

	const Light* Light::getLight(uint32_t idx)
	{
		if (idx < g_lights.size()) {
			return g_lights[idx];
		}
		return nullptr;
	}

	const std::vector<Light*>& Light::getLights()
	{
		return g_lights;
	}

	Light::Light(const LightAttribute& attrib)
		: m_param(attrib)
	{
		g_lights.push_back(this);
	}

	Light::Light(const LightAttribute& attrib, Values& val)
		: m_param(attrib)
	{
		g_lights.push_back(this);

		m_param.pos = val.get("pos", m_param.pos);
		m_param.dir = val.get("dir", m_param.dir);
		m_param.le = val.get("le", m_param.le);
	}

	Light::~Light()
	{
		auto found = std::find(g_lights.begin(), g_lights.end(), this);
		if (found != g_lights.end()) {
			g_lights.erase(found);
		}
	}
}