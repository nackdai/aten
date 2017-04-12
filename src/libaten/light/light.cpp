#include "light/light.h"

namespace aten {
#define COPY_VEC3_FLOAT3(a, v) \
	a[0] = (float)v.r;\
	a[1] = (float)v.g;\
	a[2] = (float)v.b;

	void Light::serialize(const Light* light, LightParameter& param)
	{
		COPY_VEC3_FLOAT3(param.pos, light->m_pos);
		COPY_VEC3_FLOAT3(param.dir, light->m_dir);
		COPY_VEC3_FLOAT3(param.le, light->m_le);
	}
}