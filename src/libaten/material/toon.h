#pragma once

#include <functional>
#include "material/material.h"
#include "light/light.h"

namespace aten
{
	class toon : public NPRMaterial {
	public:
		using ComputeToonShadeFunc = std::function<real(real)>;

		toon() {}
		toon(const vec3& e, Light* light)
			: NPRMaterial(e, light)
		{
		}
		toon(const vec3& e, Light* light, ComputeToonShadeFunc func)
			: NPRMaterial(e, light)
		{
			setComputeToonShadeFunc(func);
		}

		toon(Values& val)
			: NPRMaterial(val)
		{}

		virtual ~toon() {}

	public:
		virtual real pdf(
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v) const override final
		{
			return real(1);
		}

		virtual vec3 sampleDirection(
			const ray& ray,
			const vec3& normal,
			real u, real v,
			sampler* sampler) const override final;

		virtual vec3 bsdf(
			const vec3& normal,
			const vec3& wi,
			const vec3& wo,
			real u, real v) const override final;

		virtual sampling sample(
			const ray& ray,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v) const override final;

		void setComputeToonShadeFunc(ComputeToonShadeFunc func)
		{
			m_func = func;
		}

		virtual void serialize(MaterialParam& param) const override final
		{
			// TODO
			// Not supported...
			AT_ASSERT(false);
		}

	private:
		virtual vec3 bsdf(
			real cosShadow,
			real u, real v) const override final;

	private:
		// TODO
		// GPGPUâªÇ∑ÇÈèÍçáÇÕçlÇ¶Ç»Ç¢Ç∆Ç¢ÇØÇ»Ç¢....
		ComputeToonShadeFunc m_func;
	};
}
