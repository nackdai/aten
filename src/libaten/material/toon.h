#pragma once

#include <functional>
#include "material/material.h"
#include "light/light.h"

namespace aten
{
	class toon : public NPRMaterial {
	public:
		using ComputeToonShadeFunc = std::function<real(real)>;

		toon(const vec3& e, AT_NAME::Light* light)
			: NPRMaterial(MaterialType::Toon, e, light)
		{
		}
		toon(const vec3& e, AT_NAME::Light* light, ComputeToonShadeFunc func)
			: NPRMaterial(MaterialType::Toon, e, light)
		{
			setComputeToonShadeFunc(func);
		}

		toon(Values& val)
			: NPRMaterial(MaterialType::Toon, val)
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

		virtual MaterialSampling sample(
			const ray& ray,
			const vec3& normal,
			const hitrecord& hitrec,
			sampler* sampler,
			real u, real v,
			bool isLightPath = false) const override final;

		void setComputeToonShadeFunc(ComputeToonShadeFunc func)
		{
			m_func = func;
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
