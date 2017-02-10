#pragma once

#include "material/material.h"

namespace aten
{
	class emissive : public material {
	public:
		emissive() {}
		emissive(const vec3& e)
			: m_emit(e)
		{}

		virtual ~emissive() {}

		virtual bool isEmissive() override final
		{
			return true;
		}

		virtual vec3 color() const override final
		{
			return m_emit;
		}

		virtual real pdf(const vec3& normal, const vec3& dir) const override final
		{
			// NOTE
			// In this renderer, when path hit emissive material, tarcing finish.
			AT_ASSERT(false);
			return CONST_REAL(1.0);
		}

		virtual vec3 sampleDirection(const vec3& normal, sampler* sampler) const override final
		{
			// NOTE
			// In this renderer, when path hit emissive material, tarcing finish.
			AT_ASSERT(false);
			return std::move(normal);
		}

		virtual vec3 brdf(const vec3& normal, const vec3& dir) const override final
		{
			// NOTE
			// In this renderer, when path hit emissive material, tarcing finish.
			AT_ASSERT(false);
			return std::move(vec3());
		}

	private:
		vec3 m_emit;
	};
}
