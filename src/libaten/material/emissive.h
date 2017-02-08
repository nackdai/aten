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

		virtual bool isEmissive() final
		{
			return true;
		}

		virtual vec3 color() const final
		{
			return m_emit;
		}

		virtual real pdf(const vec3& normal, const vec3& dir) const final
		{
			// NOTE
			// In this renderer, when path hit emissive material, tarcing finish.
			AT_ASSERT(false);
			return CONST_REAL(1.0);
		}

		virtual vec3 sampleDirection(const vec3& normal, sampler& sampler) const final
		{
			// NOTE
			// In this renderer, when path hit emissive material, tarcing finish.
			AT_ASSERT(false);
			return std::move(normal);
		}

	private:
		vec3 m_emit;
	};
}
