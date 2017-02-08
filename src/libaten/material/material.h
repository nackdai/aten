#pragma once

#include "types.h"
#include "math/vec3.h"
#include "sampler/sampler.h"

namespace aten
{
	class material {
	public:
		material() {}
		virtual ~material() {}

		virtual bool isEmissive()
		{
			return false;
		}

		virtual vec3 color() const = 0;

		virtual real pdf(const vec3& normal, const vec3& dir) const = 0;

		virtual vec3 sampleDirection(const vec3& normal, sampler& sampler) const = 0;
	};
}
