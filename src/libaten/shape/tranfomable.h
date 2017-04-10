#pragma once

#include "types.h"
#include "scene/bvh.h"
#include "math/mat4.h"

namespace aten
{
	class transformable : public bvhnode {
	public:
		transformable() {}
		virtual ~transformable() {}

	public:
		virtual SamplingPosNormalPdf getSamplePosNormalPdf(
			const mat4& mtxL2W,
			sampler* sampler) const = 0;

		virtual bool hit(
			const ray& r,
			const mat4& mtxL2W,
			real t_min, real t_max,
			hitrecord& rec) const = 0;
	};
}
