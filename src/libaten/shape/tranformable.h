#pragma once

#include "types.h"
#include "accelerator/bvh.h"
#include "math/mat4.h"

#include <vector>

namespace aten
{
	class transformable : public bvhnode {
		static std::vector<transformable*> g_shapes;

	public:
		transformable();
		virtual ~transformable();

	public:
		virtual SamplingPosNormalPdf getSamplePosNormalPdf(
			const mat4& mtxL2W,
			sampler* sampler) const = 0;

		virtual bool hit(
			const ray& r,
			const mat4& mtxL2W,
			real t_min, real t_max,
			hitrecord& rec) const = 0;

		virtual const ShapeParameter& getParam() const
		{
			AT_ASSERT(false);
			return std::move(ShapeParameter(ShapeType::None));
		}

		virtual void getShapes(
			std::vector<ShapeParameter>& shapeparams,
			std::vector<PrimitiveParamter>& primparams) const
		{
			// Default, nothing is done...
		}

		static uint32_t getShapeNum();
		static const transformable* getShape(uint32_t idx);
		static int findShapeIdx(const transformable* shape);
		static int findShapeIdxAsHitable(const hitable* shape);
		static const std::vector<transformable*>& getShapes();
	};
}
