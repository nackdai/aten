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
		virtual void getSamplePosNormalArea(
			aten::hitable::SamplePosNormalPdfResult* result,
			const mat4& mtxL2W,
			sampler* sampler) const = 0;

		virtual void evalHitResult(
			const ray& r,
			const mat4& mtxL2W,
			hitrecord& rec) const = 0;

		virtual const ShapeParameter& getParam() const
		{
			AT_ASSERT(false);
			return std::move(ShapeParameter(ShapeType::ShapeTypeMax));
		}

		virtual void getPrimitives(std::vector<PrimitiveParamter>& primparams) const
		{
			// Nothing is done...
		}

		static uint32_t getShapeNum();
		static const transformable* getShape(uint32_t idx);
		static int findShapeIdx(const transformable* shape);
		static int findShapeIdxAsHitable(const hitable* shape);
		static const std::vector<transformable*>& getShapes();
	};
}
