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
			hitrecord& rec,
			const Intersection& isect) const = 0;

		virtual const ShapeParameter& getParam() const
		{
			AT_ASSERT(false);
			return std::move(ShapeParameter(ShapeType::ShapeTypeMax));
		}

		virtual void getPrimitives(PrimitiveParamter* primparams) const
		{
			// Nothing is done...
		}

		virtual const hitable* getHasObject() const
		{
			return nullptr;
		}

		virtual void getMatrices(
			aten::mat4& mtxL2W,
			aten::mat4& mtxW2L) const
		{
			AT_ASSERT(false);
		}

		int id() const
		{
			return m_id;
		}

		static uint32_t getShapeNum();
		static transformable* getShape(uint32_t idx);
		static int findShapeIdx(const transformable* shape);
		static int findShapeIdxAsHitable(const hitable* shape);
		static const std::vector<transformable*>& getShapes();

	private:
		int m_id{ -1 };
	};
}
