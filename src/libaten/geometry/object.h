#pragma once

#include <atomic>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/face.h"
#include "geometry/objshape.h"
#include "geometry/tranformable.h"

namespace AT_NAME
{
	template<typename T> class instance;

	class object : public aten::transformable {
		friend class instance<object>;

	public:
		object() : param(aten::GeometryType::Polygon) {}
		virtual ~object() {}

	public:
		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::Intersection& isect) const override final;

		virtual void evalHitResult(
			const aten::ray& r,
			const aten::mat4& mtxL2W,
			aten::hitrecord& rec,
			const aten::Intersection& isect) const override final;

		virtual void getPrimitives(aten::PrimitiveParamter* primparams) const override final;

		virtual const aten::GeomParameter& getParam() const override final
		{
			return param;
		}

		virtual aten::accelerator* getInternalAccelerator() override final
		{
			return m_accel;
		}

	private:
		void build();

		virtual void getSamplePosNormalArea(
			aten::hitable::SamplePosNormalPdfResult* result,
			const aten::mat4& mtxL2W, 
			aten::sampler* sampler) const override final;

	public:
		std::vector<objshape*> shapes;
		aten::GeomParameter param;
		aten::aabb bbox;

	private:
		aten::accelerator* m_accel{ nullptr };
		uint32_t m_triangles{ 0 };
	};
}
