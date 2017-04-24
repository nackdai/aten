#pragma once

#include "types.h"
#include "scene/bvh.h"
#include "math/mat4.h"
#include "shape/tranfomable.h"
#include "shape/shape.h"

namespace AT_NAME
{
	template<typename T> class instance;

	class sphere : public aten::transformable {
		friend class instance<sphere>;

	public:
		sphere(const aten::vec3& c, real r, material* m)
			: transformable(), m_param(c, r, m)
		{}
		sphere(real r, material* m)
			: sphere(aten::vec3(), r, m)
		{}

		virtual ~sphere() {}

	public:
		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec) const override final;

		static AT_DEVICE_API bool hit(
			const aten::ShapeParameter& param,
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec);

		static AT_DEVICE_API bool hit(
			const aten::ShapeParameter& param,
			const aten::ray& r,
			const aten::mat4& mtxL2W,
			real t_min, real t_max,
			aten::hitrecord& rec);

		virtual aten::aabb getBoundingbox() const override final;

		const aten::vec3& center() const
		{
			return m_param.center;
		}

		real radius() const
		{
			return m_param.radius;
		}

		virtual aten::vec3 getRandomPosOn(aten::sampler* sampler) const override final;

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(aten::sampler* sampler) const override final;

		virtual const aten::ShapeParameter& getParam() const override final
		{
			return m_param;
		}

	private:
		virtual bool hit(
			const aten::ray& r,
			const aten::mat4& mtxL2W,
			real t_min, real t_max,
			aten::hitrecord& rec) const override final;

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(
			const aten::mat4& mtxL2W,
			aten::sampler* sampler) const override final;

	private:
		aten::ShapeParameter m_param;
	};
}
