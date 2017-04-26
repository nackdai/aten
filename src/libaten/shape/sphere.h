#pragma once

#include "types.h"
#include "accelerator/bvh.h"
#include "math/mat4.h"
#include "shape/tranformable.h"
#include "shape/shape.h"

namespace AT_NAME
{
	template<typename T> class instance;

	class sphere : public aten::transformable {
		friend class instance<sphere>;

	public:
		sphere(const aten::vec3& center, real radius, material* mtrl);
		sphere(real radius, material* mtrl)
			: sphere(aten::vec3(), radius, mtrl)
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
