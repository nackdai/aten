#pragma once

#include "types.h"
#include "accelerator/bvh.h"
#include "math/mat4.h"
#include "shape/tranformable.h"
#include "shape/shape.h"

namespace AT_NAME
{
	template<typename T> class instance;

	class cube : public aten::transformable {
		friend class instance<cube>;

	public:
		cube(const aten::vec3& center, real w, real h, real d, material* mtrl);
		cube(real w, real h, real d, material* m)
			: cube(aten::vec3(0), w, h, d, m)
		{}

		virtual ~cube() {}

	public:
		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::Intersection& isect) const override final;

		const aten::vec3& center() const
		{
			return m_param.center;
		}

		const aten::vec3& size() const
		{
			return m_param.size;
		}

		virtual void getSamplePosNormalArea(
			aten::hitable::SamplePosNormalPdfResult* result,
			aten::sampler* sampler) const override final;

		virtual const aten::ShapeParameter& getParam() const override final
		{
			return m_param;
		}

	private:
		virtual void evalHitResult(
			const aten::ray& r,
			aten::hitrecord& rec, 
			const aten::Intersection& isect) const override final;

		virtual void evalHitResult(
			const aten::ray& r, 
			const aten::mat4& mtxL2W, 
			aten::hitrecord& rec,
			const aten::Intersection& isect) const override final;

		virtual void getSamplePosNormalArea(
			aten::hitable::SamplePosNormalPdfResult* result,
			const aten::mat4& mtxL2W,
			aten::sampler* sampler) const override final;

	private:
		enum Face {
			POS_X,
			NEG_X,
			POS_Y,
			NEG_Y,
			POS_Z,
			NEG_Z,
		};

		Face getRandomPosOn(aten::vec3& pos, aten::sampler* sampler) const;

		static Face findFace(const aten::vec3& d);

	private:
		aten::ShapeParameter m_param;
	};
}
