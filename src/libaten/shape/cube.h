#pragma once

#include "types.h"
#include "scene/bvh.h"
#include "math/mat4.h"
#include "shape/tranfomable.h"
#include "shape/shape.h"

namespace AT_NAME
{
	template<typename T> class instance;

	class cube : public aten::transformable {
		friend class instance<cube>;

	public:
		cube::cube(const aten::vec3& c, real w, real h, real d, material* m)
			: transformable(), m_param(c, aten::vec3(w, h, d), m)
		{}
		cube(real w, real h, real d, material* m)
			: cube(aten::vec3(0), w, h, d, m)
		{}

		virtual ~cube() {}

	public:
		virtual bool hit(
			const aten::ray& r,
			real t_min, real t_max,
			aten::hitrecord& rec) const override final;

		virtual aten::aabb getBoundingbox() const override final;

		const aten::vec3& center() const
		{
			return m_param.center;
		}

		const aten::vec3& size() const
		{
			return m_param.size;
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
		enum Face {
			POS_X,
			NEG_X,
			POS_Y,
			NEG_Y,
			POS_Z,
			NEG_Z,
		};

		Face onGetRandomPosOn(aten::vec3& pos, aten::sampler* sampler) const;

		static Face findFace(const aten::vec3& d);

	private:
		aten::ShapeParameter m_param;
	};
}
