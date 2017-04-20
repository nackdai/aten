#pragma once

#include "types.h"
#include "scene/bvh.h"
#include "math/mat4.h"
#include "shape/tranfomable.h"
#include "shape/shape.h"

namespace aten
{
	template<typename T> class instance;

	class cube : public transformable {
		friend class instance<cube>;

	public:
		cube::cube(const vec3& c, real w, real h, real d, material* m)
			: m_param(c, vec3(w, h, d)), m_mtrl(m)
		{}
		cube(real w, real h, real d, material* m)
			: cube(vec3(0), w, h, d, m)
		{}

		virtual ~cube() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		virtual aabb getBoundingbox() const override final;

		const vec3& center() const
		{
			return m_param.center;
		}

		const vec3& size() const
		{
			return m_param.size;
		}

		virtual vec3 getRandomPosOn(sampler* sampler) const override final;

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(sampler* sampler) const override final;

	private:
		virtual bool hit(
			const ray& r,
			const mat4& mtxL2W,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		virtual SamplingPosNormalPdf getSamplePosNormalPdf(
			const mat4& mtxL2W,
			sampler* sampler) const override final;

	private:
		enum Face {
			POS_X,
			NEG_X,
			POS_Y,
			NEG_Y,
			POS_Z,
			NEG_Z,
		};

		Face onGetRandomPosOn(vec3& pos, sampler* sampler) const;

		static Face findFace(const vec3& d);

	private:
		ShapeParameter m_param;
		material* m_mtrl{ nullptr };
	};
}
