#pragma once

#include "types.h"
#include "scene/bvh.h"

namespace aten
{
	class sphere : public bvhnode {
	public:
		sphere() {}
		sphere(const vec3& c, real r, material* m)
			: m_center(c), m_radius(r), m_mtrl(m)
		{};

		virtual ~sphere() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override final;

		virtual aabb getBoundingbox() const override final;

		const vec3& center() const
		{
			return m_center;
		}

		real radius() const
		{
			return m_radius;
		}

		virtual vec3 getRandomPosOn(sampler* sampler) const override final;

		virtual vec3 getRandomPosOn(real r1, real r2) const override final;

	private:
		vec3 m_center;
		real m_radius{ real(0) };
		material* m_mtrl{ nullptr };
	};
}
