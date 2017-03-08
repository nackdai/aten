#pragma once

#include "types.h"
#include "scene/bvh.h"

namespace aten
{
	class cube : public bvhnode {
	public:
		cube() {}
		cube(const vec3& c, real w, real h, real d, material* m);

		virtual ~cube() {}

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

		const vec3& size() const
		{
			return m_size;
		}

		virtual vec3 getRandomPosOn(sampler* sampler) const override final;

	private:
		enum Face {
			POS_X,
			NEG_X,
			POS_Y,
			NEG_Y,
			POS_Z,
			NEG_Z,
		};

		static Face findFace(const vec3& d);

	private:
		vec3 m_center;
		vec3 m_size;
		aabb m_bbox;
		material* m_mtrl{ nullptr };
	};
}
