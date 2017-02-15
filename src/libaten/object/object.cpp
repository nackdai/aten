#include "object/object.h"

namespace aten
{
	bool shape::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{

	}

	objinstance::objinstance(object* obj)
	{
		m_obj = obj;

		build(
			(hitable**)&m_obj->m_shapes[0], 
			m_obj->m_shapes.size());
	}

	bool objinstance::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		// TODO
		// Compute ray to objinstance coordinate.

		auto isHit = bvh::hit(r, t_min, t_max, rec);
		return isHit;
	}

	aabb objinstance::getBoundingbox() const
	{
		// TODO
		// Compute by transform matrix;

		auto box = m_obj->m_aabb;

		return std::move(box);
	}
}
