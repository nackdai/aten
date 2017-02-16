#include "object/object.h"
#include "math/intersect.h"

namespace aten
{
	bool shape::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		// TODO
		// ‚³‚ç‚ÉBVH‚·‚é‚©...

		bool isHit = false;

		for (uint32_t i = 0; i < faces.size(); i++) {
			const auto& f = faces[i];

			const auto& v0 = vertices[f.idx[0]];
			const auto& v1 = vertices[f.idx[1]];
			const auto& v2 = vertices[f.idx[2]];

			const auto res = intersertTriangle(r, v0.pos, v1.pos, v2.pos);

			if (res.isIntersect) {
				if (res.t < rec.t) {
					rec.t = res.t;

					rec.normal = v0.nml;
					rec.p = r.org + rec.t * r.dir;

					rec.obj = (hitable*)this;
					rec.mtrl = mtrl;

					isHit = true;
				}
			}
		}

		return isHit;
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

		auto isHit = bvhnode::hit(r, t_min, t_max, rec);
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
