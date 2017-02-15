#include "primitive/sphere.h"

namespace aten
{
	bool sphere::hit(
		const ray& r, 
		real t_min, real t_max,
		hitrecord& rec) const
	{
		const vec3 p_o = m_center - r.org;
		const real b = dot(p_o, r.dir);

		// ”»•Ê®.
		const real D4 = b * b - dot(p_o, p_o) + m_radius * m_radius;

		if (D4 < CONST_REAL(0.0)) {
			return false;
		}

		const real sqrt_D4 = aten::sqrt(D4);
		const real t1 = b - sqrt_D4;
		const real t2 = b + sqrt_D4;

		if (t1 < AT_MATH_EPSILON && t2 < AT_MATH_EPSILON) {
			return false;
		}

		if (t1 > AT_MATH_EPSILON) {
			rec.t = t1;
		}
		else {
			rec.t = t2;
		}

		rec.p = r.org + rec.t * r.dir;
		rec.normal = (rec.p - m_center) / m_radius; // ³‹K‰»‚µ‚Ä–@ü‚ğ“¾‚é
		rec.obj = (hitable*)this;
		rec.mtrl = m_mtrl;

		return true;
	}

	aabb sphere::getBoundingbox() const
	{
		vec3 _min = m_center - m_radius;
		vec3 _max = m_center + m_radius;

		aabb ret(_min, _max);

		return std::move(ret);
	}
}
