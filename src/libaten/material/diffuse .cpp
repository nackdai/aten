#include "material/diffuse.h"

namespace aten {
	real diffuse::pdf(const vec3& normal, const vec3& dir) const
	{
		auto c = dot(normal, dir);
		AT_ASSERT(c > AT_MATH_EPSILON);

		auto ret = c / AT_MATH_PI;
		
		return ret;
	}

	vec3 diffuse::sampleDirection(
		const vec3& in,
		const vec3& normal, 
		sampler* sampler) const
	{
		// normalの方向を基準とした正規直交基底(w, u, v)を作る.
		// この基底に対する半球内で次のレイを飛ばす.
		vec3 w, u, v;

		w = normal;

		// wと平行にならないようにする.
		if (fabs(w.x) > 0.1) {
			u = normalize(cross(vec3(0.0, 1.0, 0.0), w));
		}
		else {
			u = normalize(cross(vec3(1.0, 0.0, 0.0), w));
		}
		v = cross(w, u);

		// コサイン項を使った重点的サンプリング.
		const real r1 = 2 * AT_MATH_PI * sampler->nextSample();
		const real r2 = sampler->nextSample();
		const real r2s = sqrt(r2);

		const real x = aten::cos(r1) * r2s;
		const real y = aten::sin(r1) * r2s;
		const real z = aten::sqrt(CONST_REAL(1.0) - r2);

		vec3 dir = normalize((u * x + v * y + w * z));

		return std::move(dir);
	}

	vec3 diffuse::brdf(const vec3& normal, const vec3& dir) const
	{
		vec3 ret = m_color / AT_MATH_PI;
		return ret;
	}
}