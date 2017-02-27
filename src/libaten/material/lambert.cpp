#include "material/lambert.h"

namespace aten {
	real lambert::pdf(
		const vec3& normal, 
		const vec3& wi,
		const vec3& wo) const
	{
		auto c = dot(normal, wo);

		//AT_ASSERT(c > AT_MATH_EPSILON);
		c = aten::abs(c);

		auto ret = c / AT_MATH_PI;
		
		return ret;
	}

	vec3 lambert::sampleDirection(
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
		const real z = aten::sqrt(real(1) - r2);

		vec3 dir = normalize((u * x + v * y + w * z));

		return std::move(dir);
	}

	vec3 lambert::bsdf(
		const vec3& normal,
		const vec3& wi,
		const vec3& wo,
		real u, real v) const
	{
		vec3 albedo = m_color;

		if (m_tex) {
			auto texclr = m_tex->at(u, v);
			albedo *= texclr;
		}

		vec3 ret = albedo / AT_MATH_PI;
		return ret;
	}

	material::sampling lambert::sample(
		const vec3& in,
		const vec3& normal,
		const hitrecord& hitrec,
		sampler* sampler,
		real u, real v) const
	{
		sampling ret;

		ret.dir = sampleDirection(in, normal, sampler);
		ret.pdf = pdf(normal, in, ret.dir);
		ret.bsdf = bsdf(normal, in, ret.dir, u, v);

		return std::move(ret);
	}
}