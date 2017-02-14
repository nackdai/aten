#include "material/refraction.h"

namespace aten
{
	real refraction::pdf(const vec3& normal, const vec3& dir) const
	{
		AT_ASSERT(false);

		auto ret = CONST_REAL(1.0);
		return ret;
	}

	vec3 refraction::sampleDirection(
		const vec3& in,
		const vec3& normal,
		sampler* sampler) const
	{
		AT_ASSERT(false);

		auto reflect = in - 2 * dot(normal, in) * normal;
		reflect.normalize();

		return std::move(reflect);
	}

	vec3 refraction::brdf(const vec3& normal, const vec3& dir) const
	{
		AT_ASSERT(false);

		return std::move(m_color);
	}

	// NOTE
	// Schlick によるフレネル反射率の近似.
	// http://yokotakenji.me/log/math/4501/
	// https://en.wikipedia.org/wiki/Schlick%27s_approximation

	// NOTE
	// フレネル反射率について.
	// http://d.hatena.ne.jp/hanecci/20130525/p3

	real schlick(
		const vec3& in, 
		const vec3& normal,
		real ni, real nt)
	{
		// NOTE
		// Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
		// R0 = ((n1 - n2) / (n1 + n2))^2

		auto r0 = (ni - nt) / (ni + nt);
		r0 = r0 * r0;

		auto c = dot(-in, normal);

		return r0 + (1 - r0) * aten::pow((1 - c), 5);
	}

	material::sampling refraction::sample(
		const vec3& in,
		const vec3& normal,
		sampler* sampler) const
	{
		sampling ret;

		// レイが入射してくる側の物体の屈折率.
		real ni = CONST_REAL(1.0);	// 真空

		// 物体内部の屈折率.
		real nt = m_nt;

		real cos_i = dot(in, normal);

		bool isEnter = (cos_i > CONST_REAL(0.0));

		vec3 n = normal;

		if (isEnter) {
			// レイが出ていくので、全部反対.
			nt = CONST_REAL(1.0);
			ni = m_nt;

			cos_i = -cos_i;
			n = -n;
		}

		real ni_nt = ni / nt;

		// NOTE
		// cos_t^2 = 1 - sin_t^2
		// sin_t^2 = (ni/nt)^2 * sin_i^2 = (ni/nt)^2 * (1 - cos_i^2)
		// sin_i / sin_t = nt/ni -> sin_t = (ni/nt) * sin_i = (ni/nt) * sqrt(1 - cos_i)
		real cos_t_2 = CONST_REAL(1.0) - (ni_nt * ni_nt) * (CONST_REAL(1.0) - cos_i * cos_i);

		if (cos_t_2 < CONST_REAL(0.0)) {
			// 全反射.
			ret.pdf = CONST_REAL(1.0);

			ret.dir = in - 2 * dot(normal, in) * normal;
			ret.dir.normalize();

			auto c = dot(normal, in);
			if (c > CONST_REAL(0.0)) {
				ret.brdf = m_color / c;
			}

			return std::move(ret);
		}

		// NOTE
		// https://www.vcl.jp/~kanazawa/raytracing/?page_id=478

		auto nt_ni_2 = (nt / ni) * (nt / ni);
		auto cos_i_2 = cos_i * cos_i;
		vec3 refract = ni_nt * in - (aten::sqrt(nt_ni_2 - (1 - cos_i_2)) - cos_i) * n;
		refract.normalize();

		// 反射方向の光が反射してray.dirの方向に運ぶ割合。同時に屈折方向の光が反射する方向に運ぶ割合.
		auto fresnel = schlick(in, n, ni, nt);

		// 屈折なので.
		fresnel = CONST_REAL(1.0) - fresnel;

		// レイの運ぶ放射輝度は屈折率の異なる物体間を移動するとき、屈折率の比の二乗の分だけ変化する.
		real nn = ni_nt * ni_nt;

		auto cos_t = aten::sqrt(cos_t_2);

		ret.pdf = CONST_REAL(1.0);
		ret.dir = refract;
		ret.brdf = fresnel * nn * m_color / cos_t;
		ret.into = true;

		return std::move(ret);
	}
}
