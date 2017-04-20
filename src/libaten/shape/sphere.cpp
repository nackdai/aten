#include "shape/sphere.h"

namespace aten
{
	static void getUV(real& u, real& v, const vec3& p)
	{
		auto phi = aten::asin(p.y);
		auto theta = aten::atan(p.x / p.z);

		u = (theta + AT_MATH_PI_HALF) / AT_MATH_PI;
		v = (phi + AT_MATH_PI_HALF) / AT_MATH_PI;
	}

	bool sphere::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		bool isHit = hit(m_param, r, mat4::Identity, t_min, t_max, rec);

		if (isHit) {
			rec.obj = (hitable*)this;
			rec.mtrl = m_mtrl;
		}

		return isHit;
	}

	bool sphere::hit(
		const ray& r,
		const mat4& mtxL2W,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		bool isHit = hit(m_param, r, mtxL2W, t_min, t_max, rec);

		if (isHit) {
			rec.obj = (hitable*)this;
			rec.mtrl = m_mtrl;
		}

		return isHit;
	}

	bool sphere::hit(
		const ShapeParameter& param,
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec)
	{
		return hit(param, r, mat4::Identity, t_min, t_max, rec);
	}

	bool sphere::hit(
		const ShapeParameter& param,
		const ray& r,
		const mat4& mtxL2W,
		real t_min, real t_max,
		hitrecord& rec)
	{
		// NOTE
		// https://www.slideshare.net/h013/edupt-kaisetsu-22852235
		// p52 - p58

		const vec3 p_o = param.center - r.org;
		const real b = dot(p_o, r.dir);

		// ”»•ÊŽ®.
		const real D4 = b * b - dot(p_o, p_o) + param.radius * param.radius;

		if (D4 < real(0)) {
			return false;
		}

		const real sqrt_D4 = aten::sqrt(D4);
		const real t1 = b - sqrt_D4;
		const real t2 = b + sqrt_D4;

		if (t1 > AT_MATH_EPSILON) {
			rec.t = t1;
		}
		else if (t2 > AT_MATH_EPSILON) {
			rec.t = t2;
		}
		else {
			return false;
		}

		rec.p = r.org + rec.t * r.dir;
		rec.normal = (rec.p - param.center) / param.radius; // ³‹K‰»‚µ‚Ä–@ü‚ð“¾‚é

		// tangent coordinate.
		rec.du = normalize(getOrthoVector(rec.normal));
		rec.dv = normalize(cross(rec.normal, rec.du));

		{
			auto tmp = param.center + vec3(param.radius, 0, 0);

			auto center = mtxL2W.apply(param.center);
			tmp = mtxL2W.apply(tmp);

			auto radius = (tmp - center).length();

			rec.area = 4 * AT_MATH_PI * radius * radius;
		}

		getUV(rec.u, rec.v, rec.normal);

		return true;
	}

	aabb sphere::getBoundingbox() const
	{
		return std::move(m_param.bbox);
	}

	vec3 sphere::getRandomPosOn(sampler* sampler) const
	{
		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();

		auto r = m_param.radius;

		auto z = real(2) * r1 - real(1); // [0,1] -> [-1, 1]

		auto sin_theta = aten::sqrt(1 - z * z);
		auto phi = 2 * AT_MATH_PI * r2;

		auto x = aten::cos(phi) * sin_theta;
		auto y = aten::sin(phi) * sin_theta;

		vec3 dir(x, y, z);
		dir.normalize();

		auto p = dir * (r + AT_MATH_EPSILON);

		vec3 posOnSphere = m_param.center + p;

		return std::move(posOnSphere);
	}

	hitable::SamplingPosNormalPdf sphere::getSamplePosNormalPdf(sampler* sampler) const
	{
		return getSamplePosNormalPdf(mat4::Identity, sampler);
	}

	hitable::SamplingPosNormalPdf sphere::getSamplePosNormalPdf(
		const mat4& mtxL2W,
		sampler* sampler) const
	{
		auto p = getRandomPosOn(sampler);
		auto n = normalize(p - m_param.center);

		real area = real(1);
		{
			auto tmp = m_param.center + vec3(m_param.radius, 0, 0);

			auto center = mtxL2W.apply(m_param.center);
			tmp = mtxL2W.apply(tmp);

			auto radius = (tmp - center).length();

			area = 4 * AT_MATH_PI * radius * radius;
		}

		return std::move(hitable::SamplingPosNormalPdf(p + n * AT_MATH_EPSILON, n, real(1) / area));
	}
}
