#include "shape/sphere.h"

namespace AT_NAME
{
	static AT_DEVICE_API void getUV(real& u, real& v, const aten::vec3& p)
	{
		auto phi = aten::asin(p.y);
		auto theta = aten::atan(p.x / p.z);

		u = (theta + AT_MATH_PI_HALF) / AT_MATH_PI;
		v = (phi + AT_MATH_PI_HALF) / AT_MATH_PI;
	}

	sphere::sphere(const aten::vec3& center, real radius, material* mtrl)
		: transformable(), m_param(center, radius, mtrl)
	{
		auto _min = center - radius;
		auto _max = center + radius;

		m_aabb.init(_min, _max);
	}

	bool sphere::hit(
		const aten::ray& r,
		real t_min, real t_max,
		aten::hitrecord& rec,
		aten::hitrecordOption& recOpt) const
	{
		bool isHit = hit(&m_param, r, t_min, t_max, &rec);

		if (isHit) {
			rec.obj = (hitable*)this;
			rec.mtrl = (material*)m_param.mtrl.ptr;
		}

		return isHit;
	}

	bool AT_DEVICE_API sphere::hit(
		const aten::ShapeParameter* param,
		const aten::ray& r,
		real t_min, real t_max,
		aten::hitrecord* rec)
	{
		// NOTE
		// https://www.slideshare.net/h013/edupt-kaisetsu-22852235
		// p52 - p58

		const aten::vec3 p_o = param->center - r.org;
		const real b = dot(p_o, r.dir);

		// ”»•ÊŽ®.
		const real D4 = b * b - dot(p_o, p_o) + param->radius * param->radius;

		if (D4 < real(0)) {
			return false;
		}

		const real sqrt_D4 = aten::sqrt(D4);
		const real t1 = b - sqrt_D4;
		const real t2 = b + sqrt_D4;

#if 0
		if (t1 > AT_MATH_EPSILON) {
			rec->t = t1;
		}
		else if (t2 > AT_MATH_EPSILON) {
			rec->t = t2;
		}
		else {
			return false;
		}
#elif 1
		bool close = aten::isClose(aten::abs(b), sqrt_D4, 25000);

		if (t1 > AT_MATH_EPSILON && !close) {
			rec->t = t1;
		}
		else if (t2 > AT_MATH_EPSILON && !close) {
			rec->t = t2;
		}
		else {
			return false;
		}
#else
		if (t1 < 0 && t2 < 0) {
			return false;
		}
		else if (t1 > 0 && t2 > 0) {
			rec->t = aten::cmpMin(t1, t2);
		}
		else {
			rec->t = aten::cmpMax(t1, t2);
		}
#endif

		return true;
	}

	void sphere::evalHitResult(
		const aten::ray& r, 
		aten::hitrecord& rec,
		const aten::hitrecordOption& recOpt) const
	{
		evalHitResult(&m_param, r, aten::mat4(), &rec);
	}

	void sphere::evalHitResult(
		const aten::ray& r,
		const aten::mat4& mtxL2W,
		aten::hitrecord& rec,
		const aten::hitrecordOption& recOpt) const
	{
		evalHitResult(&m_param, r, mtxL2W, &rec);
	}

	void sphere::evalHitResult(
		const aten::ShapeParameter* param, 
		const aten::ray& r, 
		aten::hitrecord* rec)
	{
		evalHitResult(param, r, aten::mat4(), rec);
	}

	void sphere::evalHitResult(
		const aten::ShapeParameter* param,
		const aten::ray& r,
		const aten::mat4& mtxL2W, 
		aten::hitrecord* rec)
	{
		rec->p = r.org + rec->t * r.dir;
		rec->normal = (rec->p - param->center) / param->radius; // ³‹K‰»‚µ‚Ä–@ü‚ð“¾‚é

		// tangent coordinate.
		rec->du = normalize(getOrthoVector(rec->normal));
		rec->dv = normalize(cross(rec->normal, rec->du));

		{
			auto tmp = param->center + aten::make_float3(param->radius, 0, 0);

			auto center = mtxL2W.apply(param->center);
			tmp = mtxL2W.apply(tmp);

			auto radius = (tmp - center).length();

			rec->area = 4 * AT_MATH_PI * radius * radius;
		}

		getUV(rec->u, rec->v, rec->normal);
	}

	void sphere::getSamplePosNormalArea(
		aten::hitable::SamplePosNormalPdfResult* result,
		aten::sampler* sampler) const
	{
		return getSamplePosNormalArea(result, aten::mat4::Identity, sampler);
	}

	AT_DEVICE_API void sphere::getSamplePosNormalArea(
		aten::hitable::SamplePosNormalPdfResult* result,
		const aten::ShapeParameter* param,
		aten::sampler* sampler)
	{
		getSamplePosNormalArea(result, param, aten::mat4(), sampler);
	}

	void sphere::getSamplePosNormalArea(
		aten::hitable::SamplePosNormalPdfResult* result,
		const aten::mat4& mtxL2W,
		aten::sampler* sampler) const
	{
		getSamplePosNormalArea(result, &m_param, mtxL2W, sampler);
	}

	void sphere::getSamplePosNormalArea(
		aten::hitable::SamplePosNormalPdfResult* result,
		const aten::ShapeParameter* param,
		const aten::mat4& mtxL2W,
		aten::sampler* sampler)
	{
		auto r1 = sampler->nextSample();
		auto r2 = sampler->nextSample();

		auto r = param->radius;

		auto z = real(2) * r1 - real(1); // [0,1] -> [-1, 1]

		auto sin_theta = aten::sqrt(1 - z * z);
		auto phi = 2 * AT_MATH_PI * r2;

		auto x = aten::cos(phi) * sin_theta;
		auto y = aten::sin(phi) * sin_theta;

		aten::vec3 dir = aten::make_float3(x, y, z);
		dir.normalize();

		auto p = dir * (r + AT_MATH_EPSILON);

		result->pos = param->center + p;

		result->nml = normalize(result->pos - param->center);

		result->area = real(1);
		{
			auto tmp = param->center + aten::make_float3(param->radius, 0, 0);

			auto center = mtxL2W.apply(param->center);
			tmp = mtxL2W.apply(tmp);

			auto radius = (tmp - center).length();

			result->area = 4 * AT_MATH_PI * radius * radius;
		}
	}
}
