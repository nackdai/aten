#include "camera/pinhole.h"

namespace AT_NAME {
	void PinholeCamera::init(
		const aten::vec3& origin,
		const aten::vec3& lookat,
		const aten::vec3& up,
		real vfov,	// vertical fov.
		uint32_t width, uint32_t height)
	{
		real theta = Deg2Rad(vfov);

		m_param.aspect = width / (real)height;

		real half_height = aten::tan(theta / 2);
		real half_width = m_param.aspect * half_height;

		m_param.origin = origin;

		// カメラ座標ベクトル.
		m_param.dir = normalize(lookat - origin);
		m_param.right = normalize(cross(m_param.dir, up));
		m_param.up = cross(m_param.right, m_param.dir);

		m_param.center = origin + m_param.dir;

		// スクリーンのUVベクトル.
		m_param.u = half_width * m_param.right;
		m_param.v = half_height * m_param.up;

		m_param.dist = height / (real(2.0) * aten::tan(theta / 2));

		m_param.width = width;
		m_param.height = height;
	}

	CameraSampleResult PinholeCamera::sample(
		real s, real t,
		aten::sampler* sampler) const
	{
		CameraSampleResult result;
		sample(&result, &m_param, s, t);
		return std::move(result);
	}

	AT_DEVICE_API void PinholeCamera::sample(
		CameraSampleResult* result,
		const aten::CameraParameter* param,
		real s, real t)
	{
		// [0, 1] -> [-1, 1]
		s = 2 * s - 1;
		t = 2 * t - 1;

		result->posOnLens = s * param->u + t * param->v;
		result->posOnLens = result->posOnLens + param->center;

		result->r.dir = normalize(result->posOnLens - param->origin);

		result->nmlOnLens = param->dir;
		result->posOnImageSensor = param->origin;

		result->r.org = param->origin;

		result->pdfOnLens = 1;
		result->pdfOnImageSensor = 1;
	}

	void PinholeCamera::revertRayToPixelPos(
		const aten::ray& ray,
		int& px, int& py) const
	{
		// dir 方向へのスクリーン距離.
		//     /|
		//  x / |
		//   /  |
		//  / θ|
		// +----+
		//    d
		// cosθ = x / d => x = d / cosθ

		real c = dot(ray.dir, m_param.dir);
		real dist = m_param.dist / c;

		aten::vec3 screenPos = m_param.origin + ray.dir * dist - m_param.center;

		real u = dot(screenPos, m_param.right) + m_param.width * real(0.5);
		real v = dot(screenPos, m_param.up) + m_param.height * real(0.5);

		px = (int)u;
		py = (int)v;
	}

	real PinholeCamera::convertImageSensorPdfToScenePdf(
		real pdfImage,	// Not used.
		const aten::vec3& hitPoint,
		const aten::vec3& hitpointNml,
		const aten::vec3& posOnImageSensor,
		const aten::vec3& posOnLens,
		const aten::vec3& posOnObjectPlane) const
	{
		real pdf = real(1) / (m_param.width * m_param.height);

		aten::vec3 v = hitPoint - posOnLens;

		{
			aten::vec3 dir = normalize(v);
			const real cosTheta = dot(dir, m_param.dir);
			const real dist = m_param.dist / (cosTheta + real(0.0001));
			const real dist2 = dist * dist;
			pdf = pdf / (cosTheta / dist2);
		}

		{
			aten::vec3 dv = hitPoint - posOnLens;
			const real dist2 = dv.squared_length();
			dv.normalize();
			const real c = dot(hitpointNml, dv);

			pdf = pdf * aten::abs(c / dist2);
		}

		return pdf;
	}

	real PinholeCamera::getWdash(
		const aten::vec3& hitPoint,
		const aten::vec3& hitpointNml,
		const aten::vec3& posOnImageSensor,
		const aten::vec3& posOnLens,
		const aten::vec3& posOnObjectPlane) const
	{
		const real W = real(1) / (m_param.width * m_param.height);

		aten::vec3 v = hitPoint - posOnLens;
		const real dist = v.length();
		v.normalize();

		// imagesensor -> lens
		const real c0 = dot(v, m_param.dir);
		const real d0 = m_param.dist / c0;
		const real G0 = c0 / (d0 * d0);

		// hitpoint -> camera
		const real c1 = dot(normalize(hitpointNml), -v);
		const real d1 = dist;
		const real G1 = c1 / (d1 * d1);

		real W_dash = W / G0 * G1;

		return W_dash;
	}

	real PinholeCamera::hitOnLens(
		const aten::ray& r,
		aten::vec3& posOnLens,
		aten::vec3& posOnObjectPlane,
		aten::vec3& posOnImageSensor,
		int& x, int& y) const
	{
		int px;
		int py;

		revertRayToPixelPos(r, px, py);

		if ((px >= 0) && (px < m_param.width)
			&& (py >= 0) && (py < m_param.height))
		{
			x = px;
			y = py;

			real u = (real)x / (real)m_param.width;
			real v = (real)y / (real)m_param.height;

			auto camsample = sample(u, v, nullptr);
			posOnLens = camsample.posOnLens;

			real lens_t = (posOnLens - r.org).length();

			return lens_t;
		}

		return -AT_MATH_INF;
	}
}