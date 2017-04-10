#include "camera/pinhole.h"
namespace aten {
	void PinholeCamera::init(
		vec3 origin, vec3 lookat, vec3 up,
		real vfov,	// vertical fov.
		uint32_t width, uint32_t height)
	{
		real theta = Deg2Rad(vfov);

		m_aspect = width / (real)height;

		real half_height = aten::tan(theta / 2);
		real half_width = m_aspect * half_height;

		m_origin = origin;

		// カメラ座標ベクトル.
		m_dir = normalize(lookat - origin);
		m_right = normalize(cross(m_dir, up));
		m_up = cross(m_right, m_dir);

		m_center = origin + m_dir;

		// スクリーンのUVベクトル.
		m_u = half_width * m_right;
		m_v = half_height * m_up;

		m_dist = height / (2.0 * aten::tan(theta / 2));

		m_width = width;
		m_height = height;
	}

	CameraSampleResult PinholeCamera::sample(
		real s, real t,
		sampler* sampler) const
	{
		CameraSampleResult result;

		// [0, 1] -> [-1, 1]
		s = 2 * s - 1;
		t = 2 * t - 1;

		auto screenPos = s * m_u + t * m_v;
		screenPos = screenPos + m_center;

		auto dirToScr = screenPos - m_origin;

		result.posOnLens = screenPos;
		result.nmlOnLens = m_dir;
		result.posOnImageSensor = m_origin;
		result.r = ray(m_origin, dirToScr);

		result.pdfOnLens = 1;
		result.pdfOnImageSensor = 1;

		return std::move(result);
	}

	void PinholeCamera::revertRayToPixelPos(
		const ray& ray,
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

		real c = dot(ray.dir, m_dir);
		real dist = m_dist / c;

		vec3 screenPos = m_origin + ray.dir * dist - m_center;

		real u = dot(screenPos, m_right) + m_width * 0.5;
		real v = dot(screenPos, m_up) + m_height * 0.5;

		px = (int)u;
		py = (int)v;
	}

	real PinholeCamera::getPdfImageSensorArea(
		const vec3& hitPoint,
		const vec3& hitpointNml,
		const vec3& posOnImageSensor,
		const vec3& posOnLens,
		const vec3& posOnObjectPlane) const
	{
		real pdf = real(1) / (m_width * m_height);

		vec3 v = hitPoint - posOnLens;

		vec3 dir = normalize(v);
		const real cosTheta = dot(dir, m_dir);
		const real dist = m_dist / (cosTheta + real(0.0001));
		const real dist2 = dist * dist;
		pdf = pdf / (cosTheta / dist2);

		return pdf;
	}

	real PinholeCamera::getWdash(
		const vec3& hitPoint,
		const vec3& hitpointNml,
		const vec3& posOnImageSensor,
		const vec3& posOnLens,
		const vec3& posOnObjectPlane) const
	{
		const real W = real(1) / (m_width * m_height);

		vec3 v = hitPoint - posOnLens;
		const real dist = v.length();
		v.normalize();

		// imagesensor -> lens
		const real c0 = dot(v, m_dir);
		const real d0 = m_dist / c0;
		const real G0 = c0 / (d0 * d0);

		// hitpoint -> camera
		const real c1 = dot(normalize(hitpointNml), -v);
		const real d1 = dist;
		const real G1 = c1 / (d1 * d1);

		real W_dash = W / G0 * G1;

		return W_dash;
	}
}