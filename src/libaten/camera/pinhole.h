#pragma once

#include "camera/camera.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "math/math.h"

namespace aten {
	class PinholeCamera : public camera {
	public:
		PinholeCamera() {}
		virtual ~PinholeCamera() {}

		void init(
			vec3 origin, vec3 lookat, vec3 up,
			real vfov,
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
		}

		virtual CameraSampleResult sample(
			real s, real t,
			sampler* sampler) const override final
		{
			CameraSampleResult result;

			// [0, 1] -> [-1, 1]
			s = 2 * s - 1;
			t = 2 * t - 1;

			auto screenPos = s * m_u + t * m_v;
			screenPos = screenPos + m_center;

			auto dirToScr = screenPos - m_origin;

			result.posOnLens = screenPos;
			result.posOnImageSensor = m_origin;
			result.r = ray(m_origin, dirToScr);

			return std::move(result);
		}

		virtual const vec3& getPos() const override final
		{
			return m_origin;
		}
		virtual const vec3& getDir() const override final
		{
			return m_dir;
		}

	private:
		vec3 m_origin;

		real m_aspect;
		vec3 m_center;

		vec3 m_u;
		vec3 m_v;

		vec3 m_dir;
		vec3 m_right;
		vec3 m_up;
	};
}