#pragma once

#include "camera/camera.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "math/math.h"

namespace aten {
	class EquirectCamera : public camera {
	public:
		EquirectCamera() {}
		virtual ~EquirectCamera() {}

		void init(
			vec3 origin, vec3 lookat, vec3 up,
			uint32_t width, uint32_t height)
		{
			auto aspect = width / (real)height;
			AT_ASSERT(aspect == 2);

			m_origin = origin;

			// カメラ座標ベクトル.
			m_dir = normalize(lookat - origin);
			m_right = normalize(cross(m_dir, up));
			m_up = cross(m_right, m_dir);
		}

		virtual CameraSampleResult sample(
			real s, real t,
			sampler* sampler) const override final;

		virtual const vec3& getPos() const override final
		{
			return m_origin;
		}
		virtual const vec3& getDir() const override final
		{
			return m_dir;
		}

		void revertRayToPixelPos(
			const ray& ray,
			int& px, int& py) const override final
		{
			// Not supported...
			AT_ASSERT(false);
		}

	private:
		vec3 m_origin;

		vec3 m_dir;
		vec3 m_right;
		vec3 m_up;
	};
}