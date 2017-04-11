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
			real vfov,	// vertical fov.
			uint32_t width, uint32_t height);

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
			int& px, int& py) const override final;

		virtual real getPdfImageSensorArea(
			const vec3& hitPoint,
			const vec3& hitpointNml,
			const vec3& posOnImageSensor,
			const vec3& posOnLens,
			const vec3& posOnObjectPlane) const override final;

		virtual real getWdash(
			const vec3& hitPoint,
			const vec3& hitpointNml,
			const vec3& posOnImageSensor,
			const vec3& posOnLens,
			const vec3& posOnObjectPlane) const override final;

		virtual real hitOnLens(
			const ray& r,
			vec3& posOnLens,
			vec3& posOnObjectPlane,
			vec3& posOnImageSensor,
			int& x, int& y) const override final;

	private:
		vec3 m_origin;

		real m_aspect;
		vec3 m_center;

		vec3 m_u;
		vec3 m_v;

		vec3 m_dir;
		vec3 m_right;
		vec3 m_up;

		real m_dist;
		int m_width;
		int m_height;
	};
}