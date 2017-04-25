#pragma once

#include "camera/camera.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "math/math.h"

namespace AT_NAME {
	class PinholeCamera : public camera {
	public:
		PinholeCamera() {}
		virtual ~PinholeCamera() {}

		void init(
			const aten::vec3& origin,
			const aten::vec3& lookat,
			const aten::vec3& up,
			real vfov,	// vertical fov.
			uint32_t width, uint32_t height);

		virtual CameraSampleResult sample(
			real s, real t,
			aten::sampler* sampler) const override final;

		static AT_DEVICE_API CameraSampleResult sample(
			const aten::CameraParameter& param,
			real s, real t,
			aten::sampler* sampler);

		virtual const aten::vec3& getPos() const override final
		{
			return m_param.origin;
		}
		virtual const aten::vec3& getDir() const override final
		{
			return m_param.dir;
		}

		const aten::CameraParameter& param() const
		{
			return m_param;
		}

		void revertRayToPixelPos(
			const aten::ray& ray,
			int& px, int& py) const override final;

		virtual real convertImageSensorPdfToScenePdf(
			real pdfImage,	// Not used.
			const aten::vec3& hitPoint,
			const aten::vec3& hitpointNml,
			const aten::vec3& posOnImageSensor,
			const aten::vec3& posOnLens,
			const aten::vec3& posOnObjectPlane) const override final;

		virtual real getWdash(
			const aten::vec3& hitPoint,
			const aten::vec3& hitpointNml,
			const aten::vec3& posOnImageSensor,
			const aten::vec3& posOnLens,
			const aten::vec3& posOnObjectPlane) const override final;

		virtual real hitOnLens(
			const aten::ray& r,
			aten::vec3& posOnLens,
			aten::vec3& posOnObjectPlane,
			aten::vec3& posOnImageSensor,
			int& x, int& y) const override final;

	private:
		aten::CameraParameter m_param;
	};
}