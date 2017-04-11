#pragma once

#include "camera/camera.h"
#include "math/vec3.h"
#include "sampler/sampler.h"

namespace aten {
	class ThinLensCamera : public camera {
	public:
		ThinLensCamera() {}
		virtual ~ThinLensCamera() {}

	public:
		void init(
			int width, int height,
			vec3 lookfrom, vec3 lookat, vec3 vup,
			real imageSensorSize,
			real imageSensorToLensDistance,
			real lensToObjectplaneDistance,
			real lensRadius,
			real W_scale);

		virtual CameraSampleResult sample(
			real s, real t,
			sampler* sampler) const override final;

		virtual real hitOnLens(
			const ray& r,
			vec3& posOnLens,
			vec3& posOnObjectPlane,
			vec3& posOnImageSensor,
			int& x, int& y) const override final;

		virtual real getPdfImageSensorArea(
			const vec3& hitPoint,
			const vec3& hitpointNml,
			const vec3& posOnImageSensor,
			const vec3& posOnLens,
			const vec3& posOnObjectPlane) const override final;

		virtual real getSensitivity(
			const vec3& posOnImagesensor,
			const vec3& posOnLens) const override final;

		virtual real getWdash(
			const vec3& hitPoint,
			const vec3& hitpointNml,
			const vec3& posOnImageSensor,
			const vec3& posOnLens,
			const vec3& posOnObjectPlane) const override final;

		virtual bool needRevert() const
		{
			return true;
		}

		virtual bool isPinhole() const
		{
			return false;
		}

		virtual const vec3& getPos() const override final
		{
			return m_imagesensor.center;
		}
		virtual const vec3& getDir() const override final
		{
			return m_imagesensor.dir;
		}

		void revertRayToPixelPos(
			const ray& ray,
			int& px, int& py) const override final;

	private:
		// 解像度.
		int m_imageWidthPx;
		int m_imageHeightPx;

		// イメージセンサ.
		struct ImageSensor {
			vec3 center;
			vec3 dir;
			vec3 up;
			vec3 u;
			vec3 v;
			vec3 lower_left;
			real width;
			real height;
		} m_imagesensor;

		// 物理的なピクセルサイズ.
		real m_pixelWidth;
		real m_pixelHeight;

		// レンズ.
		struct Lens {
			vec3 center;
			vec3 u;
			vec3 v;
			vec3 normal;
			real radius;
		} m_lens;

		struct ObjectPlane {
			vec3 center;
			vec3 u;
			vec3 v;
			vec3 normal;
			vec3 lower_left;
		} m_objectplane;

		real m_imageSensorToLensDistance;
		real m_lensToObjectplaneDistance;

		// イメージセンサの感度
		real m_W;
	};
}
