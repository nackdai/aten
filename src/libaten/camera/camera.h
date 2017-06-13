#pragma once

#include "types.h"
#include "math/ray.h"
#include "sampler/sampler.h"

namespace aten {
	// TODO
	// Only for Pinhole Cmaera...
	struct CameraParameter {
		vec3 origin;

		real aspect;
		vec3 center;

		vec3 u;
		vec3 v;

		vec3 dir;
		vec3 right;
		vec3 up;

		real dist;
		int width;
		int height;
	};
};

namespace AT_NAME {
	struct CameraSampleResult {
		aten::ray r;
		aten::vec3 posOnImageSensor;
		aten::vec3 posOnLens;
		aten::vec3 nmlOnLens;
		aten::vec3 posOnObjectplane;
		real pdfOnImageSensor{ real(1) };
		real pdfOnLens{ real(1) };
	};

	class camera {
	public:
		camera() {}
		virtual ~camera() {}

		virtual void update() = 0;

		virtual CameraSampleResult sample(
			real s, real t,
			aten::sampler* sampler) const = 0;

		virtual real convertImageSensorPdfToScenePdf(
			real pdfImage,
			const aten::vec3& hitPoint,
			const aten::vec3& hitpointNml,
			const aten::vec3& posOnImageSensor,
			const aten::vec3& posOnLens,
			const aten::vec3& posOnObjectPlane) const
		{
			return real(1);
		}

		virtual real getSensitivity(
			const aten::vec3& posOnImagesensor,
			const aten::vec3& posOnLens) const
		{
			return real(1);
		}

		virtual real getWdash(
			const aten::vec3& hitPoint,
			const aten::vec3& hitpointNml,
			const aten::vec3& posOnImageSensor,
			const aten::vec3& posOnLens,
			const aten::vec3& posOnObjectPlane) const
		{
			return real(1);
		}

		virtual real hitOnLens(
			const aten::ray& r,
			aten::vec3& posOnLens,
			aten::vec3& posOnObjectPlane,
			aten::vec3& posOnImageSensor,
			int& x, int& y) const
		{
			return -AT_MATH_INF;
		}

		virtual bool needRevert() const
		{
			return false;
		}

		virtual bool isPinhole() const
		{
			return true;
		}

		virtual const aten::vec3& getPos() const = 0;
		virtual const aten::vec3& getDir() const = 0;

		virtual aten::vec3& getPos() = 0;
		virtual aten::vec3& getAt() = 0;

		virtual void revertRayToPixelPos(
			const aten::ray& ray,
			int& px, int& py) const = 0;

		virtual real getImageSensorWidth() const
		{
			return real(1);
		}

		virtual real getImageSensorHeight() const
		{
			return real(1);
		}
	};
}