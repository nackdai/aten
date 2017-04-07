#pragma once

#include "types.h"
#include "math/ray.h"
#include "sampler/sampler.h"

namespace aten {
	struct CameraSampleResult {
		ray r;
		vec3 posOnImageSensor;
		vec3 posOnLens;
		vec3 nmlOnLens;
		vec3 posOnObjectplane;
		real pdfOnImageSensor{ real(1) };
		real pdfOnLens{ real(1) };
	};

	class camera {
	public:
		camera() {}
		virtual ~camera() {}

		virtual CameraSampleResult sample(
			real s, real t,
			sampler* sampler) const = 0;

		virtual real getPdfImageSensorArea(
			const vec3& hitPoint,
			const vec3& hitpointNml,
			const vec3& posOnImageSensor,
			const vec3& posOnLens,
			const vec3& posOnObjectPlane) const
		{
			return real(1);
		}

		virtual real getSensitivity(
			const vec3& posOnImagesensor,
			const vec3& posOnLens) const
		{
			return real(1);
		}

		virtual real getWdash(
			const vec3& hitPoint,
			const vec3& hitpointNml,
			const vec3& posOnImageSensor,
			const vec3& posOnLens,
			const vec3& posOnObjectPlane) const
		{
			return real(1);
		}

		virtual bool needRevert() const
		{
			return false;
		}

		virtual const vec3& getPos() const = 0;
		virtual const vec3& getDir() const = 0;

		virtual void revertRayToPixelPos(
			const ray& ray,
			int& px, int& py) const = 0;
	};
}