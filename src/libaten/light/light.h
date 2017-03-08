#pragma once

#include "scene/hitable.h"
#include "math/vec3.h"
#include "sampler/sampler.h"

namespace aten {
	struct LightSampleResult {
		vec3 pos;					// light position.
		vec3 dir;					// light direction from the position.
		vec3 nml;					// light object surface normal.
		vec3 le;					// light color.
		vec3 finalColor;			// le * intensity
		real intensity{ real(1) };	// light intensity(include attenuation).
		real pdf{ real(0) };		// light sampling pdf.

		hitable* obj{ nullptr };	// light object(only for area light)
	};

	class Light : public hitable {
	protected:
		Light() {}
		virtual ~Light() {}

	public:
		void setPos(const vec3& pos)
		{
			m_pos = pos;
		}

		void setDir(const vec3& dir)
		{
			m_dir = normalize(dir);
		}

		void setLe(const vec3& le)
		{
			m_le = le;
		}

		const vec3& getPos() const
		{
			return m_pos;
		}

		const vec3& getDir() const
		{
			return m_dir;
		}

		const vec3& getLe() const
		{
			return m_le;
		}

		virtual real samplePdf(const ray& r) const
		{
			// Not used...
			AT_ASSERT(false);
			return real(0);
		}

		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const = 0;

		virtual bool isSingular() const
		{
			return true;
		}

		virtual bool isInifinite() const
		{
			return false;
		}

		virtual const hitable* getLightObject() const
		{
			return nullptr;
		}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec) const override
		{
			// Usually, light is not hit.
			AT_ASSERT(false);
			return false;
		}

		virtual aabb getBoundingbox() const
		{
			// Most light don't have bounding box...
			AT_ASSERT(false);
			return std::move(aabb());
		}

	protected:
		vec3 m_pos;
		vec3 m_dir;
		vec3 m_le;
	};
}