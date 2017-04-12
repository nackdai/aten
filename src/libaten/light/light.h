#pragma once

#include "scene/hitable.h"
#include "math/vec3.h"
#include "sampler/sampler.h"
#include "misc/value.h"

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

	struct LightParameter {
		float pos[3];
		float dir[3];
		float le[3];

		// For pointlight, spotlight.
		float constAttn{ 1 };
		float linearAttn{ 0 };
		float expAttn{ 0 };

		// For spotlight.
		float innerAngle;
		float outerAngle;
		float falloff{ 0 };

		int object{ -1 };
		int envmap{ -1 };
	};

	class Light : public hitable {
	protected:
		Light() {}

		Light(Values& val)
		{
			m_pos = val.get("pos", m_pos);
			m_dir = val.get("dir", m_pos);
			m_le = val.get("le", m_pos);
		}

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

		virtual bool isIBL() const
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

		virtual void serialize(LightParameter& param) const = 0;

	protected:
		static void serialize(const Light* light, LightParameter& param);

	protected:
		vec3 m_pos;
		vec3 m_dir;
		vec3 m_le;
	};
}