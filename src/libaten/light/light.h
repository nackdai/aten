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
		union {
			vec3 pos;
			real posArray[3];
		};
		union {
			vec3 dir;
			real dirArray[3];
		};
		union {
			vec3 le;
			real leArray[3];
		};

		// For pointlight, spotlight.
		real constAttn{ 1 };
		real linearAttn{ 0 };
		real expAttn{ 0 };

		// For spotlight.
		real innerAngle;
		real outerAngle;
		real falloff{ 0 };

		union UnionIdxPtr {
			int idx;
			void* ptr{ nullptr };
		};

		UnionIdxPtr object;
		UnionIdxPtr envmap;

		LightParameter() {}
	};

	class Light : public hitable {
	protected:
		Light() {}

		Light(Values& val)
		{
			m_param.pos = val.get("pos", m_param.pos);
			m_param.dir = val.get("dir", m_param.dir);
			m_param.le = val.get("le", m_param.le);
		}

		virtual ~Light() {}

	public:
		void setPos(const vec3& pos)
		{
			m_param.pos = pos;
		}

		void setDir(const vec3& dir)
		{
			m_param.dir = normalize(dir);
		}

		void setLe(const vec3& le)
		{
			m_param.le = le;
		}

		const vec3& getPos() const
		{
			return m_param.pos;
		}

		const vec3& getDir() const
		{
			return m_param.dir;
		}

		const vec3& getLe() const
		{
			return m_param.le;
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

	protected:
		LightParameter m_param;
	};
}