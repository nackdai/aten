#pragma once

#include "scene/hitable.h"
#include "math/vec3.h"
#include "sampler/sampler.h"

namespace aten {
	struct LightSampleResult {
		vec3 dir;
		vec3 nml;
		vec3 le;
		real pdf{ real(0) };
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

		virtual real getPdf(const vec3& org, sampler* sampler) const = 0;

		virtual vec3 sampleDirToLight(const vec3& org, sampler* sampler) const = 0;

		virtual vec3 sampleNormalOnLight(const vec3& org, sampler* sampler) const = 0;

		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const = 0;

		virtual bool isSingular() const
		{
			return true;
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
			return false;
		}

	protected:
		vec3 m_pos;
		vec3 m_dir;
		vec3 m_le;
	};
}