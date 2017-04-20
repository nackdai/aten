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

		void* obj{ nullptr };	// light object(only for area light)
	};

	struct LightAttribute {
		struct {
			const uint32_t isSingular : 1;
			const uint32_t isInfinite : 1;
			const uint32_t isIBL : 1;
		};

		AT_DEVICE_API LightAttribute(
			bool _isSingular = false,
			bool _isInfinite = false,
			bool _isIBL = false)
			: isSingular(_isSingular), isInfinite(_isInfinite), isIBL(_isIBL)
		{}
	};

	#define LightAttributeArea			LightAttribute(false, false, false)
	#define LightAttributeSingluar		LightAttribute(true,  false, false)
	#define LightAttributeDirectional	LightAttribute(true,  true,  false)
	#define LightAttributeIBL			LightAttribute(false, true,  true)

	enum LightType {
		Area,
		IBL,
		Direction,
		Point,
		Spot,
	};

	struct LightParameter {
		LightType type;

		vec3 pos;
		vec3 dir;
		vec3 le;

		// For pointlight, spotlight.
		real constAttn{ 1 };
		real linearAttn{ 0 };
		real expAttn{ 0 };

		// For spotlight.
		real innerAngle{ AT_MATH_PI };
		real outerAngle{ AT_MATH_PI };
		real falloff{ 0 };

		UnionIdxPtr object;
		UnionIdxPtr envmap;

		LightAttribute attrib;

		AT_DEVICE_API LightParameter(LightType _type, const LightAttribute& _attrib)
			: attrib(_attrib), type(_type)
		{}
	};

	class Light {
		static std::vector<Light*> g_lights;

	protected:
		Light(LightType type, const LightAttribute& attrib);
		Light(LightType type, const LightAttribute& attrib, Values& val);

		virtual ~Light();

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

		bool isSingular() const
		{
			return m_param.attrib.isSingular;
		}

		bool isInfinite() const
		{
			return m_param.attrib.isInfinite;
		}

		bool isIBL() const
		{
			return m_param.attrib.isIBL;
		}

		virtual const hitable* getLightObject() const
		{
			return nullptr;
		}

		const LightParameter& param() const
		{
			return m_param;
		}

		virtual hitable::SamplingPosNormalPdf getSamplePosNormalPdf(sampler* sampler) const
		{
			// TODO
			// Only for AreaLight...
			AT_ASSERT(false);
			return std::move(hitable::SamplingPosNormalPdf(vec3(0), vec3(1, 0, 0), real(0)));
		}

		static uint32_t getLightNum();
		static const Light* getLight(uint32_t idx);
		static const std::vector<Light*>& getLights();

	protected:
		LightParameter m_param;
	};
}