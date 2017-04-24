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

	#define LightAttributeArea			aten::LightAttribute(false, false, false)
	#define LightAttributeSingluar		aten::LightAttribute(true,  false, false)
	#define LightAttributeDirectional	aten::LightAttribute(true,  true,  false)
	#define LightAttributeIBL			aten::LightAttribute(false, true,  true)

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
}

namespace AT_NAME
{
	class Light {
		static std::vector<Light*> g_lights;

	protected:
		Light(aten::LightType type, const aten::LightAttribute& attrib);
		Light(aten::LightType type, const aten::LightAttribute& attrib, aten::Values& val);

		virtual ~Light();

	public:
		void setPos(const aten::vec3& pos)
		{
			m_param.pos = pos;
		}

		void setDir(const aten::vec3& dir)
		{
			m_param.dir = aten::normalize(dir);
		}

		void setLe(const aten::vec3& le)
		{
			m_param.le = le;
		}

		const aten::vec3& getPos() const
		{
			return m_param.pos;
		}

		const aten::vec3& getDir() const
		{
			return m_param.dir;
		}

		const aten::vec3& getLe() const
		{
			return m_param.le;
		}

		virtual aten::LightSampleResult sample(const aten::vec3& org, aten::sampler* sampler) const = 0;

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

		virtual const aten::hitable* getLightObject() const
		{
			return nullptr;
		}

		const aten::LightParameter& param() const
		{
			return m_param;
		}

		virtual aten::hitable::SamplingPosNormalPdf getSamplePosNormalPdf(aten::sampler* sampler) const
		{
			// TODO
			// Only for AreaLight...
			AT_ASSERT(false);
			return std::move(aten::hitable::SamplingPosNormalPdf(aten::vec3(0), aten::vec3(1, 0, 0), real(0)));
		}

		static uint32_t getLightNum();
		static const Light* getLight(uint32_t idx);
		static const std::vector<Light*>& getLights();

	protected:
		aten::LightParameter m_param;
	};
}