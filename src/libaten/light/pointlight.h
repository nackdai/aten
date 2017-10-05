#pragma once

#include "light/light.h"

namespace AT_NAME {
	class PointLight : public Light {
	public:
		PointLight()
			: Light(aten::LightType::Point, LightAttributeSingluar)
		{}
		PointLight(
			const aten::vec3& pos,
			const aten::vec3& le,
			real constAttn = 1,
			real linearAttn = 0,
			real expAttn = 0)
			: Light(aten::LightType::Point, LightAttributeSingluar)
		{
			m_param.pos = pos;
			m_param.le = le;

			setAttenuation(constAttn, linearAttn, expAttn);
		}

		PointLight(aten::Values& val)
			: Light(aten::LightType::Point, LightAttributeSingluar, val)
		{
			m_param.constAttn = val.get("constAttn", m_param.constAttn);
			m_param.linearAttn = val.get("linearAttn", m_param.linearAttn);
			m_param.expAttn = val.get("expAttn", m_param.expAttn);

			setAttenuation(m_param.constAttn, m_param.linearAttn, m_param.expAttn);
		}

		virtual ~PointLight() {}

	public:
		void setAttenuation(
			real constAttn,
			real linearAttn,
			real expAttn)
		{
			m_param.constAttn = std::max(constAttn, real(0));
			m_param.linearAttn = std::max(linearAttn, real(0));
			m_param.expAttn = std::max(expAttn, real(0));
		}

		virtual aten::LightSampleResult sample(const aten::vec3& org, aten::sampler* sampler) const override final
		{
			aten::LightSampleResult result;
			sample(&result, &m_param, org, sampler);
			return std::move(result);
		}

		static AT_DEVICE_API void sample(
			aten::LightSampleResult* result,
			const aten::LightParameter* param,
			const aten::vec3& org,
			aten::sampler* sampler)
		{-
			result->pos = param->pos;
			result->pdf = real(1);
			result->dir = ((aten::vec3)param->pos) - org;
			result->nml = aten::vec3();	// Not used...

			auto dist2 = aten::squared_length(result->dir);
			auto dist = aten::sqrt(dist2);

			// 減衰率.
			// http://ogldev.atspace.co.uk/www/tutorial20/tutorial20.html
			// 上記によると、L = Le / dist2 で正しいが、3Dグラフィックスでは見た目的にあまりよろしくないので、減衰率を使って計算する.
			real attn = param->constAttn + param->linearAttn * dist + param->expAttn * dist2;

			// TODO
			// Is it correct?
			attn = aten::cmpMax(attn, real(1));
			
			result->le = param->le;
			result->intensity = 1 / attn;
			result->finalColor = param->le / attn;
		}
	};
}