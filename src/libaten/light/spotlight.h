#pragma once

#include "light/light.h"

namespace aten {
	class SpotLight : public Light {
	public:
		SpotLight() {}
		SpotLight(
			const vec3& pos,	// light position.
			const vec3& dir,	// light direction from the position.
			const vec3& le,		// light color.
			real constAttn,
			real linearAttn,
			real expAttn,
			real innerAngle,	// Umbra angle of spotlight in radians.
			real outerAngle,	// Penumbra angle of spotlight in radians.
			real falloff)		// Falloff factor.
		{
			m_pos = pos;
			m_dir = normalize(dir);
			m_le = le;

			setAttenuation(constAttn, linearAttn, expAttn);
			setSpotlightFactor(innerAngle, outerAngle, falloff);
		}

		virtual ~SpotLight() {}

	public:
		void setAttenuation(
			real constAttn,
			real linearAttn,
			real expAttn)
		{
			m_constAttn = std::max(constAttn, real(0));
			m_linearAttn = std::max(linearAttn, real(0));
			m_expAttn = std::max(expAttn, real(0));
		}

		void setSpotlightFactor(
			real innerAngle,	// Umbra angle of spotlight in radians.
			real outerAngle,	// Penumbra angle of spotlight in radians.
			real falloff)		// Falloff factor.
		{
			m_innerAngle = aten::clamp<real>(innerAngle, 0, AT_MATH_PI - AT_MATH_EPSILON);
			m_outerAngle = aten::clamp<real>(outerAngle, innerAngle, AT_MATH_PI - AT_MATH_EPSILON);
			m_falloff = falloff;
		}

		virtual real getPdf(const vec3& org, sampler* sampler) const override final
		{
			return real(1);
		}

		virtual vec3 sampleDirToLight(const vec3& org, sampler* sampler) const override final
		{
			vec3 dir = m_pos - org;
			return std::move(dir);
		}

		virtual vec3 sampleNormalOnLight(const vec3& org, sampler* sampler) const override final
		{
			// Do not use...
			return std::move(vec3());
		}

		virtual LightSampleResult sample(const vec3& org, sampler* sampler) const override final
		{
			LightSampleResult result;

			result.pos = m_pos;
			result.pdf = getPdf(org, sampler);
			result.dir = sampleDirToLight(org, sampler);
			result.nml = sampleNormalOnLight(org, sampler);

			// NOTE
			// https://msdn.microsoft.com/ja-jp/library/bb172279(v=vs.85).aspx

			auto theta = m_innerAngle;
			auto phi = m_outerAngle;

			auto lightdir = normalize(result.dir);

			auto rho = dot(-m_dir, lightdir);

			auto cosHalfTheta = aten::cos(m_innerAngle * 0.5);
			auto cosHalfPhi = aten::cos(m_outerAngle * 0.5);

			real spot = 0;

			if (rho > cosHalfTheta) {
				// 本影内に入っている -> 最大限ライトの影響を受ける.
				spot = 1;
			}
			else if (rho <= cosHalfPhi) {
				// 半影外に出ている -> ライトの影響を全く受けない.
				spot = 0;
			}
			else {
				// 本影の外、半影の中.
				spot = (rho - cosHalfPhi) / (cosHalfTheta - cosHalfPhi);
				spot = aten::pow(spot, m_falloff);
			}

			// 減衰率.
			// http://ogldev.atspace.co.uk/www/tutorial20/tutorial20.html
			// 上記によると、L = Le / dist2 で正しいが、3Dグラフィックスでは見た目的にあまりよろしくないので、減衰率を使って計算する.
			auto dist2 = result.dir.squared_length();
			auto dist = aten::sqrt(dist2);
			real attn = m_constAttn + m_linearAttn * dist + m_expAttn * dist2;

			// TODO
			// Is it correct?
			attn = std::max(attn, real(1));
			
			result.le = m_le * spot / attn;

			return std::move(result);
		}

	private:
		// NOTE
		// http://ogldev.atspace.co.uk/www/tutorial20/tutorial20.html
		real m_constAttn{ 1 };
		real m_linearAttn{ 0 };
		real m_expAttn{ 0 };

		// NOTE
		// https://msdn.microsoft.com/ja-jp/library/bb172279(v=vs.85).aspx
		real m_innerAngle;		// スポットライトの本影角度 (ラジアン単位) 範囲 :[0, pi).
		real m_outerAngle;		// スポットライトの半影角度 (ラジアン単位) 範囲 :[innerAngle, pi)
		real m_falloff{ 0 };	// フォールオフ係数.
	};
}