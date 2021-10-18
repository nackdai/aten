#pragma once

#include "light/light.h"

namespace aten {
    class Values;
}

namespace AT_NAME {
    class SpotLight : public Light {
    public:
        SpotLight()
            : Light(aten::LightType::Spot, LightAttributeSingluar)
        {}
        SpotLight(
            const aten::vec3& pos,  // light position.
            const aten::vec3& dir,  // light direction from the position.
            const aten::vec3& le,   // light color.
            real constAttn,
            real linearAttn,
            real expAttn,
            real innerAngle,    // Umbra angle of spotlight in radians.
            real outerAngle,    // Penumbra angle of spotlight in radians.
            real falloff)       // Falloff factor.
            : Light(aten::LightType::Spot, LightAttributeSingluar)
        {
            m_param.pos = pos;
            m_param.dir = normalize(dir);
            m_param.le = le;

            setAttenuation(constAttn, linearAttn, expAttn);
            setSpotlightFactor(innerAngle, outerAngle, falloff);
        }

        SpotLight(aten::Values& val);

        virtual ~SpotLight() {}

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

        void setSpotlightFactor(
            real innerAngle,    // Umbra angle of spotlight in radians.
            real outerAngle,    // Penumbra angle of spotlight in radians.
            real falloff)        // Falloff factor.
        {
            m_param.innerAngle = aten::clamp<real>(innerAngle, 0, AT_MATH_PI - AT_MATH_EPSILON);
            m_param.outerAngle = aten::clamp<real>(outerAngle, innerAngle, AT_MATH_PI - AT_MATH_EPSILON);
            m_param.falloff = falloff;
        }

        virtual aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            aten::sampler* sampler) const override final
        {
            aten::LightSampleResult result;
            sample(&m_param, org, sampler, &result);
            return result;
        }

        static AT_DEVICE_API void sample(
            const aten::LightParameter* param,
            const aten::vec3& org,
            aten::sampler* sampler,
            aten::LightSampleResult* result)
        {
            result->pos = param->pos;
            result->dir = normalize(((aten::vec3)param->pos) - org);
            result->nml = param->dir;   // already normalized

            // NOTE
            // https://msdn.microsoft.com/ja-jp/library/bb172279(v=vs.85).aspx

            auto rho = dot(-((aten::vec3)param->dir), result->dir);

            auto cosHalfInnter = aten::cos(param->innerAngle * real(0.5));
            auto cosHalfOuter = aten::cos(param->outerAngle * real(0.5));

            if (rho > cosHalfOuter) {
                real spot = 0;

                // 半影内.
                if (rho > cosHalfInnter) {
                    // 本影内に入っている -> 最大限ライトの影響を受ける.
                    spot = 1;
                }
                else {
                    // 本影の外、半影の中.
                    // spot = real(1) - (cosHalfInnter - rho) / (cosHalfInnter - cosHalfOuter);
                    // 計算したい式は上の式だが、これを展開すると下の式になる.
                    spot = (rho - cosHalfOuter) / (cosHalfInnter - cosHalfOuter);
                }

                result->pdf = real(1);

#if 0
                // 減衰率.
                // http://ogldev.atspace.co.uk/www/tutorial20/tutorial20.html
                // 上記によると、L = Le / dist2 で正しいが、3Dグラフィックスでは見た目的にあまりよろしくないので、減衰率を使って計算する.
                auto dist2 = aten::squared_length(result->dir);
                auto dist = aten::sqrt(dist2);
                real attn = param->constAttn + param->linearAttn * dist + param->expAttn * dist2;

                // TODO
                // Is it correct?
                attn = aten::cmpMax(attn, real(1));

                result->le = param->le;
                result->finalColor = param->le * spot / attn;
#else
                result->le = param->le;
                result->finalColor = param->le * spot;

#endif
            }
            else {
                result->pdf = real(0);
                aten::set(result->le, 0, 0, 0);
                aten::set(result->finalColor, 0, 0, 0);
            }
        }
    };
}
