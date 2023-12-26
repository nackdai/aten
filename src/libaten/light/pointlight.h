#pragma once

#include "light/light.h"

namespace aten {
    class Values;
}

namespace AT_NAME {
    class PointLight : public Light {
    public:
        PointLight()
            : Light(aten::LightType::Point, aten::LightAttributeSingluar)
        {}
        PointLight(
            const aten::vec3& pos,
            const aten::vec3& le,
            real constAttn = 1,
            real linearAttn = 0,
            real expAttn = 0)
            : Light(aten::LightType::Point, aten::LightAttributeSingluar)
        {
            m_param.pos = pos;
            m_param.le = le;

            setAttenuation(constAttn, linearAttn, expAttn);
        }

        PointLight(aten::Values& val);

        virtual ~PointLight() = default;

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

        static AT_DEVICE_API void sample(
            const aten::LightParameter* param,
            const aten::vec3& org,
            aten::sampler* sampler,
            aten::LightSampleResult* result)
        {
            result->pdf = real(1);

            result->dir = ((aten::vec3)param->pos) - org;

            auto dist2 = aten::squared_length(result->dir);
            auto dist = aten::sqrt(dist2);

            result->pos = param->pos;
            result->nml = normalize(-result->dir);

            // ������.
            // http://ogldev.atspace.co.uk/www/tutorial20/tutorial20.html
            // ��L�ɂ��ƁAL = Le / dist2 �Ő��������A3D�O���t�B�b�N�X�ł͌����ړI�ɂ��܂��낵���Ȃ��̂ŁA���������g���Čv�Z����.
            real attn = param->constAttn + param->linearAttn * dist + param->expAttn * dist2;

            // TODO
            // Is it correct?
            attn = aten::cmpMax(attn, real(1));

            result->le = param->le;
            result->finalColor = param->le / attn;
        }
    };
}
