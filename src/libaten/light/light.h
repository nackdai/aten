#pragma once

#include "math/vec3.h"
#include "math/vec4.h"
#include "sampler/sampler.h"
#include "scene/hitable.h"
#include "light//light_parameter.h"

namespace aten {
    class Values;
}

namespace AT_NAME
{
    class Light {
    protected:
        Light(aten::LightType type, const aten::LightAttribute& attrib)
            : m_param(type, attrib)
        {}
        Light(aten::LightType type, const aten::LightAttribute& attrib, const aten::Values& val);

        Light() = delete;
        virtual ~Light() = default;

    public:
        template <class CONTEXT>
        static AT_DEVICE_API void sample(
            aten::LightSampleResult& result,
            const aten::LightParameter& param,
            const CONTEXT& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler,
            uint32_t lod = 0);

        void setPos(const aten::vec3& pos)
        {
            m_param.pos = pos;
        }

        void setDir(const aten::vec3& dir)
        {
            m_param.dir = normalize(dir);
        }

        void setLightColor(const aten::vec3& light_color)
        {
            m_param.light_color = light_color;
        }

        const aten::vec3& GetPos() const
        {
            return m_param.pos.v;
        }

        const aten::vec3& GetDir() const
        {
            return m_param.dir.v;
        }

        const aten::vec3& getLightColor() const
        {
            return m_param.light_color;
        }

        bool is_singular() const
        {
            return m_param.attrib.is_singular;
        }

        bool isInfinite() const
        {
            return m_param.attrib.isInfinite;
        }

        bool isIBL() const
        {
            return m_param.attrib.isIBL;
        }

        const aten::LightParameter& param() const
        {
            return m_param;
        }

        aten::LightParameter& param()
        {
            return m_param;
        }

    protected:
        aten::LightParameter m_param;
    };
}
