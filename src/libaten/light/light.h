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

        virtual ~Light() {}

    public:
        void setPos(const aten::vec3& pos)
        {
            m_param.pos = pos;
        }

        void setDir(const aten::vec3& dir)
        {
            m_param.dir = normalize(dir);
        }

        void setLe(const aten::vec3& le)
        {
            m_param.le = le;
        }

        const aten::vec3& getPos() const
        {
            return m_param.pos.v;
        }

        const aten::vec3& getDir() const
        {
            return m_param.dir.v;
        }

        const aten::vec3& getLe() const
        {
            return m_param.le.v;
        }

        virtual aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            aten::sampler* sampler) const = 0;

        virtual aten::LightSampleResult sample(
            const aten::context& ctxt,
            const aten::vec3& org,
            const aten::vec3& nml,
            aten::sampler* sampler) const
        {
            return sample(ctxt, org, sampler);
        }

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

        const aten::LightParameter& param() const
        {
            return m_param;
        }

        const aten::LightParameter& get_param() const
        {
            return m_param;
        }

    protected:
        aten::LightParameter m_param;
    };
}
