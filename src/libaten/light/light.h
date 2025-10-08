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
            : param_(type, attrib)
        {}
        Light(aten::LightType type, const aten::LightAttribute& attrib, const aten::Values& val);

        virtual ~Light() {}

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
            param_.pos = pos;
        }

        void setDir(const aten::vec3& dir)
        {
            param_.dir = normalize(dir);
        }

        void setLightColor(const aten::vec3& light_color)
        {
            param_.light_color = light_color;
        }

        const aten::vec3& GetPos() const
        {
            return param_.pos.v;
        }

        const aten::vec3& GetDir() const
        {
            return param_.dir.v;
        }

        const aten::vec3& getLightColor() const
        {
            return param_.light_color;
        }

        bool is_singular() const
        {
            return param_.attrib.is_singular;
        }

        bool isInfinite() const
        {
            return param_.attrib.isInfinite;
        }

        bool isIBL() const
        {
            return param_.attrib.isIBL;
        }

        const aten::LightParameter& param() const
        {
            return param_;
        }

        aten::LightParameter& param()
        {
            return param_;
        }

    protected:
        aten::LightParameter param_;
    };
}
