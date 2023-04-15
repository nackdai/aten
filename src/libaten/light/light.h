#pragma once

#include "math/vec3.h"
#include "math/vec4.h"
#include "sampler/sampler.h"
#include "scene/hitable.h"

namespace aten {
    class Values;
}

namespace aten {
    struct LightSampleResult {
        vec3 pos;                   // light position.
        vec3 dir;                   // light direction from the position.
        vec3 nml;                   // light object surface normal.
        vec3 le;                    // light color.
        vec3 finalColor;            // le * intensity
        real pdf{ real(0) };        // light sampling pdf.
    };

    struct LightAttribute {
        uint32_t isSingular : 1;
        uint32_t isInfinite : 1;
        uint32_t isIBL : 1;
    };

    #define LightAttributeArea          aten::LightAttribute{ false, false, false }
    #define LightAttributeSingluar      aten::LightAttribute{ true,  false, false }
    #define LightAttributeDirectional   aten::LightAttribute{ true,  true,  false }
    #define LightAttributeIBL           aten::LightAttribute{ false, true,  true }

    enum class LightType : int32_t {
        Area,
        IBL,
        Direction,
        Point,
        Spot,

        LightTypeMax,
    };

    struct LightParameter {
        vec4 pos;
        vec4 dir;
        vec4 le;

        union {
            aten::vec4 v0;

            struct {
                LightType type;

                // For pointlight, spotlight.
                real constAttn;
                real linearAttn;
                real expAttn;
            };
        };

        union {
            aten::vec4 v1;

            struct {
                // For spotlight.
                real innerAngle;
                real outerAngle;
                real falloff;

                LightAttribute attrib;
            };
        };

        union {
            aten::vec4 v2;

            struct{
                int32_t objid;
                int32_t idx;
            };
        };

        AT_DEVICE_API LightParameter()
            : v0(0), v1(0), v2(0)
        {};

        AT_DEVICE_API LightParameter(LightType _type, const LightAttribute& _attrib)
            : attrib(_attrib), type(_type)
        {
            constAttn = real(1);
            linearAttn = real(0);
            expAttn = real(0);

            innerAngle = AT_MATH_PI;
            outerAngle = AT_MATH_PI;
            falloff = real(0);

            objid = -1;
            idx = -1;
        }

        AT_DEVICE_API LightParameter(const LightParameter& rhs)
        {
            pos = rhs.pos;
            dir = rhs.dir;
            le = rhs.le;

            v0 = rhs.v0;
            v1 = rhs.v1;

            objid = rhs.objid;
            idx = rhs.idx;
        }
    };
    //AT_STATICASSERT((sizeof(LightParameter) % 64) == 0);

    static constexpr size_t LightParameter_float4_size = sizeof(LightParameter) / sizeof(aten::vec4);
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

        virtual void getSamplePosNormalArea(
            const aten::context& ctxt,
            aten::SamplePosNormalPdfResult* result,
            aten::sampler* sampler) const
        {
            // TODO
            // Only for AreaLight...
            AT_ASSERT(false);
        }

    protected:
        aten::LightParameter m_param;
    };
}
