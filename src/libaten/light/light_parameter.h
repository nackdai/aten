#pragma once

#include "math/vec3.h"
#include "math/vec4.h"

namespace aten {
    struct LightSampleResult {
        vec3 pos;               ///< light position.
        vec3 dir;               ///< light direction from the position.
        vec3 nml;               ///< light object surface normal.
        vec3 le;                ///< light color.
        vec3 finalColor;        ///< le * intensity
        real pdf{ real(0) };    ///< light sampling pdf.
    };

    struct LightAttribute {
        uint32_t isSingular : 1;
        uint32_t isInfinite : 1;
        uint32_t isIBL : 1;
    };

    AT_DEVICE_MTRL_API constexpr auto LightAttributeArea = aten::LightAttribute{ false, false, false };
    AT_DEVICE_MTRL_API constexpr auto LightAttributeSingluar = aten::LightAttribute{ true,  false, false };
    AT_DEVICE_MTRL_API constexpr auto LightAttributeDirectional = aten::LightAttribute{ true,  true,  false };
    AT_DEVICE_MTRL_API constexpr auto LightAttributeIBL = aten::LightAttribute{ false, true,  true };

    enum class LightType : int32_t {
        Area,
        IBL,
        Direction,
        Point,
        Spot,

        LightTypeMax,
    };

    struct LightParameter {
        vec4 pos;   ///< Light position.
        vec4 dir;   ///< Light direction.
        vec4 le;    ///< Light luminousness.

        union {
            aten::vec4 v0;

            struct {
                LightType type;     ///< Light type.

                real constAttn;     ///< Point/Spot light attenuation for constant.
                real linearAttn;    ///< Point/Spot light attenuation for linear.
                real expAttn;       ///< Point/Spot light attenuation for exponential.
            };
        };

        union {
            aten::vec4 v1;

            struct {
                real innerAngle;    ///< Spot light inner angle.
                real outerAngle;    ///< Spot light outer angle.
                real falloff;       ///< Spot light fall off factor.

                LightAttribute attrib;  ///< Light attribute.
            };
        };

        union {
            aten::vec4 v2;

            struct{
                int32_t objid;
                int32_t envmapidx;
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
            envmapidx = -1;
        }

        AT_DEVICE_API LightParameter(const LightParameter& rhs)
        {
            *this = rhs;
        }

        AT_DEVICE_API LightParameter& operator=(const LightParameter& rhs)
        {
            pos = rhs.pos;
            dir = rhs.dir;
            le = rhs.le;

            v0 = rhs.v0;
            v1 = rhs.v1;

            objid = rhs.objid;
            envmapidx = rhs.envmapidx;

            return *this;
        }
    };
    //AT_STATICASSERT((sizeof(LightParameter) % 64) == 0);

    static constexpr size_t LightParameter_float4_size = sizeof(LightParameter) / sizeof(aten::vec4);
}
