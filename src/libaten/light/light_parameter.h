#pragma once

#include "math/vec3.h"
#include "math/vec4.h"

namespace aten {
    /**
     * @brief Data package for the result to sample light.
     */
    struct LightSampleResult {
        vec3 pos;               ///< light position.
        vec3 dir;               ///< light direction from the position.
        vec3 nml;               ///< light object surface normal.
        vec3 le;                ///< light color.
        vec3 finalColor;        ///< le * intensity
        real pdf{ real(0) };    ///< light sampling pdf.
    };

    /**
     * @brief Description for attributes of light.
     */
    struct LightAttribute {
        uint32_t isSingular : 1;    ///< Singular light.
        uint32_t isInfinite : 1;    ///< Inifinite light.
        uint32_t isIBL : 1;         ///< Image Based Light.
    };

    AT_DEVICE_MTRL_API constexpr auto LightAttributeArea = aten::LightAttribute{ false, false, false };
    AT_DEVICE_MTRL_API constexpr auto LightAttributeSingluar = aten::LightAttribute{ true,  false, false };
    AT_DEVICE_MTRL_API constexpr auto LightAttributeDirectional = aten::LightAttribute{ true,  true,  false };
    AT_DEVICE_MTRL_API constexpr auto LightAttributeIBL = aten::LightAttribute{ false, true,  true };

    /**
     * @brief Light type.
     */
    enum class LightType : int32_t {
        Area,       ///< Area light.
        IBL,        ///< Image Based Light.
        Direction,  ///< Direction light.
        Point,      ///< Point light.
        Spot,       ///< Spot light.

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
                union {
                    struct {
                        real innerAngle;    ///< Spot light inner angle.
                        real outerAngle;    ///< Spot light outer angle.
                        real falloff;       ///< Spot light fall off factor.
                    };
                    struct {
                        int32_t objid;
                        int32_t envmapidx;
                        int32_t padding;
                    };
                };

                LightAttribute attrib;  ///< Light attribute.
            };
        };

        AT_DEVICE_API LightParameter()
            : v0(0), v1(0)
        {};

        AT_DEVICE_API LightParameter(LightType _type, const LightAttribute& _attrib)
            : attrib(_attrib), type(_type)
        {
            constAttn = real(1);
            linearAttn = real(0);
            expAttn = real(0);

            if (type == LightType::Area || type == LightType::IBL) {
                objid = -1;
                envmapidx = -1;
                padding = 0;
            }
            else {
                innerAngle = AT_MATH_PI;
                outerAngle = AT_MATH_PI;
                falloff = real(0);
            }
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

            return *this;
        }

        AT_DEVICE_API bool IsValidLightObjectId() const
        {
            if (type == LightType::Area) {
                return objid >= 0;
            }
            return false;
        }

        AT_DEVICE_API int32_t GetLightEnvmapId() const
        {
            if (type == LightType::IBL) {
                return envmapidx;
            }
            return -1;
        }
    };
    //AT_STATICASSERT((sizeof(LightParameter) % 64) == 0);

    static constexpr size_t LightParameter_float4_size = sizeof(LightParameter) / sizeof(aten::vec4);
}
