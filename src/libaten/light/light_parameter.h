#pragma once

#include "math/vec3.h"
#include "math/vec4.h"

namespace aten {
    /**
     * @brief Data package for the result to sample light.
     */
    struct LightSampleResult {
        vec3 pos;               ///< Light position.
        vec3 dir;               ///< Light direction from the position.
        vec3 nml;               ///< Light object surface normal.
        vec3 light_color;       ///< Light color.
        float pdf{ float(0) };    ///< Light sampling pdf.
    };

    /**
     * @brief Description for attributes of light.
     */
    struct LightAttribute {
        uint32_t isSingular : 1;    ///< Singular light.
        uint32_t isInfinite : 1;    ///< Inifinite light.
        uint32_t isIBL : 1;         ///< Image Based Light.
    };

    AT_DEVICE_API constexpr auto LightAttributeArea = aten::LightAttribute{ false, false, false };
    AT_DEVICE_API constexpr auto LightAttributeSingluar = aten::LightAttribute{ true,  false, false };
    AT_DEVICE_API constexpr auto LightAttributeDirectional = aten::LightAttribute{ true,  true,  false };
    AT_DEVICE_API constexpr auto LightAttributeIBL = aten::LightAttribute{ false, true,  true };

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

        union {
            aten::vec4 v0;

            struct {
                LightType type;     ///< Light type.
                vec3 light_color;   ///< Light color as RGB (0, 1).
            };
        };

        union {
            aten::vec4 v1;

            struct {
                float innerAngle;       ///< Spot light inner angle.
                float outerAngle;       ///< Spot light outer angle.
                LightAttribute attrib;  ///< Light attribute.
            };
        };

        union {
            aten::vec4 v2;

            struct {
                float scale;    ///< Scele factor to be multiplied to intensity or luminance.
                union {
                    float intensity;    ///< Light intetnsity for point/spot/area light. [W/sr]
                    float luminance;    ///< Luminance emittance for directional light. [W/(sr*m^2)]
                };
                int32_t arealight_objid;    ///< Object index to be referred as area light.
                int32_t envmapidx;          ///< Texture index as environment map.
            };
        };

        AT_HOST_DEVICE_API LightParameter()
            : v0(0), v1(0), v2(0)
        {
            innerAngle = AT_MATH_PI;
            outerAngle = AT_MATH_PI;
            arealight_objid = -1;
            envmapidx = -1;
            scale = 1.0f;
        };

        AT_HOST_DEVICE_API LightParameter(LightType _type, const LightAttribute& _attrib)
            : v0(0), v1(0), v2(0), attrib(_attrib), type(_type)
        {
            innerAngle = AT_MATH_PI;
            outerAngle = AT_MATH_PI;
            arealight_objid = -1;
            envmapidx = -1;
            scale = 1.0f;
        }

        LightParameter(const LightParameter& rhs)
        {
            *this = rhs;
        }

        LightParameter& operator=(const LightParameter& rhs)
        {
            pos = rhs.pos;
            dir = rhs.dir;

            v0 = rhs.v0;
            v1 = rhs.v1;
            v2 = rhs.v2;

            return *this;
        }

        AT_HOST_DEVICE_API bool IsValidLightObjectId() const
        {
            if (type == LightType::Area) {
                return arealight_objid >= 0;
            }
            return false;
        }

        AT_HOST_DEVICE_API int32_t GetLightEnvmapId() const
        {
            if (type == LightType::IBL) {
                return envmapidx;
            }
            return -1;
        }
    };
    AT_STATICASSERT((sizeof(LightParameter) % sizeof(aten::vec4)) == 0);
}
