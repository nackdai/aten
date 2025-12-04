#pragma once

#include "math/vec3.h"
#include "math/vec4.h"

namespace aten {
    /**
     * @brief Description for attributes of light.
     */
    struct LightAttribute {
        uint32_t is_singular : 1;    ///< Singular light.
        uint32_t isInfinite : 1;    ///< Inifinite light.
        uint32_t isIBL : 1;         ///< Image Based Light.
    };

    /**
     * @brief Data package for the result to sample light.
     */
    struct LightSampleResult {
        vec3 pos;                       ///< Light position.
        vec3 dir;                       ///< Light direction to light position.
        float dist_to_light{ 0.0F };    ///< Distnace to light position.
        vec3 nml;                       ///< Light object surface normal.
        vec3 light_color{ 0.0F };       ///< Light color.
        float pdf{ 0.0F };              ///< Light sampling pdf.
        LightAttribute attrib;          ///< Light attribute.
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

        LightType type;             ///< Light type.
        vec3 light_color{ 0.0F };   ///< Light color as RGB.

        float innerAngle{ AT_MATH_PI }; ///< Spot light inner angle.
        float outerAngle{ AT_MATH_PI }; ///< Spot light outer angle.
        LightAttribute attrib;          ///< Light attribute.
        float scale{ 1.0F };            ///< Scele factor to be multiplied to intensity or luminance.

        float intensity{ 1.0F };
        int32_t arealight_objid{ -1 };  ///< Object index to be referred as area light.
        int32_t envmapidx{ -1 };        ///< Texture index as environment map.
        int32_t padding;

        AT_HOST_DEVICE_API LightParameter(LightType _type, const LightAttribute& _attrib)
        {
            attrib = _attrib;
            type = _type;
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
