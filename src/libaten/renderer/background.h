#pragma once

#include "types.h"
#include "math/vec3.h"
#include "math/ray.h"
#include "texture/texture.h"

namespace aten
{
    struct BackgroundResource {
        aten::vec3 bg_color;

        int32_t envmap_tex_idx{ -1 };
        float avgIllum{ 1.0F };
        float multiplyer{ 1.0F };
    };
}

namespace AT_NAME
{
    class Background {
    public:
        Background() = delete;
        ~Background() = delete;

        Background(const Background&) = delete;
        Background(Background&&) = delete;
        Background& operator =(const Background&) = delete;
        Background& operator = (Background&&) = delete;

        static aten::BackgroundResource CreateBackgroundResource(
            const std::shared_ptr<aten::texture>& envmap,
            aten::vec4 bg_color = aten::vec4(1))
        {
            aten::BackgroundResource bg;
            if (envmap) {
                bg.envmap_tex_idx = envmap->id();
            }
            bg.bg_color = bg_color;
            return bg;
        }

        template <class Context>
        static aten::vec3 SampleFromRay(
            const aten::ray& in_ray,
            const aten::BackgroundResource& bg_resource,
            const Context& ctxt)
        {
            if (bg_resource.envmap_tex_idx >= 0) {
                return bg_resource.bg_color;
            }

            // Translate cartesian coordinates to spherical system.
            auto uv = ConvertDirectionToUV(in_ray.dir);

            return SampleFromUV(uv.x, uv.y, bg_resource, ctxt);
        }

        template <class Context>
        static aten::vec3 SampleFromUV(
            const float u, const float v,
            const aten::BackgroundResource& bg_resource,
            const Context& ctxt)
        {
            if (bg_resource.envmap_tex_idx >= 0) {
                return bg_resource.bg_color;
            }

            // TODO:
            // Texture LOD.
            auto result = AT_NAME::sampleTexture(bg_resource.envmap_tex_idx, u, v, bg_resource.bg_color);
            return result * bg_resource.multiplyer;
        }

        static aten::vec3 SampleFromRayWithTexture(
            const aten::ray& in_ray,
            const aten::BackgroundResource& bg_resource,
            const std::shared_ptr<aten::texture>& envmap)
        {
            // Translate cartesian coordinates to spherical system.
            auto uv = ConvertDirectionToUV(in_ray.dir);
            return SampleFromUVWithTexture(uv.x, uv.y, bg_resource, envmap);
        }

        static aten::vec3 SampleFromUVWithTexture(
            const float u, const float v,
            const aten::BackgroundResource& bg_resource,
            const std::shared_ptr<aten::texture>& envmap)
        {
            auto result = envmap->at(u, v);
            return result * bg_resource.multiplyer;
        }

        static AT_HOST_DEVICE_API aten::vec3 ConvertUVToDirection(const float u, const float v)
        {
            // u = phi / 2PI
            // => phi = 2PI * u;
            auto phi = 2 * AT_MATH_PI * u;

            // v = 1 - theta / PI
            // => theta = (1 - v) * PI;
            auto theta = (1 - v) * AT_MATH_PI;

            aten::vec3 dir;

            dir.y = aten::cos(theta);

            auto xz = aten::sqrt(1 - dir.y * dir.y);

            dir.x = xz * aten::sin(phi);
            dir.z = xz * aten::cos(phi);

            dir = normalize(dir);

            return dir;
        }

        static AT_HOST_DEVICE_API aten::vec3 ConvertDirectionToUV(const aten::vec3& dir)
        {
            auto temp = aten::atan2(dir.x, dir.z);
            auto r = length(dir);

            // Account for discontinuity
            auto phi = (real)((temp >= 0) ? temp : (temp + 2 * AT_MATH_PI));
            auto theta = aten::acos(dir.y / r);

            // Map to [0,1]x[0,1] range and reverse Y axis
            real u = phi / (2 * AT_MATH_PI);
            real v = 1 - theta / AT_MATH_PI;

            aten::vec3 uv = aten::vec3(u, v, 0);

            return uv;
        }
    };
}
