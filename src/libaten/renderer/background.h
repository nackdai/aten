#pragma once

#include "types.h"
#include "material/sample_texture.h"
#include "math/vec3.h"
#include "math/ray.h"
#include "image/texture.h"
#include "material/sample_texture.h"

namespace aten
{
    struct BackgroundResource {
        aten::vec3 bg_color{ 0.0F };

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
        static AT_HOST_DEVICE_API aten::vec4 SampleFromRay(
            const aten::vec3& in_ray,
            const aten::BackgroundResource& bg_resource,
            const Context& ctxt)
        {
            if (bg_resource.envmap_tex_idx < 0) {
                return bg_resource.bg_color;
            }

            // Translate cartesian coordinates to spherical system.
            auto uv = ConvertDirectionToUV(in_ray);

            return SampleFromUV(uv.x, uv.y, bg_resource, ctxt);
        }

        template <class Context>
        static AT_HOST_DEVICE_API aten::vec4 SampleFromUV(
            const float u, const float v,
            const aten::BackgroundResource& bg_resource,
            const Context& ctxt)
        {
            if (bg_resource.envmap_tex_idx < 0) {
                return bg_resource.bg_color;
            }

            // TODO:
            // Texture LOD.
#ifdef __CUDACC__
            // envmapidx is index to array of textures in context.
            // In GPU, sampleTexture requires texture id of CUDA. So, arguments is different.
            aten::vec4 result{ tex2D<float4>(ctxt.textures[bg_resource.envmap_tex_idx], u, v) };
#else
            const auto result = AT_NAME::sampleTexture(ctxt, bg_resource.envmap_tex_idx, u, v, aten::vec4(1));
#endif
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
            // NOTE:
            //
            // phi +pi         0          -pi  theta  v
            //     +-----+-----+-----+-----+    0     1
            //     |     |     |     |     |    |     |
            //     +-----+-----+-----+-----+   pi/2  0.5
            //     |     |     |     |     |    |     |
            //     +-----+-----+-----+-----+    pi    0
            //  u 0.5  0.25   0|1   0.75  0.5

            //     (u=0) +z (u=1)
            //           |
            //           |
            // +x <------+------ -x
            // (u=0.25)  |       (u=0.75)
            //           |
            //           -z (u=0.5)

            auto temp = aten::atan2(dir.x, dir.z);
            auto r = length(dir);

            // Account for discontinuity
            auto phi = (float)((temp >= 0) ? temp : (temp + 2 * AT_MATH_PI));
            auto theta = aten::acos(dir.y / r);

            float u = phi / (2 * AT_MATH_PI);

            // NOTE:
            // +v
            //  |
            //  0---> +u
            // Convert v coordinate as from bottom to top is 0 to 1.
            float v = 1 - theta / AT_MATH_PI;

            aten::vec3 uv = aten::vec3(u, v, 0);

            return uv;
        }
    };
}
