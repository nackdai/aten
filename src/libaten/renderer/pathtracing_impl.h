#pragma once

#include "defs.h"
#include "camera/camera.h"
#include "camera/pinhole.h"
#include "light/ibl.h"
#include "math/ray.h"
#include "renderer/aov.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#include "kernel/pt_params.h"
#else
#include "scene/host_scene_context.h"
#include "renderer/pt_params.h"
#endif

namespace AT_NAME
{
    inline AT_DEVICE_MTRL_API void generate_path(
        aten::ray& generated_ray,
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t sample, uint32_t frame,
        AT_NAME::Path& paths,
        const aten::CameraParameter& camera,
        const uint32_t rnd)
    {
        paths.attrib[idx].isHit = false;

        if (paths.attrib[idx].isKill) {
            paths.attrib[idx].isTerminate = true;
            return;
        }

        auto scramble = rnd * 0x1fe3434f
            * (((frame + sample) + 133 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
        paths.sampler[idx].init(
            (frame + sample) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
            0,
            scramble);

        float r1 = paths.sampler[idx].nextSample();
        float r2 = paths.sampler[idx].nextSample();

        float s = (ix + r1) / (float)(camera.width);
        float t = (iy + r2) / (float)(camera.height);

        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        generated_ray = camsample.r;

        paths.throughput[idx].throughput = aten::vec3(1);
        paths.throughput[idx].pdfb = 1.0f;
        paths.attrib[idx].isTerminate = false;
        paths.attrib[idx].isSingular = false;

        paths.contrib[idx].samples += 1;
    }

    namespace detail {
        template <typename A, typename B>
        inline AT_DEVICE_MTRL_API void add_vec3(A& dst, const B& add)
        {
            if constexpr (std::is_same_v<A, B> && std::is_same_v<A, aten::vec3>) {
                dst += add;
            }
            else {
                dst += make_float3(add.x, add.y, add.z);
            }
        }
    }

    template <typename AOV_BUFFER_TYPE = aten::vec4>
    inline AT_DEVICE_MTRL_API void shade_miss(
        int32_t idx,
        int32_t bounce,
        const aten::vec3& bg,
        AT_NAME::Path& paths,
        AOV_BUFFER_TYPE* aov_normal_depth = nullptr,
        AOV_BUFFER_TYPE* aov_albedo_meshid = nullptr)
    {
        if (!paths.attrib[idx].isTerminate && !paths.attrib[idx].isHit) {
            if (bounce == 0) {
                paths.attrib[idx].isKill = true;

                if (aov_normal_depth != nullptr && aov_albedo_meshid != nullptr)
                {
                    // Export bg color to albedo buffer.
                    AT_NAME::FillBasicAOVsIfHitMiss(
                        aov_normal_depth[idx],
                        aov_albedo_meshid[idx], bg);
                }
            }

            auto contrib = paths.throughput[idx].throughput * bg;
            detail::add_vec3(paths.contrib[idx].contrib, contrib);

            paths.attrib[idx].isTerminate = true;
        }
    }

    template <typename AOV_BUFFER_TYPE = aten::vec4>
    inline AT_DEVICE_MTRL_API void shade_miss_with_envmap(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        int32_t bounce,
        int32_t envmap_idx,
        float envmapAvgIllum, float envmapMultiplyer,
        const AT_NAME::context& ctxt,
        const aten::CameraParameter& camera,
        AT_NAME::Path& paths,
        const aten::ray& ray,
        AOV_BUFFER_TYPE* aov_normal_depth = nullptr,
        AOV_BUFFER_TYPE* aov_albedo_meshid = nullptr)
    {
        if (!paths.attrib[idx].isTerminate && !paths.attrib[idx].isHit) {
            aten::vec3 dir = ray.dir;

            if (bounce == 0) {
                // Suppress jittering envrinment map.
                // So, re-sample ray without random.

                // TODO
                // More efficient way...

                float s = ix / (float)(width);
                float t = iy / (float)(height);

                AT_NAME::CameraSampleResult camsample;
                AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

                dir = camsample.r.dir;
            }

            auto uv = AT_NAME::envmap::convertDirectionToUV(dir);

#ifdef __CUDACC__
            // envmapidx is index to array of textures in context.
            // In GPU, sampleTexture requires texture id of CUDA. So, arguments is different.
            const auto bg = tex2D<float4>(ctxt.textures[envmap_idx], uv.x, uv.y);
#else
            const auto bg = AT_NAME::sampleTexture(envmap_idx, uv.x, uv.y, aten::vec4(1));
#endif
            auto emit = aten::vec3(bg.x, bg.y, bg.z);

            float misW = 1.0f;
            if (bounce == 0
                || (bounce == 1 && paths.attrib[idx].isSingular))
            {
                paths.attrib[idx].isKill = true;

                if (aov_normal_depth != nullptr && aov_albedo_meshid != nullptr)
                {
                    // Export bg color to albedo buffer.
                    AT_NAME::FillBasicAOVsIfHitMiss(
                        aov_normal_depth[idx],
                        aov_albedo_meshid[idx], bg);
                }
            }
            else {
                auto pdfLight = AT_NAME::ImageBasedLight::samplePdf(emit, envmapAvgIllum);
                misW = paths.throughput[idx].pdfb / (pdfLight + paths.throughput[idx].pdfb);

                emit *= envmapMultiplyer;
            }

            auto contrib = paths.throughput[idx].throughput * misW * emit;
            detail::add_vec3(paths.contrib[idx].contrib, contrib);

            paths.attrib[idx].isTerminate = true;
        }
    }
}
