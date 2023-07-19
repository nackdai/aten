#pragma once

#include "defs.h"
#include "camera/camera.h"
#include "camera/pinhole.h"
#include "light/ibl.h"
#include "math/ray.h"
#include "renderer/pt_params.h"

namespace AT_NAME
{
    inline AT_DEVICE_API void generate_path(
        aten::ray& generated_ray,
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t sample, uint32_t frame,
        aten::Path& paths,
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

        // TODO
        paths.contrib[idx].contrib = aten::vec3(0);
    }

    inline AT_DEVICE_API void shader_miss(
        int32_t idx,
        int32_t bounce,
        const aten::vec3& bg,
        aten::Path& paths)
    {
        if (!paths.attrib[idx].isTerminate && !paths.attrib[idx].isHit) {
            if (bounce == 0) {
                paths.attrib[idx].isKill = true;
            }

            auto contrib = paths.throughput[idx].throughput * bg;
            paths.contrib[idx].contrib += aten::vec3(contrib.x, contrib.y, contrib.z);

            paths.attrib[idx].isTerminate = true;
        }
    }

    inline AT_DEVICE_API void shader_miss_with_envmap(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        int32_t bounce,
        int32_t envmap_idx,
        float envmapAvgIllum, float envmapMultiplyer,
        const aten::CameraParameter& camera,
        aten::Path& paths,
        const aten::ray& ray)
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
            const auto bg = tex2DLod<float4>(ctxt.textures[param.envmapidx], uv.x, uv.y);
#else
            const auto bg = AT_NAME::sampleTexture(envmap_idx, uv.x, uv.y, aten::vec4(1));
#endif
            auto emit = aten::vec3(bg.x, bg.y, bg.z);

            float misW = 1.0f;
            if (bounce == 0
                || (bounce == 1 && paths.attrib[idx].isSingular))
            {
                paths.attrib[idx].isKill = true;
            }
            else {
                auto pdfLight = AT_NAME::ImageBasedLight::samplePdf(emit, envmapAvgIllum);
                misW = paths.throughput[idx].pdfb / (pdfLight + paths.throughput[idx].pdfb);

                emit *= envmapMultiplyer;
            }

            auto contrib = paths.throughput[idx].throughput * misW * emit;
            paths.contrib[idx].contrib += aten::vec3(contrib.x, contrib.y, contrib.z);

            paths.attrib[idx].isTerminate = true;
        }
    }
}
