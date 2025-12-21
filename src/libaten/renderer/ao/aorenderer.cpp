#include "renderer/ao/aorenderer.h"

#include "accelerator/threaded_bvh_traverser.h"
#include "material/diffuse.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/ao/aorenderer_impl.h"
#include "renderer/pathtracing/pathtracing_impl.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (6)
#define BREAK_Y    (0)
#pragma optimize( "", off)
#endif

namespace aten
{
    float AORenderer::radiance(
        int32_t idx,
        uint32_t rnd,
        const context& ctxt,
        const ray& inRay,
        scene* scene)
    {
        const auto frame = GetFrameCount();
        const auto scramble = rnd * 0x1fe3434f * ((frame + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
        path_host_.paths.sampler[idx].init(frame % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM), 4 + 5 * 300, scramble);

        Intersection& isect = isects_[idx];

        bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
            isect,
            ctxt,
            inRay,
            AT_MATH_EPSILON, AT_MATH_INF);

        float ao_color{ 0.0F };

        if (is_hit) {
            path_host_.paths.attrib[idx].attr.isHit = true;

            ao_color = AT_NAME::ao::ShandeByAO(
                m_numAORays, m_AORadius,
                path_host_.paths.sampler[idx], ctxt, inRay, isect);
        }
        else {
            ao_color = AT_NAME::ao::ShadeByAOIfHitMiss(
                idx,
                path_host_.paths);
        }

        return ao_color;
    }

    void AORenderer::OnRender(
        context& ctxt,
        Destination& dst,
        scene* scene,
        Camera* camera)
    {
#if 1
        RenderAO(ctxt, dst, scene, camera);
#else
        // Experimental for AO with biralteral filtering.
        RenderAOWithBilateralFilter(ctxt, dst, scene, camera);
#endif
    }

    void AORenderer::RenderAO(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        Camera* camera)
    {
        int32_t width = dst.width;
        int32_t height = dst.height;
        uint32_t samples = dst.sample;

        max_depth_ = dst.maxDepth;

        path_host_.init(width, height);
        if (isects_.empty()) {
            isects_.resize(width * height);
        }

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
            auto idx = OMPUtil::getThreadIdx();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    vec3 col = vec3(0);
                    uint32_t cnt = 0;

#ifdef RELEASE_DEBUG
                    if (x == BREAK_X && y == BREAK_Y) {
                        DEBUG_BREAK();
                    }
#endif

                    int32_t idx = y * width + x;

                    const auto rnd = aten::getRandom(idx);
                    const auto& camsample = camera->param();

                    aten::ray ray;

                    GeneratePath(
                        ray,
                        idx,
                        x, y, width, height,
                        0, GetFrameCount(),
                        path_host_.paths,
                        camsample,
                        rnd);

                    const auto ao_color = radiance(
                        idx, rnd,
                        ctxt, ray, scene);

                    if (!path_host_.paths.attrib[idx].attr.is_terminated) {
                        path_host_.paths.contrib[idx].contrib = aten::vec3(ao_color);
                    }
                    auto c = path_host_.paths.contrib[idx].contrib;

                    col += c;
                    cnt++;

                    if (path_host_.paths.attrib[idx].attr.is_terminated) {
                        break;
                    }

                    col /= (float)cnt;
                    dst.buffer->put(x, y, vec4(col, 1));
                }
            }
        }
    }

    void AORenderer::RenderAOWithBilateralFilter(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        Camera* camera)
    {
        int32_t width = dst.width;
        int32_t height = dst.height;
        uint32_t samples = dst.sample;

        max_depth_ = dst.maxDepth;

        path_host_.init(width, height);
        if (isects_.empty()) {
            isects_.resize(width * height);
        }
        if (bilateral_filter_.empty()) {
            bilateral_filter_.resize(width * height);
        }

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
            auto idx = OMPUtil::getThreadIdx();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
#ifdef RELEASE_DEBUG
                    if (x == BREAK_X && y == BREAK_Y) {
                        DEBUG_BREAK();
                    }
#endif

                    int32_t idx = y * width + x;

                    const auto rnd = aten::getRandom(idx);
                    const auto& camsample = camera->param();

                    aten::ray ray;

                    GeneratePath(
                        ray,
                        idx,
                        x, y, width, height,
                        0, GetFrameCount(),
                        path_host_.paths,
                        camsample,
                        rnd);

                    const auto ao_color = radiance(
                        idx, rnd,
                        ctxt, ray, scene);

                    if (!path_host_.paths.attrib[idx].attr.is_terminated) {
                        path_host_.paths.contrib[idx].contrib = aten::vec3(ao_color);
                    }
                    auto c = path_host_.paths.contrib[idx].contrib;
                    if (path_host_.paths.attrib[idx].attr.is_terminated) {
                        break;
                    }
                }
            }

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
#if 1
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    const auto idx = y * width + x;
                    const auto filtered_color = AT_NAME::ao::ApplyBilateralFilter<aten::PathContrib, float, true>(
                        x, y,
                        width, height,
                        2.0F, 2.0F,
                        path_host_.paths.contrib, isects_.data()
                    );
                    bilateral_filter_[idx] = filtered_color;
                }
            }
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    const auto idx = y * width + x;
                    const auto filtered_color = AT_NAME::ao::ApplyBilateralFilter<aten::PathContrib, float, false>(
                        x, y,
                        width, height,
                        2.0F, 2.0F,
                        path_host_.paths.contrib, isects_.data()
                    );
                    bilateral_filter_[idx] *= filtered_color;
                    auto c = bilateral_filter_[idx];
                    c = c < 1.0F ? c * 0.5F : c;
                    dst.buffer->put(x, y, vec4(c, c, c, 1));
                }
#else
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    const auto idx = y * width + x;
                    const auto filtered_color = AT_NAME::ao::ApplyBilateralFilterOrthogonal<aten::PathContrib, float, 3, 3>(
                        x, y,
                        width, height,
                        2.0F, 2.0F,
                        path_host_.paths.contrib, isects_.data()
                    );
                    bilateral_filter_[idx] = filtered_color;
                }
            }
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    const auto idx = y * width + x;
                    const auto filtered_color = AT_NAME::ao::ApplyBilateralFilterOrthogonal<aten::PathContrib, float, 3, -3>(
                        x, y,
                        width, height,
                        2.0F, 2.0F,
                        path_host_.paths.contrib,
                        isects_.data()
                    );
                    bilateral_filter_[idx] *= filtered_color;
                    auto c = bilateral_filter_[idx];
                    c = c < 1.0F ? c * 0.5F : c;
                    dst.buffer->put(x, y, vec4(c, c, c, 1));
                }
#endif
            }
        }
    }
}
