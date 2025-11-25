#include <array>

#include "renderer/pathtracing/pathtracing.h"

#include "accelerator/threaded_bvh_traverser.h"
#include "material/material_impl.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/npr/npr_impl.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (-1)
#define BREAK_Y    (-1)
#pragma optimize( "", off)
#endif

namespace aten
{
    void PathTracing::radiance_with_feature_line(
        int32_t idx,
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        const context& ctxt,
        scene* scene,
        const aten::CameraParameter& camera)
    {
        const auto pixel_width = Camera::ComputePixelWidthAtDistance(camera, 1);

        // TODO: These value should be configurable.
        constexpr float feature_line_width = 1;
        constexpr float albedo_threshold = 0.1f;
        constexpr float normal_threshold = 0.1f;
        static const aten::vec3 LineColor(0, 1, 0);

        auto& sample_ray_info = feature_line_sample_ray_infos_[idx];
        auto* sampler = &path_host_.paths.sampler[idx];
        const auto& ray = rays_[idx];

        AT_NAME::npr::GenerateSampleRayAndDiscPerQueryRay<SampleRayNum>(
            sample_ray_info.descs, sample_ray_info.disc,
            ray, *sampler,
            feature_line_width, pixel_width);

        int32_t depth = 0;

        while (depth < m_maxDepth) {
            bool willContinue = true;
            Intersection isect;

            const auto& ray = rays_[idx];

            path_host_.paths.attrib[idx].isHit = false;

            bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect,
                ctxt,
                ray,
                AT_MATH_EPSILON, AT_MATH_INF);

            if (is_hit) {
                AT_NAME::npr::ShadeSampleRay(
                    LineColor, feature_line_width, pixel_width,
                    idx, depth,
                    ctxt, camera,
                    ray, isect,
                    path_host_.paths,
                    feature_line_sample_ray_infos_.data()
                );

                path_host_.paths.attrib[idx].isHit = true;

                shade(
                    idx,
                    path_host_.paths, ctxt, rays_.data(), shadow_rays_.data(),
                    isect, scene, m_rrDepth, depth);

                const auto& mtrl = ctxt.GetMaterial(isect.mtrlid);

                std::ignore = AT_NAME::HitShadowRay(
                    idx, depth,
                    ctxt, mtrl,
                    path_host_.paths,
                    shadow_rays_[idx]);

                willContinue = !path_host_.paths.attrib[idx].is_terminated;
            }
            else {
                AT_NAME::npr::ShadeMissSampleRay(
                    LineColor, feature_line_width, pixel_width,
                    idx, depth,
                    ctxt,
                    ray, isect,
                    path_host_.paths,
                    feature_line_sample_ray_infos_.data()
                );

                auto ibl = scene->getIBL();
                if (ibl && enable_envmap_) {
                    ShadeMissWithEnvmap(
                        idx,
                        ix, iy,
                        width, height,
                        depth,
                        bg_,
                        ctxt, camera,
                        path_host_.paths, rays_[idx]);
                }
                else {
                    ShadeMiss(
                        idx,
                        depth,
                        bg_.bg_color,
                        path_host_.paths);
                }

                willContinue = false;
            }

            if (!willContinue) {
                break;
            }

            depth++;
        }
    }
}
