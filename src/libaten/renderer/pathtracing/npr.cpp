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
        Path& paths,
        const context& ctxt,
        ray* rays,
        ShadowRay* shadow_rays,
        int32_t rrDepth,
        int32_t maxDepth,
        Camera* cam,
        scene* scene,
        aten::BackgroundResource& bg)
    {
        int32_t depth = 0;

        const auto& ray = rays[idx];
        auto* sampler = &paths.sampler[idx];

        const auto& cam_org = cam->param().origin;

        const auto pixel_width = cam->ComputePixelWidthAtDistance(1);

        constexpr size_t SampleRayNum = 8;

        // TODO: These value should be configurable.
        constexpr float feature_line_width = 1;
        constexpr float albedo_threshold = 0.1f;
        constexpr float normal_threshold = 0.1f;
        static const aten::vec3 LineColor(0, 1, 0);

        std::array<AT_NAME::npr::FeatureLine::SampleRayDesc, SampleRayNum> sample_ray_descs;
        AT_NAME::npr::FeatureLine::Disc disc;

        AT_NAME::npr::GenerateSampleRayAndDiscPerQueryRay<SampleRayNum>(
            sample_ray_descs, disc,
            ray, *sampler,
            feature_line_width, pixel_width);

        while (depth < maxDepth) {
            hitrecord hrec_query;

            bool willContinue = true;
            Intersection isect;

            // Current closest distance to feature line point.
            auto closest_feature_line_point_distance = std::numeric_limits<float>::max();

            // Whether the feature line point has been found.
            bool is_found_feature_line_point = false;

            float hit_point_distance = 0;

            // Check if the query ray hits any object.
            bool is_query_ray_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect,
                ctxt,
                ray,
                AT_MATH_EPSILON, AT_MATH_INF);

            if (is_query_ray_hit) {
                const auto& obj = ctxt.GetObject(isect.objid);
                AT_NAME::evaluate_hit_result(hrec_query, obj, ctxt, ray, isect);

                const auto distance_query_ray_hit = length(hrec_query.p - ray.org);

                // disc.centerはquery_ray.orgに一致する.
                // ただし、最初だけは、query_ray.orgはカメラ視点になっているが、
                // accumulated_distanceでカメラとdiscの距離がすでに含まれている.
                hit_point_distance = length(hrec_query.p - disc.center);

                const auto prev_disc = disc;
                disc = AT_NAME::npr::FeatureLine::ComputeDiscAtQueryRayHitPoint(
                    hrec_query.p,
                    ray.dir,
                    prev_disc.radius,
                    hit_point_distance,
                    disc.accumulated_distance);

                for (size_t i = 0; i < SampleRayNum; i++) {
                    if (sample_ray_descs[i].is_terminated) {
                        continue;
                    }

                    auto sample_ray = AT_NAME::npr::GetSampleRay(
                        depth,
                        sample_ray_descs[i],
                        prev_disc, disc);
                    if (sample_ray_descs[i].is_terminated) {
                        continue;
                    }

                    // Check if the sample ray hits any object.
                    Intersection isect_sample_ray;
                    bool is_sample_ray_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                        isect_sample_ray,
                        ctxt,
                        sample_ray,
                        AT_MATH_EPSILON, AT_MATH_INF);

                    if (is_sample_ray_hit) {
                        // Query ray hits and then sample ray hits.
                        aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = AT_NAME::npr::EvaluateQueryAndSampleRayHit(
                            sample_ray_descs[i],
                            ctxt, cam_org,
                            ray, hrec_query, distance_query_ray_hit,
                            isect_sample_ray,
                            disc,
                            is_found_feature_line_point,
                            closest_feature_line_point_distance,
                            feature_line_width, pixel_width,
                            albedo_threshold, normal_threshold);
                    }
                    else {
                        // Query ray hits but sample ray doesn't hit anything.
                        aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = AT_NAME::npr::EvaluateQueryRayHitButSampleRayNotHit(
                            sample_ray_descs[i],
                            ray, hrec_query, distance_query_ray_hit,
                            sample_ray, disc,
                            is_found_feature_line_point,
                            closest_feature_line_point_distance,
                            feature_line_width, pixel_width);
                    }

                    const auto mtrl = ctxt.GetMaterial(hrec_query.mtrlid);
                    if (!mtrl.attrib.is_glossy) {
                        // In non glossy material case, sample ray doesn't bounce anymore.
                        // TODO
                        // Even if material is glossy, how glossy depends on parameter (e.g. roughness etc).
                        // So, I need to consider how to indetify if sample ray bounce is necessary based on material.
                        sample_ray_descs[i].is_terminated = true;
                    }
                }

                if (!is_found_feature_line_point) {
                    shade(
                        idx, paths, ctxt, rays, shadow_rays,
                        isect, scene, rrDepth, depth);
                    AT_NAME::HitShadowRay(
                        idx, depth, ctxt, paths, shadow_rays[idx]);
                }
            }
            else {
                // Query ray doesn't hit anything, but evaluate a possibility that sample ray might hit something.

                // NOTE:
                // In order to compute sample ray, previous disc and next disc are necessary.
                // In first bounce, initial point is camera original.
                // So, previous disc is not necessary.

                // Query ray doesn't hit to anything, and we can't create the next disc at query ray hit point.
                // But, the disc is necessary. So, the next disc is created forcibly from the dummy query ray hit point.
                AT_NAME::npr::FeatureLine::Disc prev_disc;
                hit_point_distance = CreateNextDiscByDummyQueryRayHitPoint(depth, hit_point_distance, ray, prev_disc, disc);

                for (int32_t i = 0; i < SampleRayNum; i++) {
                    if (sample_ray_descs[i].is_terminated) {
                        continue;
                    }

                    auto sample_ray = AT_NAME::npr::GetSampleRay(
                        depth,
                        sample_ray_descs[i],
                        prev_disc, disc);
                    if (sample_ray_descs[i].is_terminated) {
                        continue;
                    }

                    // Check if the sample ray hits any object.
                    Intersection isect_sample_ray;
                    bool is_sample_ray_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                        isect_sample_ray,
                        ctxt,
                        sample_ray,
                        AT_MATH_EPSILON, AT_MATH_INF);

                    if (is_sample_ray_hit) {
                        // Query ray doesn't hit, but sample ray hits.
                        aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = AT_NAME::npr::EvaluateQueryRayNotHitButSampleRayHit(
                            ctxt, ray,
                            isect_sample_ray,
                            disc,
                            is_found_feature_line_point,
                            closest_feature_line_point_distance,
                            feature_line_width, pixel_width);
                    }
                    else {
                        // Sample ray doesn't hit anything.
                        // So, terminate sample ray traversing immediately.
                        sample_ray_descs[i].is_terminated = true;
                        break;
                    }
                }

                if (!is_found_feature_line_point) {
                    shadeMiss(ctxt, idx, scene, depth, paths, rays, bg);
                    willContinue = false;
                }
            }

            if (is_found_feature_line_point) {
                AT_NAME::npr::ComputeFeatureLineContribution<SampleRayNum>(
                    closest_feature_line_point_distance,
                    paths, idx, LineColor);
                willContinue = false;
            }

            if (!willContinue) {
                break;
            }

            depth++;

            disc.accumulated_distance += hit_point_distance;
        }
    }
}
