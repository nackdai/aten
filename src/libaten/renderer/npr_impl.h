#pragma once

#include "renderer/feature_line.h"
#include "renderer/pathtracing_impl.h"
#include "geometry/EvaluateHitResult.h"
#include "misc/tuple.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif

namespace AT_NAME
{
    inline AT_DEVICE_MTRL_API void ComputeFeatureLineContribution(
        real closest_sample_ray_distance,
        AT_NAME::Path paths,
        int32_t idx,
        int32_t sample_ray_num,
        const aten::vec3& line_color)
    {
        auto pdf_feature_line = real(1) / sample_ray_num;
        pdf_feature_line = pdf_feature_line * (closest_sample_ray_distance * closest_sample_ray_distance);

        const auto pdfb = paths.throughput[idx].pdfb;
        const auto weight = _detail::ComputeBalanceHeuristic(pdfb, pdf_feature_line);
        const auto contrib = paths.throughput[idx].throughput * weight * line_color;

        _detail::CopyVec3(paths.contrib[idx].contrib, contrib);

        // Line is treated as light. So, query path need to be killed and termnated not to bounce anymore.
        paths.attrib[idx].isKill = true;
        paths.attrib[idx].isTerminate = true;
    }

    template <int SampleRayNum, typename SCENE=void>
    inline AT_DEVICE_MTRL_API aten::tuple<int32_t, float> ShadeMissFeatureLineSampleRay(
        std::array<AT_NAME::FeatureLine::SampleRayDesc, SampleRayNum>& sample_ray_descs,
        const AT_NAME::context& ctxt,
        const aten::ray& ray,
        AT_NAME::FeatureLine::Disc& disc,
        const int32_t depth,
        const real feature_line_width,
        const real pixel_width,
        SCENE* scene = nullptr)
    {
        auto closest_sample_ray_distance = std::numeric_limits<real>::max();
        int32_t closest_sample_ray_idx = -1;
        real hit_point_distance = 0;

        // NOTE:
        // In order to compute sample ray, previous disc and next disc are necessary.
        // In first bounce, initial point is camera original.
        // So, previous disc is not necessary.

        aten::FeatureLine::Disc prev_disc;

        if (depth > 0) {
            // Make dummy point to compute next sample ray.
            const auto dummy_query_hit_pos = ray.org + real(100) * ray.dir;
            hit_point_distance = length(dummy_query_hit_pos - disc.center);

            prev_disc = disc;
            disc = aten::FeatureLine::computeNextDisc(
                dummy_query_hit_pos,
                ray.dir,
                prev_disc.radius,
                hit_point_distance,
                disc.accumulated_distance);
        }

        for (int32_t i = 0; i < SampleRayNum; i++) {
            if (sample_ray_descs[i].is_terminated) {
                continue;
            }

            auto sample_ray = aten::FeatureLine::getRayFromDesc(sample_ray_descs[i]);
            if (depth > 0) {
                // Generate next sample ray.
                const auto res_next_sample_ray = aten::FeatureLine::computeNextSampleRay(
                    sample_ray_descs[i],
                    prev_disc, disc);
                const auto is_sample_ray_valid = std::get<0>(res_next_sample_ray);
                if (!is_sample_ray_valid) {
                    sample_ray_descs[i].is_terminated = true;
                    continue;
                }
                sample_ray = std::get<1>(res_next_sample_ray);
            }

            Intersection isect_sample_ray;
            hitrecord hrec_sample;

            auto is_hit = false;

            if constexpr (!std::is_void_v<std::remove_pointer_t<SCENE>>) {
                // NOTE:
                // operation has to be related with template arg SCENE.
                if (scene) {
                    is_hit = scene->hit(ctxt, sample_ray, AT_MATH_EPSILON, AT_MATH_INF, isect_sample_ray);
                }
            }
            else {
                // TODO
            }

            if (is_hit) {
                // Evaluate sample ray hit.
                const auto& obj = ctxt.GetObject(isect_sample_ray.objid);
                AT_NAME::evaluate_hit_result(hrec_sample, obj, ctxt, ray, isect_sample_ray);

                // Compute distance between point which sample ray projects onth query ray and query ray org.
                const auto distance_sample_pos_on_query_ray = aten::FeatureLine::computeDistanceBetweenProjectedPosOnRayAndRayOrg(
                    hrec_sample.p, ray);

                // Update closest sample ray.
                if (distance_sample_pos_on_query_ray < closest_sample_ray_distance) {
                    // Check if hit point by sample ray is within feature line width.
                    const auto is_line_width = aten::FeatureLine::isInLineWidth(
                        feature_line_width,
                        ray,
                        hrec_sample.p,
                        disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
                        pixel_width);

                    if (is_line_width) {
                        // If sample ray doesn't hit anything, it is forcibly feature line.
                        closest_sample_ray_idx = i;
                        closest_sample_ray_distance = distance_sample_pos_on_query_ray;
                    }
                }
            }
        }

        return aten::make_tuple<int32_t, float>(closest_sample_ray_idx, closest_sample_ray_distance);
    }
}
