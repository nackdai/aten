#pragma once

#include "renderer/npr/feature_line.h"
#include "renderer/pathtracing_impl.h"
#include "geometry/EvaluateHitResult.h"
#include "misc/tuple.h"
#include "misc/misc.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif

namespace AT_NAME
{
    template <size_t SampleRayNum>
    inline AT_DEVICE_MTRL_API void GenerateSampleRayPerQueryRay(
        std::array<AT_NAME::FeatureLine::SampleRayDesc, SampleRayNum>& sample_ray_descs,
        AT_NAME::FeatureLine::Disc& disc,
        const aten::ray& query_ray,
        aten::sampler& sampler,
        const real feature_line_width,
        const real pixel_width)
    {
        disc = AT_NAME::FeatureLine::generateDisc(query_ray, feature_line_width, pixel_width);

        for (size_t i = 0; i < SampleRayNum; i++) {
            const auto sample_ray = AT_NAME::FeatureLine::generateSampleRay(
                sample_ray_descs[i], sampler, query_ray, disc);
            AT_NAME::FeatureLine::storeRayToDesc(sample_ray_descs[i], sample_ray);
            sample_ray_descs[i].is_terminated = false;
        }

        disc.accumulated_distance = 1;
    }

    template <size_t SampleRayNum>
    inline AT_DEVICE_MTRL_API void ComputeFeatureLineContribution(
        real closest_sample_ray_distance,
        AT_NAME::Path paths,
        int32_t idx,
        const aten::vec3& line_color)
    {
        constexpr auto PdfSampleRay = real(1) / SampleRayNum;
        auto pdf_feature_line = PdfSampleRay * (closest_sample_ray_distance * closest_sample_ray_distance);

        const auto pdfb = paths.throughput[idx].pdfb;
        const auto weight = _detail::ComputeBalanceHeuristic(pdfb, pdf_feature_line);
        const auto contrib = paths.throughput[idx].throughput * weight * line_color;

        _detail::CopyVec3(paths.contrib[idx].contrib, contrib);

        // Line is treated as light. So, query path need to be killed and termnated not to bounce anymore.
        paths.attrib[idx].isKill = true;
        paths.attrib[idx].isTerminate = true;
    }

    inline AT_DEVICE_MTRL_API aten::ray GetSampleRay(
        int32_t depth,
        AT_NAME::FeatureLine::SampleRayDesc& sample_ray_desc,
        const AT_NAME::FeatureLine::Disc& prev_disc,
        const AT_NAME::FeatureLine::Disc& curr_disc)
    {
        auto sample_ray = AT_NAME::FeatureLine::getRayFromDesc(sample_ray_desc);
        if (depth > 0) {
            // Generate next sample ray.
            const auto res_next_sample_ray = AT_NAME::FeatureLine::computeNextSampleRay(
                sample_ray_desc,
                prev_disc, curr_disc);
            const auto is_sample_ray_valid = aten::get<0>(res_next_sample_ray);
            if (is_sample_ray_valid) {
                sample_ray = aten::get<1>(res_next_sample_ray);
            }
            else {
                sample_ray_desc.is_terminated = true;
            }
        }

        return sample_ray;
    }

    inline AT_DEVICE_MTRL_API aten::tuple<bool, float> EvaluateQueryAndSampleRayHit(
        AT_NAME::FeatureLine::SampleRayDesc& sample_ray_desc,
        const AT_NAME::context& ctxt,
        const aten::vec3& cam_org,
        const aten::ray& query_ray,
        const aten::hitrecord& hrec_query,
        const real distance_query_ray_hit,
        const aten::Intersection& isect_sample_ray,
        const AT_NAME::FeatureLine::Disc& disc,
        bool is_found_closest_sample_ray_hit,
        real closest_sample_ray_distance,
        const real FeatureLineWidth,
        const real pixel_width,
        const real ThresholdAlbedo,
        const real ThresholdNormal)
    {
        // Query ray hits and then sample ray hits.

        const auto& obj = ctxt.GetObject(isect_sample_ray.objid);

        aten::hitrecord hrec_sample;
        AT_NAME::evaluate_hit_result(hrec_sample, obj, ctxt, query_ray, isect_sample_ray);

        // If sample ray hit with the different mesh from query ray one, this sample ray won't bounce in next loop.
        sample_ray_desc.is_terminated = hrec_sample.meshid != hrec_query.meshid;
        sample_ray_desc.prev_ray_hit_pos = hrec_sample.p;
        sample_ray_desc.prev_ray_hit_nml = hrec_sample.normal;

        const auto distance_sample_pos_on_query_ray = AT_NAME::FeatureLine::computeDistanceBetweenProjectedPosOnRayAndRayOrg(
            hrec_sample.p, query_ray);

        const auto is_line_width = AT_NAME::FeatureLine::isInLineWidth(
            FeatureLineWidth,
            query_ray,
            hrec_sample.p,
            disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
            pixel_width);

        if (is_line_width) {
            aten::MaterialParameter mtrl_tmp;
            AT_NAME::FillMaterial(mtrl_tmp, ctxt,hrec_query.mtrlid, hrec_query.isVoxel);
            const auto query_albedo = material::sampleAlbedoMap(&mtrl_tmp, hrec_query.u, hrec_query.v);

            AT_NAME::FillMaterial(mtrl_tmp, ctxt, hrec_sample.mtrlid, hrec_sample.isVoxel);
            const auto sample_albedo = material::sampleAlbedoMap(&mtrl_tmp, hrec_sample.u, hrec_sample.v);

            const auto query_depth = length(hrec_query.p - cam_org);
            const auto sample_depth = length(hrec_sample.p - cam_org);

            const auto is_feature_line = AT_NAME::FeatureLine::evaluateMetrics(
                query_ray.org,
                hrec_query, hrec_sample,
                query_albedo, sample_albedo,
                query_depth, sample_depth,
                ThresholdAlbedo, ThresholdNormal,
                2);
            if (is_feature_line) {
                if (distance_sample_pos_on_query_ray < closest_sample_ray_distance
                    && distance_sample_pos_on_query_ray < distance_query_ray_hit)
                {
                    // Deal with sample hit point as FeatureLine.
                    is_found_closest_sample_ray_hit = true;
                    closest_sample_ray_distance = distance_sample_pos_on_query_ray;
                }
                else if (distance_query_ray_hit < closest_sample_ray_distance) {
                    // Deal with query hit point as FeatureLine.
                    is_found_closest_sample_ray_hit = true;
                    closest_sample_ray_distance = distance_query_ray_hit;
                }
            }
        }
        return aten::make_tuple<bool, float>(is_found_closest_sample_ray_hit, closest_sample_ray_distance);
    }

    inline AT_DEVICE_MTRL_API aten::tuple<bool, float> EvaluateQueryRayHitButSampleRayNotHit(
        AT_NAME::FeatureLine::SampleRayDesc& sample_ray_desc,
        const aten::ray& query_ray,
        const aten::hitrecord& hrec_query,
        const real distance_query_ray_hit,
        const aten::ray& sample_ray,
        const AT_NAME::FeatureLine::Disc& disc,
        bool is_found_closest_sample_ray_hit,
        real closest_sample_ray_distance,
        const real feature_line_width,
        const real pixel_width)
    {
        // Query ray hits but sample ray doesn't hit anything.

        // Compute plane which query ray hits.
        const auto query_hit_plane = AT_NAME::FeatureLine::computePlane(hrec_query);

        // Even if sample ray doesn't hit anything, compute point which sample ray hit onto plane which query ray hits.
        const auto res_sample_ray_dummy_hit = AT_NAME::FeatureLine::computeRayHitPosOnPlane(
            query_hit_plane, sample_ray);

        const auto is_hit_sample_ray_dummy_plane = aten::get<0>(res_sample_ray_dummy_hit);
        if (is_hit_sample_ray_dummy_plane) {
            // Get point which sample ray hit onto plane which query ray hits.
            const auto sample_ray_dummy_hit_pos = aten::get<1>(res_sample_ray_dummy_hit);

            const auto distance_sample_pos_on_query_ray = AT_NAME::FeatureLine::computeDistanceBetweenProjectedPosOnRayAndRayOrg(
                sample_ray_dummy_hit_pos, query_ray);

            // If point which sample ray hits is within feature line width,
            // that point is treated as feature line forcibly without checking metrics.
            const auto is_line_width = AT_NAME::FeatureLine::isInLineWidth(
                feature_line_width,
                query_ray,
                sample_ray_dummy_hit_pos,
                disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
                pixel_width);
            if (is_line_width) {
                if (distance_sample_pos_on_query_ray < closest_sample_ray_distance
                    && distance_sample_pos_on_query_ray < distance_query_ray_hit)
                {
                    // Deal with sample hit point as FeatureLine.
                    is_found_closest_sample_ray_hit = true;
                    closest_sample_ray_distance = distance_sample_pos_on_query_ray;
                }
                else if (distance_query_ray_hit < closest_sample_ray_distance) {
                    // Deal with query hit point as FeatureLine.
                    is_found_closest_sample_ray_hit = true;
                    closest_sample_ray_distance = distance_query_ray_hit;
                }
            }
        }

        // Sample ray doesn't hit anything. It means sample ray causes hit miss.
        // So, traversing sample ray is terminated.
        sample_ray_desc.is_terminated = true;

        return aten::make_tuple<bool, float>(is_found_closest_sample_ray_hit, closest_sample_ray_distance);
    }

    inline AT_DEVICE_MTRL_API real ComputeNextDiscByDummyQueryRayHit(
        int32_t depth,
        real hit_point_distance,
        const aten::ray& query_ray,
        FeatureLine::Disc& prev_disc,
        FeatureLine::Disc& disc)
    {
        if (depth > 0) {
            // Query ray doesn't hit anything, so there is not specifiy hit point.
            // So, in order to compute next sample ray, make dummy hit point.
            const auto dummy_query_hit_pos = query_ray.org + real(100) * query_ray.dir;
            hit_point_distance = length(dummy_query_hit_pos - disc.center);

            prev_disc = disc;
            disc = AT_NAME::FeatureLine::computeNextDisc(
                dummy_query_hit_pos,
                query_ray.dir,
                prev_disc.radius,
                hit_point_distance,
                disc.accumulated_distance);
        }

        return hit_point_distance;
    }

    inline AT_DEVICE_MTRL_API aten::tuple<bool, float> EvaluateQueryRayNotHitButSampleRayHit(
        const AT_NAME::context& ctxt,
        const aten::ray& query_ray,
        const aten::Intersection& isect_sample_ray,
        AT_NAME::FeatureLine::Disc& disc,
        bool is_found_closest_sample_ray_hit,
        real closest_sample_ray_distance,
        const real feature_line_width,
        const real pixel_width)
    {
        // Query ray doesn't hit, but sample ray hits.

        // Evaluate sample ray hit.
        const auto& obj = ctxt.GetObject(isect_sample_ray.objid);

        aten::hitrecord hrec_sample;
        AT_NAME::evaluate_hit_result(hrec_sample, obj, ctxt, query_ray, isect_sample_ray);

        // Compute distance between point which sample ray projects onth query ray and query ray org.
        const auto distance_sample_pos_on_query_ray = AT_NAME::FeatureLine::computeDistanceBetweenProjectedPosOnRayAndRayOrg(
            hrec_sample.p, query_ray);

        // Update closest sample ray.
        if (distance_sample_pos_on_query_ray < closest_sample_ray_distance) {
            // Check if hit point by sample ray is within feature line width.
            const auto is_line_width = AT_NAME::FeatureLine::isInLineWidth(
                feature_line_width,
                query_ray,
                hrec_sample.p,
                disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
                pixel_width);

            if (is_line_width) {
                // If sample ray hits somethng and it's within feature line width,
                // it is forcibly treated as feature line without checking metrics.
                is_found_closest_sample_ray_hit = true;
                closest_sample_ray_distance = distance_sample_pos_on_query_ray;
            }
        }

        return aten::make_tuple<bool, float>(is_found_closest_sample_ray_hit, closest_sample_ray_distance);
    }
}
