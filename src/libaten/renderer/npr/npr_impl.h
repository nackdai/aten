#pragma once

#include "renderer/npr/feature_line.h"
#include "renderer/npr/feature_line_config.h"
#include "renderer/pathtracing/pathtracing_impl.h"
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

namespace AT_NAME {
namespace npr {
    /**
     * @brief Generate sample rays per a query ray.
     *
     * @tparam SampleRayNum Nubmer of sample rays per a query ray.
     * @param[out] sample_ray_descs Descriptions of sample ray per a query ray.
     * @param[out] disc Disc at query ray hit point.
     * @param[in,out] sampler Sampler for radom number.
     * @param[in] feature_line_width Feature line width in screen.
     * @param[in] pixel_width Pixel width at distance 1 from camera.
     */
    template <size_t SampleRayNum>
    inline AT_DEVICE_API void GenerateSampleRayAndDiscPerQueryRay(
        std::array<AT_NAME::npr::FeatureLine::SampleRayDesc, SampleRayNum>& sample_ray_descs,
        AT_NAME::npr::FeatureLine::Disc& disc,
        const aten::ray& query_ray,
        aten::sampler& sampler,
        const float feature_line_width,
        const float pixel_width)
    {
        disc = AT_NAME::npr::FeatureLine::GenerateDisc(query_ray, feature_line_width, pixel_width);

#pragma unroll
        for (size_t i = 0; i < SampleRayNum; i++) {
            auto& desc = sample_ray_descs[i];
            const auto sample_ray = AT_NAME::npr::FeatureLine::GenerateSampleRay(
                desc, sampler, query_ray, disc);
            AT_NAME::npr::FeatureLine::StoreRayInSampleRayDesc(desc, sample_ray);
            desc.is_terminated = false;
        }

        disc.accumulated_distance = 1;
    }

    /**
     * @brief Compute the contribution by the feature line.
     *
     * @tparam SampleRayNum Nubmer of sample rays per a query ray.
     * @param[in] closest_feature_line_point_distance TODO
     * @param[in,out] paths Information of paths.
     * @param[in] idx Index to the pixel.
     * @param[in] line_color Feature line color.
     */
    template <size_t SampleRayNum>
    inline AT_DEVICE_API void ComputeFeatureLineContribution(
        float closest_feature_line_point_distance,
        AT_NAME::Path& paths,
        int32_t idx,
        const aten::vec3& line_color)
    {
        constexpr auto PdfSampleRay = float(1) / SampleRayNum;
        auto pdf_feature_line = PdfSampleRay * (closest_feature_line_point_distance * closest_feature_line_point_distance);

        const auto pdfb = paths.throughput[idx].pdfb;
        const auto weight = _detail::ComputeBalanceHeuristic(pdfb, pdf_feature_line);
        const auto contrib = paths.throughput[idx].throughput * weight * line_color;

        aten::CopyVec(paths.contrib[idx].contrib, contrib);

        // Line is treated as light. So, query path need to be killed and termnated not to bounce anymore.
        paths.attrib[idx].attr.is_terminated = true;
    }

    /**
     * @brief Get a sample ray.
     *
     * @param[in] depth Depth of bounce.
     * @param[in,out] sample_ray_desc Sample ray description.
     * @param[in] prev_disc Disc at the previous hit point by the query ray.
     * @param[in] prev_disc Disc at the current hit point by the query ray.
     */
    inline AT_DEVICE_API aten::ray GetSampleRay(
        int32_t depth,
        AT_NAME::npr::FeatureLine::SampleRayDesc& sample_ray_desc,
        const AT_NAME::npr::FeatureLine::Disc& prev_disc,
        const AT_NAME::npr::FeatureLine::Disc& curr_disc)
    {
        auto sample_ray = AT_NAME::npr::FeatureLine::ExtractRayFromSampleRayDesc(sample_ray_desc);
        if (depth > 0) {
            // Generate next sample ray.
            const auto res_next_sample_ray = AT_NAME::npr::FeatureLine::ComputeNextSampleRay(
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

    /**
     * @brief Evaluate the case which both query ray and sample ray hit.
     *
     * @param[in,out] sample_ray_desc Sample ray description.
     * @param[in] ctxt Scene context.
     * @param[in] cam_org Camera's origin position.
     * @param[in] query_ray Query ray.
     * @param[in] sample_ray Sample ray.
     * @param[in] hrec_query Hit record by the query ray.
     * @param[in] distance_query_ray_hit Distance between query ray hit point and query ray's origin.
     * @param[in] isect_sample_ray Intersection data by the sample ray.
     * @param[in] disc Disc at query ray hit point.
     * @param[in] is_found_feature_line_point Whether the feature line point has been found.
     * @param[in] closest_feature_line_point_distance Current closest distance to feature line point.
     * @param[in] feature_line_width Feature line width in screen.
     * @param[in] pixel_width Pixel width at distance 1 from camera.
     * @param[in] albedo_threshold Threshold to evaluate albedo.
     * @param[in] normal_threshold Threshold to evaluate normal.
     * @return Tuple for the updated is_found_feature_line_point and closest_feature_line_point_distance.
     */
    inline AT_DEVICE_API aten::tuple<bool, float> EvaluateQueryAndSampleRayHit(
        AT_NAME::npr::FeatureLine::SampleRayDesc& sample_ray_desc,
        const AT_NAME::context& ctxt,
        const aten::vec3& cam_org,
        const aten::ray& query_ray,
        const aten::ray& sample_ray,
        const aten::hitrecord& hrec_query,
        const float distance_query_ray_hit,
        const aten::Intersection& isect_sample_ray,
        const AT_NAME::npr::FeatureLine::Disc& disc,
        bool is_found_feature_line_point,
        float closest_feature_line_point_distance,
        const aten::FeatureLineConfig& config,
        const aten::FeatureLineMtrlConfig& mtrl_config,
        const float pixel_width
    )
    {
        // Query ray hit and then sample ray's hit.

        const auto& obj = ctxt.GetObject(isect_sample_ray.objid);

        aten::hitrecord hrec_sample;
        AT_NAME::evaluate_hit_result(hrec_sample, obj, ctxt, sample_ray, isect_sample_ray);

        // If sample ray hit with the different mesh from query ray one, this sample ray won't bounce in next loop.
        sample_ray_desc.is_terminated = hrec_sample.meshid != hrec_query.meshid;
        sample_ray_desc.prev_ray_hit_pos = hrec_sample.p;
        sample_ray_desc.prev_ray_hit_nml = hrec_sample.normal;

        // Distance between projected sample ray hit point on the query ray and query ray's origin.
        const auto distance_sample_pos_on_query_ray = AT_NAME::npr::FeatureLine::ComputeDistanceBetweenProjectedPositionOnRayAndRayOrigin(
            hrec_sample.p, query_ray);

        // Check if the query ray is in feature line width.
        const auto is_line_width = AT_NAME::npr::FeatureLine::IsInLineWidth(
            config.line_width,
            query_ray,
            hrec_sample.p,
            disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
            pixel_width);

        if (is_line_width) {
            // Get merrics.
            aten::MaterialParameter mtrl_tmp;
            AT_NAME::FillMaterial(mtrl_tmp, ctxt, hrec_query.mtrlid, hrec_query.isVoxel);
            const auto query_albedo = AT_NAME::sampleTexture(ctxt, mtrl_tmp.albedoMap, hrec_query.u, hrec_query.v, mtrl_tmp.baseColor);

            AT_NAME::FillMaterial(mtrl_tmp, ctxt, hrec_sample.mtrlid, hrec_sample.isVoxel);
            const auto sample_albedo = AT_NAME::sampleTexture(ctxt, mtrl_tmp.albedoMap, hrec_sample.u, hrec_sample.v, mtrl_tmp.baseColor);

            const auto query_depth = length(hrec_query.p - cam_org);
            const auto sample_depth = length(hrec_sample.p - cam_org);

            // Check if the hit point can be treated as the feature line.
            const auto is_feature_line = AT_NAME::npr::FeatureLine::EvaluateMetrics(
                query_ray.org,
                hrec_query, hrec_sample,
                query_albedo, sample_albedo,
                config, ctxt.GetMaterial(hrec_query.mtrlid).feature_line,
                query_depth, sample_depth,
                2);
            if (is_feature_line) {
                if (distance_sample_pos_on_query_ray < closest_feature_line_point_distance
                    && distance_sample_pos_on_query_ray < distance_query_ray_hit)
                {
                    // The projected sample ray hit point is the closest at this moment,
                    // and it is closer than query ray hit point.
                    // Deal with "sample" ray hit point as FeatureLine.
                    is_found_feature_line_point = true;
                    closest_feature_line_point_distance = distance_sample_pos_on_query_ray;
                }
                else if (distance_query_ray_hit < closest_feature_line_point_distance) {
                    // The query hit point is closer than the current closest feature line point.
                    // Deal with "query" ray hit point as FeatureLine.
                    is_found_feature_line_point = true;
                    closest_feature_line_point_distance = distance_query_ray_hit;
                }
            }
        }
        return aten::make_tuple<bool, float>(is_found_feature_line_point, closest_feature_line_point_distance);
    }

    /**
     * @brief Evaluate the case which the query ray hits but the sample ray doesn't hit.
     *
     * @param[in,out] sample_ray_desc Sample ray description.
     * @param[in] query_ray Query ray.
     * @param[in] hrec_query Hit record by the query ray.
     * @param[in] distance_query_ray_hit Distance between query ray hit point and query ray's origin.
     * @param[in] sample_ray Sample ray.
     * @param[in] disc Disc at query ray hit point.
     * @param[in] is_found_feature_line_point Whether the feature line point has been found.
     * @param[in] closest_feature_line_point_distance Current closest distance to feature line point.
     * @param[in] feature_line_width Feature line width in screen.
     * @param[in] pixel_width Pixel width at distance 1 from camera.
     * @return Tuple for the updated is_found_feature_line_point and closest_feature_line_point_distance.
     */
    inline AT_DEVICE_API aten::tuple<bool, float> EvaluateQueryRayHitButSampleRayNotHit(
        AT_NAME::npr::FeatureLine::SampleRayDesc& sample_ray_desc,
        const aten::ray& query_ray,
        const aten::hitrecord& hrec_query,
        const float distance_query_ray_hit,
        const aten::ray& sample_ray,
        const AT_NAME::npr::FeatureLine::Disc& disc,
        bool is_found_feature_line_point,
        float closest_feature_line_point_distance,
        const float feature_line_width,
        const float pixel_width)
    {
        // Query ray hits but sample ray doesn't hit anything.

        // Compute plane which query ray hits.
        const auto query_hit_plane = AT_NAME::npr::FeatureLine::ComputePlane(hrec_query);

        // Even if sample ray doesn't hit anything, compute point which sample ray hit onto plane which query ray hits.
        const auto res_sample_ray_dummy_hit = AT_NAME::npr::FeatureLine::ComputeRayHitPositionOnPlane(
            query_hit_plane, sample_ray);

        const auto is_hit_sample_ray_dummy_plane = aten::get<0>(res_sample_ray_dummy_hit);
        if (is_hit_sample_ray_dummy_plane) {
            // Get point which sample ray hit onto plane which query ray hits.
            const auto sample_ray_dummy_hit_pos = aten::get<1>(res_sample_ray_dummy_hit);

            const auto distance_sample_pos_on_query_ray = AT_NAME::npr::FeatureLine::ComputeDistanceBetweenProjectedPositionOnRayAndRayOrigin(
                sample_ray_dummy_hit_pos, query_ray);

            // If point which sample ray hits is within feature line width,
            // that point is treated as feature line forcibly without checking metrics.
            const auto is_line_width = AT_NAME::npr::FeatureLine::IsInLineWidth(
                feature_line_width,
                query_ray,
                sample_ray_dummy_hit_pos,
                disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
                pixel_width);
            if (is_line_width) {
                if (distance_sample_pos_on_query_ray < closest_feature_line_point_distance
                    && distance_sample_pos_on_query_ray < distance_query_ray_hit)
                {
                    // The projected sample ray hit point is the closest at this moment,
                    // and it is closer than query ray hit point.
                    // Deal with "sample" ray hit point as FeatureLine.
                    is_found_feature_line_point = true;
                    closest_feature_line_point_distance = distance_sample_pos_on_query_ray;
                }
                else if (distance_query_ray_hit < closest_feature_line_point_distance) {
                    // The query hit point is closer than the current closest feature line point.
                    // Deal with "query" ray hit point as FeatureLine.
                    is_found_feature_line_point = true;
                    closest_feature_line_point_distance = distance_query_ray_hit;
                }
            }
        }

        // Sample ray doesn't hit anything. It means sample ray causes hit miss.
        // So, traversing sample ray is terminated.
        sample_ray_desc.is_terminated = true;

        return aten::make_tuple<bool, float>(is_found_feature_line_point, closest_feature_line_point_distance);
    }

    /**
     * @brief Create next disc forcibly by dummy query ray hit point.
     *
     * @param[in] Count of bounce.
     * @param[in] hit_point_distance Previous distance between hit point and query ray origin.
     * @param[in] query_ray Query ray.
     * @param[out] prev_disc Disc description to keep the current disc as the previous one.
     * @param[in] disc Current disc.
     * @return Distance between the dummy hit point and query ray origin.
     */
    inline AT_DEVICE_API float CreateNextDiscByDummyQueryRayHitPoint(
        int32_t depth,
        float hit_point_distance,
        const aten::ray& query_ray,
        FeatureLine::Disc& prev_disc,
        FeatureLine::Disc& disc)
    {
        if (depth > 0) {
            // Query ray doesn't hit anything, so there is no specific hit point.
            // So, in order to compute next sample ray, make dummy hit point.
            const auto dummy_query_hit_pos = query_ray.org + float(100) * query_ray.dir;

            // Current disc center = query ray origin. So, we can use whichever.
            hit_point_distance = length(dummy_query_hit_pos - disc.center);

            prev_disc = disc;
            disc = AT_NAME::npr::FeatureLine::ComputeDiscAtQueryRayHitPoint(
                dummy_query_hit_pos,
                query_ray.dir,
                prev_disc.radius,
                hit_point_distance,
                disc.accumulated_distance);
        }

        return hit_point_distance;
    }

    /**
     * @brief Evaluate the case which the query ray doesn't hit but the sample ray hits.
     *
     * @param[in] ctxt Scene context.
     * @param[in] query_ray Query ray.
     * @param[in] isect_sample_ray Intersection data by the sample ray.
     * @param[in] disc Disc at query ray hit point.
     * @param[in] is_found_feature_line_point Whether the feature line point has been found.
     * @param[in] closest_feature_line_point_distance Current closest distance to feature line point.
     * @param[in] feature_line_width Feature line width in screen.
     * @param[in] pixel_width Pixel width at distance 1 from camera.
     * @return Tuple for the updated is_found_feature_line_point and closest_feature_line_point_distance.
     */
    inline AT_DEVICE_API aten::tuple<bool, float> EvaluateQueryRayNotHitButSampleRayHit(
        const AT_NAME::context& ctxt,
        const aten::ray& query_ray,
        const aten::Intersection& isect_sample_ray,
        const AT_NAME::npr::FeatureLine::Disc& disc,
        bool is_found_feature_line_point,
        float closest_feature_line_point_distance,
        const float feature_line_width,
        const float pixel_width)
    {
        // Query ray doesn't hit, but sample ray hits.

        // Evaluate sample ray hit.
        const auto& obj = ctxt.GetObject(isect_sample_ray.objid);

        aten::hitrecord hrec_sample;
        AT_NAME::evaluate_hit_result(hrec_sample, obj, ctxt, query_ray, isect_sample_ray);

        // Compute distance between point which sample ray projects on the query ray and query ray origin.
        const auto distance_sample_pos_on_query_ray = AT_NAME::npr::FeatureLine::ComputeDistanceBetweenProjectedPositionOnRayAndRayOrigin(
            hrec_sample.p, query_ray);

        // Update closest sample ray.
        if (distance_sample_pos_on_query_ray < closest_feature_line_point_distance) {
            // Check if hit point by sample ray is within feature line width.
            const auto is_line_width = AT_NAME::npr::FeatureLine::IsInLineWidth(
                feature_line_width,
                query_ray,
                hrec_sample.p,
                disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
                pixel_width);

            if (is_line_width) {
                // If sample ray hits somethng and it's within feature line width,
                // it is forcibly treated as feature line without checking metrics.
                is_found_feature_line_point = true;
                closest_feature_line_point_distance = distance_sample_pos_on_query_ray;
            }
        }

        return aten::make_tuple<bool, float>(is_found_feature_line_point, closest_feature_line_point_distance);
    }

#if 1
    template <int32_t SampleRayNum>
    inline AT_DEVICE_API void ShadeSampleRay(
        const float pixel_width,
        const int32_t idx,
        const int32_t depth,
        const AT_NAME::context& ctxt,
        const aten::CameraParameter& camera,
        const aten::ray& query_ray,
        const aten::Intersection& isect,
        AT_NAME::Path& paths,
        FeatureLine::SampleRayInfo<SampleRayNum>* sample_ray_infos
    )
    {
        const auto& query_ray_hit_mtrl = ctxt.GetMaterial(isect.mtrlid);
        if (!query_ray_hit_mtrl.feature_line.enable) {
            return;
        }

        const aten::vec3 line_color{ ctxt.scene_rendering_config.feature_line.line_color };
        const float feature_line_width{ ctxt.scene_rendering_config.feature_line.line_width };

        const auto& cam_org = camera.origin;

        auto& sample_ray_info = sample_ray_infos[idx];
        auto& sample_ray_descs = sample_ray_info.descs;
        auto& disc = sample_ray_info.disc;

        // Current closest distance to feature line point.
        auto closest_feature_line_point_distance = std::numeric_limits<float>::max();

        // Whether the feature line point has been found.
        bool is_found_feature_line_point = false;

        float hit_point_distance = 0;

        aten::hitrecord hrec_query;

        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));
        AT_NAME::evaluate_hit_result(hrec_query, obj, ctxt, query_ray, isect);

        const auto distance_query_ray_hit = length(hrec_query.p - query_ray.org);

        // disc.centerはquery_ray.orgに一致する.
        // ただし、最初だけは、query_ray.orgはカメラ視点になっているが、
        // accumulated_distanceでカメラとdiscの距離がすでに含まれている.
        hit_point_distance = length(hrec_query.p - disc.center);

        const auto prev_disc = disc;

        // Update to the new computed disc.
        disc = AT_NAME::npr::FeatureLine::ComputeDiscAtQueryRayHitPoint(
            hrec_query.p,
            query_ray.dir,
            prev_disc.radius,
            hit_point_distance,
            disc.accumulated_distance);

        aten::Intersection isect_sample_ray;

#pragma unroll
        for (size_t i = 0; i < SampleRayNum; i++) {
            auto& desc = sample_ray_descs[i];
            if (desc.is_terminated) {
                continue;
            }

            auto sample_ray = GetSampleRay(
                depth,
                desc,
                prev_disc, disc);
            if (desc.is_terminated) {
                continue;
            }

            auto is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                isect_sample_ray, ctxt, sample_ray,
                AT_MATH_EPSILON, AT_MATH_INF);

            if (is_hit) {
                const auto& sample_ray_hit_mtrl = ctxt.GetMaterial(isect_sample_ray.mtrlid);
                if (sample_ray_hit_mtrl.feature_line.enable) {
                    // Query ray hits and then sample ray hits.
                    aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = EvaluateQueryAndSampleRayHit(
                        desc,
                        ctxt, cam_org,
                        query_ray, sample_ray,
                        hrec_query, distance_query_ray_hit,
                        isect_sample_ray,
                        disc,
                        is_found_feature_line_point,
                        closest_feature_line_point_distance,
                        ctxt.scene_rendering_config.feature_line,
                        sample_ray_hit_mtrl.feature_line,
                        pixel_width);
                }
                else {
                    desc.is_terminated = true;
                }
            }
            else {
                // Query ray hits but sample ray doesn't hit anything.
                aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = EvaluateQueryRayHitButSampleRayNotHit(
                    desc,
                    query_ray, hrec_query, distance_query_ray_hit,
                    sample_ray, disc,
                    is_found_feature_line_point,
                    closest_feature_line_point_distance,
                    ctxt.scene_rendering_config.feature_line.line_width, pixel_width);
            }

            const auto mtrl = ctxt.GetMaterial(hrec_query.mtrlid);
            if (!mtrl.attrib.is_glossy) {
                // In non glossy material case, sample ray doesn't bounce anymore.
                // TODO
                // Even if material is glossy, how glossy depends on parameter (e.g. roughness etc).
                // So, I need to consider how to indetify if sample ray bounce is necessary based on material.
                desc.is_terminated = true;
            }
        }

        if (is_found_feature_line_point) {
            ComputeFeatureLineContribution<SampleRayNum>(
                closest_feature_line_point_distance, paths, idx, line_color);
        }

        disc.accumulated_distance += hit_point_distance;
    }

    template <int32_t SampleRayNum>
    inline AT_DEVICE_API void ShadeMissSampleRay(
        const float pixel_width,
        const int32_t idx,
        const int32_t depth,
        const AT_NAME::context& ctxt,
        const aten::ray& query_ray,
        AT_NAME::Path& paths,
        FeatureLine::SampleRayInfo<SampleRayNum>* sample_ray_infos
    )
    {
        const aten::vec3 line_color{ ctxt.scene_rendering_config.feature_line.line_color };
        const float feature_line_width{ ctxt.scene_rendering_config.feature_line.line_width };

        auto& sample_ray_info = sample_ray_infos[idx];
        auto& sample_ray_descs = sample_ray_info.descs;
        auto& disc = sample_ray_info.disc;

        auto closest_feature_line_point_distance = std::numeric_limits<float>::max();
        bool is_found_feature_line_point = false;
        float hit_point_distance = 0;

        // Query ray doesn't hit anything, but evaluate a possibility that sample ray might hit something.

        // NOTE:
        // In order to compute sample ray, previous disc and next disc are necessary.
        // In first bounce, initial point is camera original.
        // So, previous disc is not necessary.

        // Query ray doesn't hit to anything, and we can't create the next disc at query ray hit point.
        // But, the disc is necessary. So, the next disc is created forcibly from the dummy query ray hit point.

        AT_NAME::npr::FeatureLine::Disc prev_disc;
        hit_point_distance = CreateNextDiscByDummyQueryRayHitPoint(depth, hit_point_distance, query_ray, prev_disc, disc);

        aten::Intersection isect_sample_ray;

#pragma unroll
        for (size_t i = 0; i < SampleRayNum; i++) {
            auto& desc = sample_ray_descs[i];
            if (!desc.is_terminated) {
                auto sample_ray = GetSampleRay(
                    depth,
                    desc,
                    prev_disc, disc);
                if (!desc.is_terminated) {
                    auto is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
                        isect_sample_ray, ctxt, sample_ray,
                        AT_MATH_EPSILON, AT_MATH_INF);

                    if (is_hit) {
                        const auto& sample_ray_hit_mtrl = ctxt.GetMaterial(isect_sample_ray.mtrlid);
                        if (sample_ray_hit_mtrl.feature_line.enable) {
                            // Query ray doesn't hit, but sample ray hits.
                            aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = EvaluateQueryRayNotHitButSampleRayHit(
                                ctxt, query_ray,
                                isect_sample_ray,
                                disc,
                                is_found_feature_line_point,
                                closest_feature_line_point_distance,
                                feature_line_width, pixel_width);
                        }
                        else {
                            desc.is_terminated = true;
                        }
                    }
                    else {
                        // Sample ray doesn't hit anything. It means sample ray causes hit miss.
                        // So, traversing sample ray is terminated.
                        desc.is_terminated = true;
                    }
                }
            }
#ifdef __CUDACC__
            const auto warp_all_done = __all_sync(__activemask(), desc.is_terminated);

            if (warp_all_done) {
                break;
            }
#endif
        }

        if (is_found_feature_line_point) {
            ComputeFeatureLineContribution<SampleRayNum>(
                closest_feature_line_point_distance, paths, idx, line_color);
        }
    }
#endif
}
}
