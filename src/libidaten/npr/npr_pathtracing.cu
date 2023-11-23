#include "npr/npr_pathtracing.h"

#include "kernel/accelerator.cuh"
#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "renderer/pathtracing/pt_params.h"
#include "renderer/npr/npr_impl.h"

namespace npr_kernel {
    __global__ void GenerateSampleRay(
        idaten::NPRPathTracing::SampleRayInfo* sample_ray_infos,
        idaten::Path paths,
        const aten::ray* __restrict__ rays,
        const int32_t* __restrict__ hitindices,
        int32_t* hitnum,
        real feature_line_width,
        real pixel_width)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        auto& sample_ray_info = sample_ray_infos[idx];
        const auto& ray = rays[idx];

        AT_NAME::npr::GenerateSampleRayAndDiscPerQueryRay<idaten::NPRPathTracing::SampleRayNum>(
            sample_ray_info.descs, sample_ray_info.disc,
            ray, paths.sampler[idx],
            feature_line_width, pixel_width);
    }

    __global__ void shadeSampleRay(
        aten::vec3 line_color,  // TODO
        real feature_line_width,
        real pixel_width,
        idaten::NPRPathTracing::SampleRayInfo* sample_ray_infos,
        int32_t depth,
        const int32_t* __restrict__ hitindices,
        int32_t* hitnum,
        idaten::Path paths,
        const aten::CameraParameter camera,
        const aten::Intersection* __restrict__ isects,
        const aten::ray* __restrict__ rays,
        idaten::context ctxt,
        const aten::ObjectParameter* __restrict__ shapes,
        const aten::MaterialParameter* __restrict__ mtrls,
        const aten::LightParameter* __restrict__ lights,
        const aten::TriangleParameter* __restrict__ prims,
        const aten::mat4* __restrict__ matrices)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        if (paths.attrib[idx].isKill || paths.attrib[idx].isTerminate) {
            paths.attrib[idx].isTerminate = true;
            return;
        }

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        aten::hitrecord hrec_query;

        const auto& isect = isects[idx];

        const auto& query_ray = rays[idx];
        auto* sampler = &paths.sampler[idx];

        const auto& cam_org = camera.origin;

        constexpr auto SampleRayNum = aten::array_size<decltype(idaten::NPRPathTracing::SampleRayInfo::descs)>::size;

        // TODO: These value should be configurable.
        constexpr real albedo_threshold = 0.1f;
        constexpr real normal_threshold = 0.1f;

        auto& sample_ray_info = sample_ray_infos[idx];
        auto& sample_ray_descs = sample_ray_info.descs;
        auto& disc = sample_ray_info.disc;

        // Current closest distance to feature line point.
        auto closest_feature_line_point_distance = std::numeric_limits<real>::max();

        // Whether the feature line point has been found.
        bool is_found_feature_line_point = false;

        real hit_point_distance = 0;

        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));
        AT_NAME::evaluate_hit_result(hrec_query, obj, ctxt, query_ray, isect);

        const auto distance_query_ray_hit = length(hrec_query.p - query_ray.org);

        // disc.centerはquery_ray.orgに一致する.
        // ただし、最初だけは、query_ray.orgはカメラ視点になっているが、
        // accumulated_distanceでカメラとdiscの距離がすでに含まれている.
        hit_point_distance = length(hrec_query.p - disc.center);

        const auto prev_disc = disc;
        disc = AT_NAME::npr::FeatureLine::ComputeDiscAtQueryRayHitPoint(
            hrec_query.p,
            query_ray.dir,
            prev_disc.radius,
            hit_point_distance,
            disc.accumulated_distance);

        for (size_t i = 0; i < SampleRayNum; i++) {
            if (sample_ray_info.descs[i].is_terminated) {
                continue;
            }

            auto sample_ray = AT_NAME::npr::GetSampleRay(
                depth,
                sample_ray_descs[i],
                prev_disc, disc);
            if (sample_ray_descs[i].is_terminated) {
                continue;
            }

            aten::Intersection isect_sample_ray;

            auto is_hit = intersectClosest(&ctxt, sample_ray, &isect_sample_ray);
            if (is_hit) {
                // Query ray hits and then sample ray hits.
                aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = AT_NAME::npr::EvaluateQueryAndSampleRayHit(
                    sample_ray_descs[i],
                    ctxt, cam_org,
                    query_ray, hrec_query, distance_query_ray_hit,
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
                    query_ray, hrec_query, distance_query_ray_hit,
                    sample_ray, disc,
                    is_found_feature_line_point,
                    closest_feature_line_point_distance,
                    feature_line_width, pixel_width);
            }

            const auto mtrl = ctxt.GetMaterial(hrec_query.mtrlid);
            if (!mtrl.attrib.isGlossy) {
                // In non glossy material case, sample ray doesn't bounce anymore.
                // TODO
                // Even if material is glossy, how glossy depends on parameter (e.g. roughness etc).
                // So, I need to consider how to indetify if sample ray bounce is necessary based on material.
                sample_ray_descs[i].is_terminated = true;
            }
        }

        if (is_found_feature_line_point) {
            AT_NAME::npr::ComputeFeatureLineContribution<SampleRayNum>(
                closest_feature_line_point_distance, paths, idx, line_color);
        }

        disc.accumulated_distance += hit_point_distance;
    }

    __global__ void shadeMissSampleRay(
        int32_t width, int32_t height,
        aten::vec3 line_color,  // TODO
        real feature_line_width,
        real pixel_width,
        idaten::NPRPathTracing::SampleRayInfo* sample_ray_infos,
        int32_t depth,
        const int32_t* __restrict__ hitindices,
        int32_t* hitnum,
        idaten::Path paths,
        const aten::CameraParameter camera,
        const aten::Intersection* __restrict__ isects,
        const aten::ray* __restrict__ rays,
        idaten::context ctxt,
        const aten::ObjectParameter* __restrict__ shapes,
        const aten::MaterialParameter* __restrict__ mtrls,
        const aten::LightParameter* __restrict__ lights,
        const aten::TriangleParameter* __restrict__ prims,
        const aten::mat4* __restrict__ matrices)
    {
        auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        if (paths.attrib[idx].isTerminate || paths.attrib[idx].isHit) {
            return;
        }

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        // Query ray doesn't hit anything, but evaluate a possibility that sample ray might hit something.

        const auto& isect = isects[idx];

        const auto& query_ray = rays[idx];
        auto* sampler = &paths.sampler[idx];

        const auto& cam_org = camera.origin;

        constexpr auto SampleRayNum = aten::array_size<decltype(idaten::NPRPathTracing::SampleRayInfo::descs)>::size;

        // TODO: These value should be configurable.
        constexpr real albedo_threshold = 0.1f;
        constexpr real normal_threshold = 0.1f;

        auto& sample_ray_info = sample_ray_infos[idx];
        auto& sample_ray_descs = sample_ray_info.descs;
        auto& disc = sample_ray_info.disc;

        auto closest_feature_line_point_distance = std::numeric_limits<real>::max();
        bool is_found_feature_line_point = false;
        real hit_point_distance = 0;

        // NOTE:
        // In order to compute sample ray, previous disc and next disc are necessary.
        // In first bounce, initial point is camera original.
        // So, previous disc is not necessary.

        AT_NAME::npr::FeatureLine::Disc prev_disc;
        hit_point_distance = CreateNextDiscByDummyQueryRayHitPoint(depth, hit_point_distance, query_ray, prev_disc, disc);

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

            aten::Intersection isect_sample_ray;

            auto is_hit = intersectClosest(&ctxt, sample_ray, &isect_sample_ray);
            if (is_hit) {
                // Query ray doesn't hit, but sample ray hits.
                aten::tie(is_found_feature_line_point, closest_feature_line_point_distance) = AT_NAME::npr::EvaluateQueryRayNotHitButSampleRayHit(
                    ctxt, query_ray,
                    isect_sample_ray,
                    disc,
                    is_found_feature_line_point,
                    closest_feature_line_point_distance,
                    feature_line_width, pixel_width);
            }
            else {
                // Sample ray doesn't hit anything. It means sample ray causes hit miss.
                // So, traversing sample ray is terminated.
                sample_ray_descs[i].is_terminated = true;
                break;
            }
        }

        if (is_found_feature_line_point) {
            AT_NAME::npr::ComputeFeatureLineContribution<SampleRayNum>(
                closest_feature_line_point_distance, paths, idx, line_color);
        }
    }
}

namespace idaten {
    void NPRPathTracing::onShade(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height,
        int32_t sample,
        int32_t bounce, int32_t rrBounce)
    {
        if (is_enable_feature_line_) {
            dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
            dim3 threadPerBlock(64);

            if (sample_ray_infos_.empty()) {
                sample_ray_infos_.resize(width * height);
            }

            auto& hitcount = m_compaction.getCount();

            const auto pixel_width = AT_NAME::camera::computePixelWidthAtDistance(m_cam, 1);

            if (bounce == 0) {
                npr_kernel::GenerateSampleRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
                    sample_ray_infos_.data(),
                    path_host_->paths,
                    m_rays.data(),
                    m_hitidx.data(),
                    hitcount.data(),
                    feature_line_width_,
                    pixel_width);
                checkCudaKernel(GenerateSampleRay);
            }

            npr_kernel::shadeSampleRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
                aten::vec3(0, 1, 0),
                feature_line_width_,
                pixel_width,
                sample_ray_infos_.data(),
                bounce,
                m_hitidx.data(),
                hitcount.data(),
                path_host_->paths,
                m_cam,
                m_isects.data(),
                m_rays.data(),
                ctxt_host_.ctxt,
                ctxt_host_.shapeparam.data(),
                ctxt_host_.mtrlparam.data(),
                ctxt_host_.lightparam.data(),
                ctxt_host_.primparams.data(),
                ctxt_host_.mtxparams.data());
            checkCudaKernel(shadeSampleRay);
        }

        PathTracing::onShade(
            outputSurf,
            width, height,
            sample,
            bounce, rrBounce);
    }

    void NPRPathTracing::missShade(
        int32_t width, int32_t height,
        int32_t bounce)
    {
        if (is_enable_feature_line_) {
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid(
                (width + block.x - 1) / block.x,
                (height + block.y - 1) / block.y);

            if (sample_ray_infos_.empty()) {
                sample_ray_infos_.resize(width * height);
            }

            auto& hitcount = m_compaction.getCount();

            const auto pixel_width = AT_NAME::camera::computePixelWidthAtDistance(m_cam, 1);

            // Sample ray hit miss never happen at 1st bounce.
            npr_kernel::shadeMissSampleRay << <grid, block, 0, m_stream >> > (
                width, height,
                aten::vec3(0, 1, 0),
                feature_line_width_,
                pixel_width,
                sample_ray_infos_.data(),
                bounce,
                m_hitidx.data(),
                hitcount.data(),
                path_host_->paths,
                m_cam,
                m_isects.data(),
                m_rays.data(),
                ctxt_host_.ctxt,
                ctxt_host_.shapeparam.data(),
                ctxt_host_.mtrlparam.data(),
                ctxt_host_.lightparam.data(),
                ctxt_host_.primparams.data(),
                ctxt_host_.mtxparams.data());
            checkCudaKernel(shadeMissSampleRay);
        }

        PathTracing::missShade(width, height, bounce);
    }
}
