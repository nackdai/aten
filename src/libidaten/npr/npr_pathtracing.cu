#include "npr/npr_pathtracing.h"

#include "kernel/accelerator.cuh"
#include "kernel/context.cuh"
#include "kernel/intersect.cuh"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

namespace npr_pt {
    __global__ void generateSampleRay(
        idaten::NPRPathTracing::SampleRayInfo* sample_ray_infos,
        idaten::Path* paths,
        const aten::ray* __restrict__ rays,
        const int* __restrict__ hitindices,
        int* hitnum,
        real FeatureLineWidth,
        real pixel_width
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        auto& sample_ray_info = sample_ray_infos[idx];
        const auto& ray = rays[idx];

        sample_ray_info.disc = AT_NAME::FeatureLine::generateDisc(ray, FeatureLineWidth, pixel_width);
        for (size_t i = 0; i < AT_NAME::NPRPathTracing::SampleRayNum; i++) {
            const auto sample_ray = AT_NAME::FeatureLine::generateSampleRay(
                sample_ray_info.descs[i], paths->sampler[idx], ray, sample_ray_info.disc);
            AT_NAME::FeatureLine::storeRayToDesc(sample_ray_info.descs[i], sample_ray);
            sample_ray_info.descs[i].is_terminated = false;
        }

        sample_ray_info.disc.accumulated_distance = 1;
    }

    __global__ void shadeSampleRay(
        real FeatureLineWidth,
        real pixel_width,
        idaten::NPRPathTracing::SampleRayInfo* sample_ray_infos,
        int depth,
        const int* __restrict__ hitindices,
        int* hitnum,
        idaten::Path* paths,
        const aten::CameraParameter* __restrict__ camera,
        const aten::Intersection* __restrict__ isects,
        const aten::ray* __restrict__ rays,
        const aten::GeomParameter* __restrict__ shapes, int geomnum,
        const aten::MaterialParameter* __restrict__ mtrls,
        const aten::LightParameter* __restrict__ lights, int lightnum,
        cudaTextureObject_t* nodes,
        const aten::PrimitiveParamter* __restrict__ prims,
        cudaTextureObject_t vtxPos,
        cudaTextureObject_t vtxNml,
        const aten::mat4* __restrict__ matrices,
        cudaTextureObject_t* textures)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        if (paths->attrib[idx].isKill || paths->attrib[idx].isTerminate) {
            paths->attrib[idx].isTerminate = true;
            return;
        }

        idaten::Context ctxt;
        {
            ctxt.geomnum = geomnum;
            ctxt.shapes = shapes;
            ctxt.mtrls = mtrls;
            ctxt.lightnum = lightnum;
            ctxt.lights = lights;
            ctxt.nodes = nodes;
            ctxt.prims = prims;
            ctxt.vtxPos = vtxPos;
            ctxt.vtxNml = vtxNml;
            ctxt.matrices = matrices;
            ctxt.textures = textures;
        }

        const auto& query_ray = rays[idx];

        aten::hitrecord rec;

        const auto& isect = isects[idx];

        auto obj = &ctxt.shapes[isect.objid];
        evalHitResult(&ctxt, obj, query_ray, &rec, &isect);

        constexpr real ThresholdAlbedo = 0.1f;
        constexpr real ThresholdNormal = 0.1f;

        auto closest_sample_ray_distance = std::numeric_limits<real>::max();
        int32_t closest_sample_ray_idx = -1;
        real hit_point_distance = 0;

        auto& sample_ray_info = sample_ray_infos[idx];

        const auto& cam_org = camera->origin;

        auto disc = sample_ray_info.disc;

        const auto distance_query_ray_hit = length(rec.p - query_ray.org);

        // disc.centerはquery_ray.orgに一致する.
        // ただし、最初だけは、query_ray.orgはカメラ視点になっているが、
        // accumulated_distanceでカメラとdiscの距離がすでに含まれている.
        hit_point_distance = length(rec.p - disc.center);

        const auto prev_disc = disc;
        disc = AT_NAME::FeatureLine::computeNextDisc(
            rec.p,
            query_ray.dir,
            prev_disc.radius,
            hit_point_distance,
            disc.accumulated_distance);

        for (size_t i = 0; i < idaten::NPRPathTracing::SampleRayNum; i++) {
            if (sample_ray_info.descs[i].is_terminated) {
                continue;
            }

            if (depth > 0) {
                // Generate next sample ray.
                const auto res_next_sample_ray = AT_NAME::FeatureLine::computeNextSampleRay(
                    sample_ray_info.descs[i],
                    prev_disc, disc);
                const auto is_sample_ray_valid = aten::get<0>(res_next_sample_ray);
                if (!is_sample_ray_valid) {
                    sample_ray_info.descs[i].is_terminated = true;
                    continue;
                }
                const auto sample_ray = aten::get<1>(res_next_sample_ray);
                AT_NAME::FeatureLine::storeRayToDesc(sample_ray_info.descs[i], sample_ray);
            }

            const auto sample_ray = AT_NAME::FeatureLine::getRayFromDesc(sample_ray_info.descs[i]);

            aten::Intersection isect_sample_ray;
            aten::hitrecord hrec_sample;

            auto is_hit = intersectClosest(&ctxt, sample_ray, &isect_sample_ray);
#if 1
            if (is_hit) {
                auto obj = &ctxt.shapes[isect_sample_ray.objid];
                evalHitResult(&ctxt, obj, sample_ray, &hrec_sample, &isect_sample_ray);

                // If sample ray hit with the different mesh from query ray one, this sample ray won't bounce in next loop.
                sample_ray_info.descs[i].is_terminated = isect_sample_ray.meshid != isect.meshid;
                sample_ray_info.descs[i].prev_ray_hit_pos = hrec_sample.p;
                sample_ray_info.descs[i].prev_ray_hit_nml = hrec_sample.normal;

                const auto distance_sample_pos_on_query_ray = AT_NAME::FeatureLine::computeDistanceBetweenProjectedPosOnRayAndRayOrg(
                    hrec_sample.p, query_ray);

                const auto is_line_width = AT_NAME::FeatureLine::isInLineWidth(
                    FeatureLineWidth,
                    query_ray,
                    hrec_sample.p,
                    disc.accumulated_distance - 1, // NOTE: -1 is for initial camera distance.
                    pixel_width);
                if (is_line_width) {
                    const auto& query_mtrl = ctxt.mtrls[rec.mtrlid];
                    const auto& sample_mtrl = ctxt.mtrls[hrec_sample.mtrlid];
                    const auto query_albedo = AT_NAME::sampleTexture(query_mtrl.albedoMap, rec.u, rec.v, query_mtrl.baseColor);
                    const auto sample_albedo = AT_NAME::sampleTexture(sample_mtrl.albedoMap, hrec_sample.u, hrec_sample.v, sample_mtrl.baseColor);
                    const auto query_depth = length(rec.p - cam_org);
                    const auto sample_depth = length(hrec_sample.p - cam_org);

                    const auto is_feature_line = AT_NAME::FeatureLine::evaluateMetrics(
                        query_ray.org,
                        rec, hrec_sample,
                        query_albedo, sample_albedo,
                        query_depth, sample_depth,
                        ThresholdAlbedo, ThresholdNormal,
                        2);

                    if (is_feature_line) {
                        if (distance_sample_pos_on_query_ray < closest_sample_ray_distance
                            && distance_sample_pos_on_query_ray < distance_query_ray_hit)
                        {
                            // Deal with sample hit point as FeatureLine.
                            closest_sample_ray_idx = i;
                            closest_sample_ray_distance = distance_sample_pos_on_query_ray;
                        }
                        else if (distance_query_ray_hit < closest_sample_ray_distance) {
                            // Deal with query hit point as FeatureLine.
                            closest_sample_ray_idx = idaten::NPRPathTracing::SampleRayNum;
                            closest_sample_ray_distance = distance_query_ray_hit;
                        }
                    }
                }
            }
#endif
            const auto& mtrl = ctxt.mtrls[rec.mtrlid];
            if (!mtrl.attrib.isGlossy) {
                // In non glossy material case, sample ray doesn't bounce anymore.
                // TODO
                // Even if material is glossy, how glossy depends on parameter (e.g. roughness etc).
                // So, I need to consider how to indetify if sample ray bounce is necessary based on material.
                sample_ray_info.descs[i].is_terminated = true;
            }
        }

        if (closest_sample_ray_idx >= 0) {
            paths->contrib[idx].contrib = make_float3(0, 1, 0);
            paths->attrib[idx].isKill = true;
            paths->attrib[idx].isTerminate = true;
        }

        disc.accumulated_distance += hit_point_distance;
        sample_ray_info.disc = disc;
    }
}

namespace idaten {
    void NPRPathTracing::onShade(
        cudaSurfaceObject_t outputSurf,
        int width, int height,
        int sample,
        int bounce, int rrBounce,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        dim3 blockPerGrid(((m_tileDomain.w * m_tileDomain.h) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        if (sample_ray_infos_.empty()) {
            sample_ray_infos_.init(width * height);
        }

        auto& hitcount = m_compaction.getCount();

        const auto pixel_width = AT_NAME::camera::computePixelWidthAtDistance(m_camParam, 1);

        if (bounce == 0) {
            npr_pt::generateSampleRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
                sample_ray_infos_.ptr(),
                m_paths.ptr(),
                m_rays.ptr(),
                m_hitidx.ptr(),
                hitcount.ptr(),
                feature_line_width_,
                pixel_width);
            checkCudaKernel(generateSampleRay);
        }

        npr_pt::shadeSampleRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            feature_line_width_,
            pixel_width,
            sample_ray_infos_.ptr(),
            bounce,
            m_hitidx.ptr(),
            hitcount.ptr(),
            m_paths.ptr(),
            m_cam.ptr(),
            m_isects.ptr(),
            m_rays.ptr(),
            m_shapeparam.ptr(), m_shapeparam.num(),
            m_mtrlparam.ptr(),
            m_lightparam.ptr(), m_lightparam.num(),
            m_nodetex.ptr(),
            m_primparams.ptr(),
            texVtxPos, texVtxNml,
            m_mtxparams.ptr(),
            m_tex.ptr());
        checkCudaKernel(shadeSampleRay);

        PathTracing::onShade(
            outputSurf,
            width, height,
            sample,
            bounce, rrBounce,
            texVtxPos, texVtxNml);
    }
}