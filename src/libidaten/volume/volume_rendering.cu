#include "volume/volume_rendering.h"

#include "aten4idaten.h"
#include "kernel/accelerator.cuh"
#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_common.h"
#include "kernel/StreamCompaction.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/volume/volume_pathtracing_impl.h"

namespace vpt
{
    __global__ void ShadeVolumePT(
        int32_t width, int32_t height,
        idaten::context ctxt,
        const aten::ObjectParameter* __restrict__ shapes,
        const aten::MaterialParameter* __restrict__ mtrls,
        const aten::LightParameter* __restrict__ lights,
        const aten::TriangleParameter* __restrict__ prims,
        const aten::mat4* __restrict__ matrices,
        idaten::Path paths,
        const int32_t* __restrict__ hitindices,
        int32_t* hitnum,
        const aten::Intersection* __restrict__ isects,
        aten::ray* rays,
        int32_t sample, int32_t frame,
        int32_t depth_for_rr, int32_t max_depth,
        uint32_t* random,
        AT_NAME::ShadowRay* shadow_rays)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        if (paths.attrib[idx].isTerminate) {
            paths.attrib[idx].will_update_depth = false;
            return;
        }

        const auto bounce = paths.throughput[idx].depth_count;

        const auto russianProb = AT_NAME::ComputeRussianProbability(
            bounce, depth_for_rr,
            paths.attrib[idx], paths.throughput[idx],
            paths.sampler[idx]);
        if (paths.attrib[idx].isTerminate) {
            paths.attrib[idx].will_update_depth = false;
            return;
        }

        paths.throughput[idx].throughput /= russianProb;

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        __shared__ aten::MaterialParameter shMtrls[64];

        const auto& isect = isects[idx];

#if IDATEN_SAMPLER == IDATEN_SAMPLER_SOBOL
        auto scramble = random[idx] * 0x1fe3434f;
        paths.sampler[idx].init(frame + sample, 4 + bounce * 300, scramble);
#elif IDATEN_SAMPLER == IDATEN_SAMPLER_CMJ
        auto rnd = random[idx];
        auto scramble = rnd * 0x1fe3434f
            * (((frame + sample) + 331 * rnd) / (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM));
        paths.sampler[idx].init(
            (frame + sample) % (aten::CMJ::CMJ_DIM * aten::CMJ::CMJ_DIM),
            4 + bounce * 300,
            scramble);
#endif

        auto ray = rays[idx];
        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

        aten::hitrecord rec;
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < 0.0f;

        // 交差位置の法線.
        // 物体からのレイの入出を考慮.
        aten::vec3 orienting_normal = rec.normal;

        AT_NAME::FillMaterial(
            shMtrls[threadIdx.x],
            ctxt,
            rec.mtrlid,
            rec.isVoxel);

        auto& shadow_ray = shadow_rays[idx];
        shadow_ray.isActive = false;

        bool is_scattered = false;

        if (AT_NAME::HasMedium(paths.throughput[idx].mediums)) {
            aten::ray next_ray;

            aten::tie(is_scattered, next_ray) = AT_NAME::SampleMedium(
                paths.throughput[idx],
                paths.sampler[idx],
                ctxt,
                ray, isect);

            if (is_scattered) {
                shadow_ray.isActive = true;

                shadow_ray.rayorg = next_ray.org;
                shadow_ray.raydir = next_ray.dir;
            }

            ray = next_ray;
        }

        bool is_reflected_or_refracted = false;

        if (is_scattered) {
            rays[idx] = ray;
        }
        else {
            // Implicit conection to light.
            auto is_hit_implicit_light = AT_NAME::HitImplicitLight(
                ctxt, isect.objid,
                isBackfacing,
                bounce,
                paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
                ray,
                rec.p, orienting_normal,
                rec.area,
                shMtrls[threadIdx.x]);
            if (is_hit_implicit_light) {
                paths.attrib[idx].will_update_depth = false;
                return;
            }

            const auto curr_ray = ray;

            if (shMtrls[threadIdx.x].is_medium && !AT_NAME::IsSubsurface(shMtrls[threadIdx.x])) {
                auto ray_base_nml = dot(ray.dir, orienting_normal) > 0
                    ? orienting_normal
                    : -orienting_normal;
                rays[idx] = aten::ray(rec.p, ray.dir, ray_base_nml);
            }
            else {
                auto albedo = AT_NAME::sampleTexture(shMtrls[threadIdx.x].albedoMap, rec.u, rec.v, shMtrls[threadIdx.x].baseColor, bounce);

                // Apply normal map.
                int32_t normalMap = shMtrls[threadIdx.x].normalMap;
                auto pre_sampled_r = AT_NAME::material::applyNormal(
                    &shMtrls[threadIdx.x],
                    normalMap,
                    orienting_normal, orienting_normal,
                    rec.u, rec.v,
                    ray.dir,
                    &paths.sampler[idx]);

                if (!shMtrls[threadIdx.x].attrib.isTranslucent && isBackfacing) {
                    orienting_normal = -orienting_normal;
                }

                // NEE
                aten::LightSampleResult light_sample;
                float light_select_prob = 0.0F;
                int target_light_idx = -1;
                aten::tie(light_sample, light_select_prob, target_light_idx) = AT_NAME::SampleLight(
                    ctxt, shMtrls[threadIdx.x], bounce,
                    paths.sampler[idx],
                    rec.p, orienting_normal);

                if (target_light_idx >= 0) {
                    float transmittance = 1.0F;
                    float is_visilbe_to_light = false;

                    aten::tie(is_visilbe_to_light, transmittance) = AT_NAME::TraverseShadowRay(
                        ctxt, paths.sampler[idx],
                        light_sample,
                        rec.p, orienting_normal,
                        paths.throughput[idx].mediums);

                    if (is_visilbe_to_light) {
                        auto radiance = AT_NAME::ComputeRadianceNEE(
                            ray, orienting_normal,
                            shMtrls[threadIdx.x], pre_sampled_r, rec.u, rec.v,
                            light_select_prob, light_sample);
                        if (radiance.has_value()) {
                            const auto& r = radiance.value();
                            const auto contrib = paths.throughput[idx].throughput * transmittance* r* static_cast<aten::vec3>(albedo);
                            AT_NAME::_detail::AddVec3(paths.contrib[idx].contrib, contrib);
                        }
                    }
                }

                AT_NAME::MaterialSampling sampling;

                AT_NAME::material::sampleMaterial(
                    &sampling,
                    &shMtrls[threadIdx.x],
                    orienting_normal,
                    ray.dir,
                    rec.normal,
                    &paths.sampler[idx], pre_sampled_r,
                    rec.u, rec.v);

                AT_NAME::PrepareForNextBounce(
                    idx,
                    rec, isBackfacing, russianProb,
                    orienting_normal,
                    shMtrls[threadIdx.x], sampling,
                    albedo,
                    paths,
                    rays);
            }

            AT_NAME::UpdateMedium(
                curr_ray, rays[idx].dir, orienting_normal,
                shMtrls[threadIdx.x], paths.throughput[idx].mediums);
        }

        paths.attrib[idx].will_update_depth = is_scattered || is_reflected_or_refracted;
    }

    __global__ void TraverseShadowRay(
        int32_t max_depth,
        idaten::context ctxt,
        const aten::ObjectParameter* __restrict__ shapes,
        const aten::MaterialParameter* __restrict__ mtrls,
        const aten::LightParameter* __restrict__ lights,
        const aten::TriangleParameter* __restrict__ prims,
        const aten::mat4* __restrict__ matrices,
        idaten::Path paths,
        const int32_t* __restrict__ hitindices,
        int32_t* hitnum,
        const aten::Intersection* __restrict__ isects,
        const AT_NAME::ShadowRay* __restrict__ shadow_rays
    )
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        const auto& shadow_ray = shadow_rays[idx];

        if (shadow_ray.isActive) {
            ctxt.shapes = shapes;
            ctxt.mtrls = mtrls;
            ctxt.lights = lights;
            ctxt.prims = prims;
            ctxt.matrices = matrices;

            const auto bounce = paths.throughput[idx].depth_count;
            const auto& isect = isects[idx];

            aten::ray next_ray(shadow_ray.rayorg, shadow_ray.raydir);

            // NOTE:
            // In volume, there is no normal.
            // On the other hand, there is a possibility that we need to compute the tangent cooridnate.
            // Therefore, next ray direction is treated as the normal alternatively.
            const auto& nml = next_ray.dir;

            const auto& mtrl = ctxt.GetMaterial(isect.mtrlid);

            aten::LightSampleResult light_sample;
            float light_select_prob = 0.0F;
            int target_light_idx = -1;
            aten::tie(light_sample, light_select_prob, target_light_idx) = AT_NAME::SampleLight(
                ctxt, mtrl, bounce,
                paths.sampler[idx],
                next_ray.org, nml,
                false);

            if (target_light_idx >= 0) {
                float transmittance = 1.0F;
                float is_visilbe_to_light = false;

                aten::tie(is_visilbe_to_light, transmittance) = AT_NAME::TraverseShadowRay(
                    ctxt, paths.sampler[idx],
                    light_sample,
                    next_ray.org, nml,
                    paths.throughput[idx].mediums);

                if (is_visilbe_to_light) {
                    const auto& medium = AT_NAME::GetCurrentMedium(ctxt, paths.throughput[idx].mediums);
                    const auto phase_f = AT_NAME::HenyeyGreensteinPhaseFunction::Evaluate(
                        medium.phase_function_g,
                        -next_ray.dir, light_sample.dir);

                    const auto dist2 = aten::sqr(light_sample.dist_to_light);

                    // Geometry term.
                    const auto G = 1.0F / dist2;

                    const auto Ls = transmittance * phase_f * G * light_sample.light_color / light_sample.pdf / light_select_prob;
                    const auto contrib = paths.throughput[idx].throughput * Ls;

                    AT_NAME::_detail::AddVec3(paths.contrib[idx].contrib, contrib);
                }
            }
        }

        // Update depth count for the next bounce.
        if (paths.attrib[idx].will_update_depth) {
            paths.throughput[idx].depth_count += 1;
        }
        paths.attrib[idx].will_update_depth = false;

        if (paths.throughput[idx].depth_count > max_depth) {
            paths.attrib[idx].isTerminate = true;
        }
    }
}

namespace idaten {
    void VolumeRendering::onShade(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height,
        int32_t sample,
        int32_t bounce, int32_t rrBounce, int32_t max_depth)
    {
        dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
        dim3 threadPerBlock(64);

        auto& hitcount = m_compaction.getCount();

        vpt::ShadeVolumePT << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            width, height,
            ctxt_host_.ctxt,
            ctxt_host_.shapeparam.data(),
            ctxt_host_.mtrlparam.data(),
            ctxt_host_.lightparam.data(),
            ctxt_host_.primparams.data(),
            ctxt_host_.mtxparams.data(),
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            m_isects.data(),
            m_rays.data(),
            sample,
            m_frame,
            rrBounce, max_depth,
            m_random.data(),
            m_shadowRays.data());

        checkCudaKernel(ShadeVolumePT);

        vpt::TraverseShadowRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
            max_depth,
            ctxt_host_.ctxt,
            ctxt_host_.shapeparam.data(),
            ctxt_host_.mtrlparam.data(),
            ctxt_host_.lightparam.data(),
            ctxt_host_.primparams.data(),
            ctxt_host_.mtxparams.data(),
            path_host_->paths,
            m_hitidx.data(), hitcount.data(),
            m_isects.data(),
            m_shadowRays.data());

        checkCudaKernel(TraverseShadowRay);
    }
}
