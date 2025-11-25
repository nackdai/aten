#include "npr/npr_pathtracing.h"

#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "accelerator/threaded_bvh_traverser.h"
#include "renderer/pathtracing/pt_params.h"
#include "renderer/npr/npr_impl.h"

namespace npr_kernel {
    __global__ void GenerateSampleRay(
        AT_NAME::npr::FeatureLine::SampleRayInfo<idaten::NPRPathTracing::SampleRayNum>* sample_ray_infos,
        idaten::context ctxt,
        idaten::Path paths,
        const aten::ray* __restrict__ rays,
        const int32_t* __restrict__ hitindices,
        int32_t* hitnum,
        float pixel_width)
    {
        int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        auto& sample_ray_info = sample_ray_infos[idx];
        const auto& ray = rays[idx];

        const float feature_line_width{ ctxt.scene_rendering_config.feature_line.line_width };

        AT_NAME::npr::GenerateSampleRayAndDiscPerQueryRay<idaten::NPRPathTracing::SampleRayNum>(
            sample_ray_info.descs, sample_ray_info.disc,
            ray, paths.sampler[idx],
            feature_line_width, pixel_width);
    }

    __global__ void shadeSampleRay(
        float pixel_width,
        AT_NAME::npr::FeatureLine::SampleRayInfo<8>* sample_ray_infos,
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

        if (paths.attrib[idx].is_terminated) {
            return;
        }

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        const auto& query_ray = rays[idx];
        const auto& isect = isects[idx];

        AT_NAME::npr::ShadeSampleRay(
            pixel_width,
            idx, depth,
            ctxt,
            camera, query_ray, isect,
            paths, sample_ray_infos
        );
    }

    __global__ void shadeMissSampleRay(
        int32_t width, int32_t height,
        float pixel_width,
        AT_NAME::npr::FeatureLine::SampleRayInfo<idaten::NPRPathTracing::SampleRayNum>* sample_ray_infos,
        int32_t depth,
        const int32_t* __restrict__ hitindices,
        int32_t* hitnum,
        idaten::Path paths,
        const aten::CameraParameter camera,
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

        if (paths.attrib[idx].is_terminated || paths.attrib[idx].isHit) {
            return;
        }

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        // Query ray doesn't hit anything, but evaluate a possibility that sample ray might hit something.
        const auto& query_ray = rays[idx];

        AT_NAME::npr::ShadeMissSampleRay(
            pixel_width,
            idx, depth,
            ctxt,
            query_ray,
            paths,
            sample_ray_infos
        );
    }
}

namespace idaten {
    void NPRPathTracing::onShade(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height,
        int32_t sample,
        int32_t bounce, int32_t rrBounce, int32_t max_depth)
    {
        if (ctxt_host_->ctxt.scene_rendering_config.feature_line.enabled) {
            dim3 blockPerGrid(((width * height) + 64 - 1) / 64);
            dim3 threadPerBlock(64);

            if (sample_ray_infos_.empty()) {
                sample_ray_infos_.resize(width * height);
            }

            auto& hitcount = m_compaction.getCount();

            const auto pixel_width = AT_NAME::Camera::ComputePixelWidthAtDistance(m_cam, 1);

            if (bounce == 0) {
                npr_kernel::GenerateSampleRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
                    sample_ray_infos_.data(),
                    ctxt_host_->ctxt,
                    path_host_->paths,
                    m_rays.data(),
                    m_hitidx.data(),
                    hitcount.data(),
                    pixel_width);
                checkCudaKernel(GenerateSampleRay);
            }

            npr_kernel::shadeSampleRay << <blockPerGrid, threadPerBlock, 0, m_stream >> > (
                pixel_width,
                sample_ray_infos_.data(),
                bounce,
                m_hitidx.data(),
                hitcount.data(),
                path_host_->paths,
                m_cam,
                m_isects.data(),
                m_rays.data(),
                ctxt_host_->ctxt,
                ctxt_host_->shapeparam.data(),
                ctxt_host_->mtrlparam.data(),
                ctxt_host_->lightparam.data(),
                ctxt_host_->primparams.data(),
                ctxt_host_->mtxparams.data());
            checkCudaKernel(shadeSampleRay);
        }

        PathTracing::onShade(
            outputSurf,
            width, height,
            sample,
            bounce, rrBounce, max_depth);
    }

    void NPRPathTracing::missShade(
        int32_t width, int32_t height,
        int32_t bounce)
    {
        if (ctxt_host_->ctxt.scene_rendering_config.feature_line.enabled) {
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid(
                (width + block.x - 1) / block.x,
                (height + block.y - 1) / block.y);

            if (sample_ray_infos_.empty()) {
                sample_ray_infos_.resize(width * height);
            }

            auto& hitcount = m_compaction.getCount();

            const auto pixel_width = AT_NAME::Camera::ComputePixelWidthAtDistance(m_cam, 1);

            // Sample ray hit miss never happen at 1st bounce.
            npr_kernel::shadeMissSampleRay << <grid, block, 0, m_stream >> > (
                width, height,
                pixel_width,
                sample_ray_infos_.data(),
                bounce,
                m_hitidx.data(),
                hitcount.data(),
                path_host_->paths,
                m_cam,
                m_rays.data(),
                ctxt_host_->ctxt,
                ctxt_host_->shapeparam.data(),
                ctxt_host_->mtrlparam.data(),
                ctxt_host_->lightparam.data(),
                ctxt_host_->primparams.data(),
                ctxt_host_->mtxparams.data());
            checkCudaKernel(shadeMissSampleRay);
        }

        PathTracing::missShade(width, height, bounce);
    }
}
