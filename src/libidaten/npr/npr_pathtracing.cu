#include "npr/npr_pathtracing.h"

#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "accelerator/threaded_bvh_traverser.h"
#include "renderer/ao/aorenderer_impl.h"
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

        constexpr auto SampleRayNum = std::remove_pointer_t<decltype(sample_ray_infos)>::size;
        AT_NAME::npr::GenerateSampleRayAndDiscPerQueryRay<SampleRayNum>(
            sample_ray_info.descs, sample_ray_info.disc,
            ray, paths.sampler[idx],
            feature_line_width, pixel_width);
    }

    __global__ void shadeSampleRay(
        float pixel_width,
        AT_NAME::npr::FeatureLine::SampleRayInfo<idaten::NPRPathTracing::SampleRayNum>* sample_ray_infos,
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

        if (paths.attrib[idx].attr.is_terminated) {
            return;
        }

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        const auto& query_ray = rays[idx];
        const auto& isect = isects[idx];

        constexpr auto SampleRayNum = std::remove_pointer_t<decltype(sample_ray_infos)>::size;
        AT_NAME::npr::ShadeSampleRay<SampleRayNum>(
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

        if (paths.attrib[idx].attr.is_terminated || paths.attrib[idx].attr.isHit) {
            return;
        }

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        // Query ray doesn't hit anything, but evaluate a possibility that sample ray might hit something.
        const auto& query_ray = rays[idx];

        constexpr auto SampleRayNum = std::remove_pointer_t<decltype(sample_ray_infos)>::size;
        AT_NAME::npr::ShadeMissSampleRay<SampleRayNum>(
            pixel_width,
            idx, depth,
            ctxt,
            query_ray,
            paths,
            sample_ray_infos
        );
    }

    __global__ void ShadeAO(
        int32_t width, int32_t height,
        int32_t ao_num_rays, float ao_radius,
        float* dst,
        idaten::Path paths,
        int32_t* hitindices,
        int32_t* hitnum,
        const aten::Intersection* __restrict__ isects,
        aten::ray* rays,
        idaten::context ctxt,
        const aten::ObjectParameter* __restrict__ shapes,
        const aten::MaterialParameter* __restrict__ mtrls,
        const aten::LightParameter* __restrict__ lights,
        const aten::TriangleParameter* __restrict__ prims,
        const aten::mat4* __restrict__ matrices)
    {
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        auto idx = getIdx(ix, iy, width);

        if (idx >= *hitnum) {
            return;
        }

        idx = hitindices[idx];

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        const auto& ray = rays[idx];
        const auto& isect = isects[idx];

        const auto ao_color = AT_NAME::ao::ShandeByAO(
            ao_num_rays, ao_radius,
            paths.sampler[idx], ctxt, ray, isect);

        dst[idx] = ao_color;
    }

    __global__ void ApplyBilateralFilter(
        int32_t width, int32_t height,
        cudaSurfaceObject_t dst,
        const float* __restrict__ src,
        const aten::Intersection* __restrict__ isects
    )
    {
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        const auto idx = getIdx(ix, iy, width);

        const auto filtered_color = AT_NAME::ao::ApplyBilateralFilter<float, float, true>(
            ix, iy,
            width, height,
            2.0F, 2.0F,
            src, isects
        );
        surf2Dwrite(filtered_color, dst, ix * sizeof(float), iy, cudaBoundaryModeTrap);
    }
}

namespace idaten {
    void NPRPathTracing::PreShade(
        int32_t width, int32_t height,
        int32_t bounce,
        cudaSurfaceObject_t outputSurf)
    {
        PathTracing::PreShade(width, height, bounce, outputSurf);

        if (bounce == 0 && hatching_shadow_ == HatchingShadow::AOBase) {
            dim3 thread_per_block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 block_per_grid(
                (width + thread_per_block.x - 1) / thread_per_block.x,
                (height + thread_per_block.y - 1) / thread_per_block.y);

            ao_result_buffer_.resize(width * height);
            ctxt_host_->screen_space_texture.Init(width, height);
            ctxt_host_->BindToDeviceContext();

            npr_kernel::ShadeAO << <block_per_grid, thread_per_block, 0, m_stream >> > (
                width, height,
                getNumRays(), getRadius(),
                ao_result_buffer_.data(),
                path_host_->paths,
                m_hitidx.data(),
                m_compaction.getCount().data(),
                m_isects.data(),
                m_rays.data(),
                ctxt_host_->ctxt,
                ctxt_host_->shapeparam.data(),
                ctxt_host_->mtrlparam.data(),
                ctxt_host_->lightparam.data(),
                ctxt_host_->primparams.data(),
                ctxt_host_->mtxparams.data());

            checkCudaKernel(ShadeAO);

            npr_kernel::ApplyBilateralFilter << <block_per_grid, thread_per_block, 0, m_stream >> > (
                width, height,
                ctxt_host_->screen_space_texture.GetSurfaceObject(),
                ao_result_buffer_.data(),
                m_isects.data());

            checkCudaKernel(ApplyBilateralFilter);
        }
    }

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
