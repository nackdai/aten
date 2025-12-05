#include "kernel/renderer.h"
#include "kernel/device_scene_context.cuh"
#include "kernel/pt_common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "accelerator/threaded_bvh_traverser.h"

namespace idaten::kernel::debug
{
    __global__ void textureViewer(
        uint32_t texIdx,
        int32_t width, int32_t height,
        cudaTextureObject_t* textures,
        cudaSurfaceObject_t outSurface)
    {
        const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
        const auto iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) {
            return;
        }

        float u = ix / (float)width;
        float v = iy / (float)height;

        AT_NAME::context ctxt;
        ctxt.textures = textures;

        auto texclr = AT_NAME::sampleTexture(ctxt, texIdx, u, v, aten::vec4(1, 0, 0, 1));

        surf2Dwrite(
            make_float4(texclr.r, texclr.g, texclr.b, texclr.a),
            outSurface,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
    }

    __global__ void RenderAOV(
        cudaSurfaceObject_t dst,
        idaten::Renderer::AOV aov,
        int32_t width, int32_t height,
        aten::CameraParameter camera,
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

        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.matrices = matrices;

        float s = (ix + 0.5f) / (float)(width);
        float t = (iy + 0.5f) / (float)(height);

        AT_NAME::CameraSampleResult camsample;
        AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

        const auto ray = camsample.r;

        aten::Intersection isect;
        bool is_hit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
            isect, ctxt, ray,
            AT_MATH_EPSILON, AT_MATH_INF);

        float4 clr = make_float4(1);

        if (is_hit) {
            const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

            aten::hitrecord rec;
            AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

            aten::MaterialParameter mtrl;
            AT_NAME::FillMaterial(
                mtrl,
                ctxt,
                rec.mtrlid,
                rec.isVoxel);

            if (aov == idaten::Renderer::AOV::Albedo) {
                auto albedo = AT_NAME::sampleTexture(
                    ctxt, mtrl.albedoMap,
                    rec.u, rec.v,
                    mtrl.baseColor);
                clr = make_float4(albedo.x, albedo.y, albedo.z, 1.0F);
            }
            else if (aov == idaten::Renderer::AOV::Normal) {
                auto normal = rec.normal * 0.5f + 0.5f;
                clr = make_float4(normal.x, normal.y, normal.z, 1);
            }
            else if (aov == idaten::Renderer::AOV::WireFrame) {
                bool isHitEdge = (isect.hit.tri.a < 1e-2)
                    || (isect.hit.tri.b < 1e-2)
                    || (1 - isect.hit.tri.a - isect.hit.tri.b < 1e-2);
                clr = isHitEdge ? make_float4(0) : make_float4(1);
            }
            else if (aov == idaten::Renderer::AOV::BaryCentric) {
                auto c = 1 - isect.hit.tri.a - isect.hit.tri.b;
                clr = make_float4(isect.hit.tri.a, isect.hit.tri.b, c, 1);
            }
        }
        else {
            auto emit = AT_NAME::Background::SampleFromRay(ray.dir, ctxt.scene_rendering_config.bg, ctxt);
            clr = make_float4(emit.x, emit.y, emit.z, 1.0F);
        }

        surf2Dwrite(
            clr,
            dst,
            ix * sizeof(float4), iy,
            cudaBoundaryModeTrap);
    }
}

namespace idaten
{
    void Renderer::viewTextures(
        uint32_t idx,
        int32_t screenWidth, int32_t screenHeight)
    {
        m_glimg.map();
        auto outputSurf = m_glimg.bind();

        if (!ctxt_host_->texRsc.empty()) {
            std::vector<cudaTextureObject_t> tmp;
            for (auto& tex_rsc : ctxt_host_->texRsc) {
                auto cudaTex = tex_rsc.bind();
                tmp.push_back(cudaTex);
            }
            ctxt_host_->tex.writeFromHostToDeviceByNum(&tmp[0], (uint32_t)tmp.size());
        }

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (screenWidth + block.x - 1) / block.x,
            (screenHeight + block.y - 1) / block.y);

        idaten::kernel::debug::textureViewer << <grid, block >> > (
            idx,
            screenWidth, screenHeight,
            ctxt_host_->tex.data(),
            outputSurf);

        for (auto& tex_rsc : ctxt_host_->texRsc) {
            tex_rsc.unbind();
        }

        m_glimg.unbind();
        m_glimg.unmap();
    }

    void Renderer::ViewAOV(
        AOV aov,
        int32_t screenWidth, int32_t screenHeight)
    {
        m_glimg.map();
        auto outputSurf = m_glimg.bind();

        if (!ctxt_host_->texRsc.empty()) {
            std::vector<cudaTextureObject_t> tmp;
            for (auto& tex_rsc : ctxt_host_->texRsc) {
                auto cudaTex = tex_rsc.bind();
                tmp.push_back(cudaTex);
            }
            ctxt_host_->tex.writeFromHostToDeviceByNum(&tmp[0], (uint32_t)tmp.size());
        }

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (screenWidth + block.x - 1) / block.x,
            (screenHeight + block.y - 1) / block.y);

        idaten::kernel::debug::RenderAOV << <grid, block >> > (
            outputSurf,
            aov,
            screenWidth, screenHeight,
            m_cam,
            ctxt_host_->ctxt,
            ctxt_host_->shapeparam.data(),
            ctxt_host_->mtrlparam.data(),
            ctxt_host_->lightparam.data(),
            ctxt_host_->primparams.data(),
            ctxt_host_->mtxparams.data());

        for (auto& tex_rsc : ctxt_host_->texRsc) {
            tex_rsc.unbind();
        }

        m_glimg.unbind();
        m_glimg.unmap();
    }
}
