#include "svgf/svgf.h"

#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/StreamCompaction.h"
#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"
#include "renderer/pathtracing/pt_params.h"

__global__ void fillAOV(
    cudaSurfaceObject_t dst,
    AT_NAME::SVGFAovMode mode,
    int32_t width, int32_t height,
    const float4* __restrict__ aovNormalDepth,
    const float4* __restrict__ aovTexclrMeshid,
    cudaSurfaceObject_t motionDetphBuffer,
    const aten::CameraParameter camera,
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

    const aten::vec3 colors[] = {
        aten::vec3(255,   0,   0),
        aten::vec3(  0, 255,   0),
        aten::vec3(  0,   0, 255),
        aten::vec3(255, 255,   0),
        aten::vec3(255,   0, 255),
        aten::vec3(  0, 255, 255),
        aten::vec3(128, 128, 128),
        aten::vec3( 86,  99, 143),
        aten::vec3( 71, 234, 126),
        aten::vec3(124,  83,  53),
    };

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

    aten::Intersection isect;
    bool isHit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
        isect, ctxt, camsample.r,
        AT_MATH_EPSILON, AT_MATH_INF);

    float4 clr = make_float4(1);

    if (mode == AT_NAME::SVGFAovMode::Normal) {
        auto n = aovNormalDepth[idx] * 0.5f + 0.5f;
        clr = make_float4(n.x, n.y, n.z, 1);
    }
    else if (mode == AT_NAME::SVGFAovMode::Depth) {
        // TODO
    }
    else if (mode == AT_NAME::SVGFAovMode::TexColor) {
        clr = aovTexclrMeshid[idx];
    }
    else if (mode == AT_NAME::SVGFAovMode::WireFrame) {
        bool isHitEdge = (isect.a < 1e-2) || (isect.b < 1e-2) || (1 - isect.a - isect.b < 1e-2);
        clr = isHitEdge ? make_float4(0) : make_float4(1);
    }
    else if (mode == AT_NAME::SVGFAovMode::BaryCentric) {
        auto c = 1 - isect.a - isect.b;
        clr = make_float4(isect.a, isect.b, c, 1);
    }
    else if (mode == AT_NAME::SVGFAovMode::Motion) {
        float4 data;
        surf2Dread(&data, motionDetphBuffer, ix * sizeof(float4), iy);

        // TODO
        float motionX = data.x;
        float motionY = data.y;

        clr = make_float4(motionX, motionY, 0, 1);
    }
    else if (mode == AT_NAME::SVGFAovMode::ObjId) {
#if 0
        int32_t objid = isect.meshid;
#else
        int32_t objid = isect.mtrlid;
#endif
        if (objid >= 0) {
            objid %= AT_COUNTOF(colors);
            auto c = colors[objid];

            clr = make_float4(c.x, c.y, c.z, 1);
            clr /= 255.0f;
        }
        else {
            clr = make_float4(0, 0, 0, 1);
        }
    }

    surf2Dwrite(
        clr,
        dst,
        ix * sizeof(float4), iy,
        cudaBoundaryModeTrap);
}

__global__ void pickPixel(
    idaten::SVGFPathTracing::PickedInfo* dst,
    int32_t ix, int32_t iy,
    int32_t width, int32_t height,
    const aten::CameraParameter camera,
    const idaten::Path paths,
    const float4* __restrict__ aovNormalDepth,
    const float4* __restrict__ aovTexclrMeshid,
    idaten::context ctxt,
    const aten::ObjectParameter* __restrict__ shapes,
    const aten::MaterialParameter* __restrict__ mtrls,
    const aten::LightParameter* __restrict__ lights,
    const aten::TriangleParameter* __restrict__ prims,
    const aten::mat4* __restrict__ matrices)
{
    ctxt.shapes = shapes;
    ctxt.mtrls = mtrls;
    ctxt.lights = lights;
    ctxt.prims = prims;
    ctxt.matrices = matrices;

    iy = height - 1 - iy;

    float s = (ix + 0.5f) / (float)(camera.width);
    float t = (iy + 0.5f) / (float)(camera.height);

    AT_NAME::CameraSampleResult camsample;
    AT_NAME::PinholeCamera::sample(&camsample, &camera, s, t);

    aten::Intersection isect;
    bool isHit = aten::BvhTraverser::Traverse<aten::IntersectType::Closest>(
        isect, ctxt, camsample.r,
        AT_MATH_EPSILON, AT_MATH_INF);

    if (isHit) {
        const auto idx = getIdx(ix, iy, width);

        auto normalDepth = aovNormalDepth[idx];
        auto texclrMeshid = aovTexclrMeshid[idx];

        dst->ix = ix;
        dst->iy = iy;
        dst->color = aten::vec3(paths.contrib[idx].contrib.x, paths.contrib[idx].contrib.y, paths.contrib[idx].contrib.z);
        dst->normal = aten::vec3(normalDepth.x, normalDepth.y, normalDepth.z);
        dst->depth = normalDepth.w;
        dst->meshid = (int32_t)texclrMeshid.w;
        dst->triid = isect.triangle_id;
        dst->mtrlid = isect.mtrlid;
    }
    else {
        dst->ix = -1;
        dst->iy = -1;
    }
}

namespace idaten
{
    void SVGFPathTracing::OnDisplayAOV(
        cudaSurfaceObject_t outputSurf,
        int32_t width, int32_t height)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        auto& curaov = params_.GetCurrAovBuffer();

        CudaGLResourceMapper<decltype(params_.motion_depth_buffer)> rscmap(params_.motion_depth_buffer);
        auto gbuffer = params_.motion_depth_buffer.bind();

        fillAOV << <grid, block >> > (
            outputSurf,
            m_aovMode,
            width, height,
            curaov.get<AT_NAME::SVGFAovBufferType::NormalDepth>().data(),
            curaov.get<AT_NAME::SVGFAovBufferType::AlbedoMeshId>().data(),
            gbuffer,
            m_cam,
            ctxt_host_->ctxt,
            ctxt_host_->shapeparam.data(),
            ctxt_host_->mtrlparam.data(),
            ctxt_host_->lightparam.data(),
            ctxt_host_->primparams.data(),
            ctxt_host_->mtxparams.data());
    }

    void SVGFPathTracing::pick(
        int32_t ix, int32_t iy,
        int32_t width, int32_t height)
    {
        if (m_willPicklPixel) {
            m_pick.resize(1);

            auto& curaov = params_.GetCurrAovBuffer();

            pickPixel << <1, 1 >> > (
                m_pick.data(),
                m_pickedInfo.ix, m_pickedInfo.iy,
                width, height,
                m_cam,
                path_host_->paths,
                curaov.get<AT_NAME::SVGFAovBufferType::NormalDepth>().data(),
                curaov.get<AT_NAME::SVGFAovBufferType::AlbedoMeshId>().data(),
                ctxt_host_->ctxt,
                ctxt_host_->shapeparam.data(),
                ctxt_host_->mtrlparam.data(),
                ctxt_host_->lightparam.data(),
                ctxt_host_->primparams.data(),
                ctxt_host_->mtxparams.data());

            m_pick.readFromDeviceToHostByNum(&m_pickedInfo);

            m_willPicklPixel = false;
        }
    }
}
