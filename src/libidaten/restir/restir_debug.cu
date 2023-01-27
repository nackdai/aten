#include "restir/restir.h"

#include "kernel/accelerator.cuh"
#include "kernel/context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/light.cuh"
#include "kernel/material.cuh"
#include "kernel/pt_common.h"
#include "kernel/StreamCompaction.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void fillAOV(
    idaten::TileDomain tileDomain,
    cudaSurfaceObject_t dst,
    idaten::ReSTIRPathTracing::AOVMode mode,
    int width, int height,
    const float4* __restrict__ aovNormalDepth,
    const float4* __restrict__ aovTexclrMeshid,
    cudaSurfaceObject_t motionDetphBuffer,
    const aten::CameraParameter* __restrict__ camera,
    const aten::GeomParameter* __restrict__ shapes, int geomnum,
    cudaTextureObject_t* nodes,
    const aten::PrimitiveParamter* __restrict__ prims,
    cudaTextureObject_t vtxPos,
    const aten::mat4* __restrict__ matrices)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
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

    ix += tileDomain.x;
    iy += tileDomain.y;

    const auto idx = getIdx(ix, iy, width);

    float s = (ix + 0.5f) / (float)(width);
    float t = (iy + 0.5f) / (float)(height);

    AT_NAME::CameraSampleResult camsample;
    AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

    idaten::Context ctxt;
    {
        ctxt.geomnum = geomnum;
        ctxt.shapes = shapes;
        ctxt.nodes = nodes;
        ctxt.prims = prims;
        ctxt.vtxPos = vtxPos;
        ctxt.matrices = matrices;
    }

    aten::Intersection isect;
    bool isHit = intersectClosest(&ctxt, camsample.r, &isect);

    float4 clr = make_float4(1);

    if (mode == idaten::ReSTIRPathTracing::AOVMode::Normal) {
        auto n = aovNormalDepth[idx] * 0.5f + 0.5f;
        clr = make_float4(n.x, n.y, n.z, 1);
    }
    else if (mode == idaten::ReSTIRPathTracing::AOVMode::Depth) {
        // TODO
    }
    else if (mode == idaten::ReSTIRPathTracing::AOVMode::TexColor) {
        clr = aovTexclrMeshid[idx];
    }
    else if (mode == idaten::ReSTIRPathTracing::AOVMode::WireFrame) {
        bool isHitEdge = (isect.a < 1e-2) || (isect.b < 1e-2) || (1 - isect.a - isect.b < 1e-2);
        clr = isHitEdge ? make_float4(0) : make_float4(1);
    }
    else if (mode == idaten::ReSTIRPathTracing::AOVMode::BaryCentric) {
        auto c = 1 - isect.a - isect.b;
        clr = make_float4(isect.a, isect.b, c, 1);
    }
    else if (mode == idaten::ReSTIRPathTracing::AOVMode::Motion) {
        float4 data;
        surf2Dread(&data, motionDetphBuffer, ix * sizeof(float4), iy);

        // TODO
        float motionX = data.x;
        float motionY = data.y;

        clr = make_float4(motionX, motionY, 0, 1);
    }
    else if (mode == idaten::ReSTIRPathTracing::AOVMode::ObjId) {
#if 0
        int objid = isect.meshid;
#else
        int objid = isect.mtrlid;
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
    idaten::ReSTIRPathTracing::PickedInfo* dst,
    int ix, int iy,
    int width, int height,
    const aten::CameraParameter* __restrict__ camera,
    const idaten::Path* __restrict__ paths,
    const float4* __restrict__ aovNormalDepth,
    const float4* __restrict__ aovTexclrMeshid,
    const aten::GeomParameter* __restrict__ shapes, int geomnum,
    cudaTextureObject_t* nodes,
    const aten::PrimitiveParamter* __restrict__ prims,
    cudaTextureObject_t vtxPos,
    aten::mat4* matrices)
{
    iy = height - 1 - iy;

    float s = (ix + 0.5f) / (float)(camera->width);
    float t = (iy + 0.5f) / (float)(camera->height);

    AT_NAME::CameraSampleResult camsample;
    AT_NAME::PinholeCamera::sample(&camsample, camera, s, t);

    idaten::Context ctxt;
    {
        ctxt.geomnum = geomnum;
        ctxt.shapes = shapes;
        ctxt.nodes = nodes;
        ctxt.prims = prims;
        ctxt.vtxPos = vtxPos;
        ctxt.matrices = matrices;
    }

    aten::Intersection isect;
    bool isHit = intersectClosest(&ctxt, camsample.r, &isect);

    if (isHit) {
        const auto idx = getIdx(ix, iy, width);

        auto normalDepth = aovNormalDepth[idx];
        auto texclrMeshid = aovTexclrMeshid[idx];

        dst->ix = ix;
        dst->iy = iy;
        dst->color = aten::vec3(paths->contrib[idx].contrib.x, paths->contrib[idx].contrib.y, paths->contrib[idx].contrib.z);
        dst->normal = aten::vec3(normalDepth.x, normalDepth.y, normalDepth.z);
        dst->depth = normalDepth.w;
        dst->meshid = (int)texclrMeshid.w;
        dst->triid = isect.primid;
        dst->mtrlid = isect.mtrlid;
    }
    else {
        dst->ix = -1;
        dst->iy = -1;
    }
}

namespace idaten
{
    void ReSTIRPathTracing::onDisplayAOV(
        cudaSurfaceObject_t outputSurf,
        int width, int height,
        cudaTextureObject_t texVtxPos)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        CudaGLResourceMapper<decltype(m_motionDepthBuffer)> rscmap(m_motionDepthBuffer);
        auto gbuffer = m_motionDepthBuffer.bind();

        fillAOV << <grid, block >> > (
            m_tileDomain,
            outputSurf,
            m_aovMode,
            width, height,
            aov_.normal_depth().ptr(),
            aov_.albedo_meshid().ptr(),
            gbuffer,
            m_cam.ptr(),
            m_shapeparam.ptr(), m_shapeparam.num(),
            m_nodetex.ptr(),
            m_primparams.ptr(),
            texVtxPos,
            m_mtxparams.ptr());
    }

    void ReSTIRPathTracing::pick(
        int ix, int iy,
        int width, int height,
        cudaTextureObject_t texVtxPos)
    {
        if (m_willPicklPixel) {
            m_pick.init(1);

            pickPixel << <1, 1 >> > (
                m_pick.ptr(),
                m_pickedInfo.ix, m_pickedInfo.iy,
                width, height,
                m_cam.ptr(),
                m_paths.ptr(),
                aov_.normal_depth().ptr(),
                aov_.albedo_meshid().ptr(),
                m_shapeparam.ptr(), m_shapeparam.num(),
                m_nodetex.ptr(),
                m_primparams.ptr(),
                texVtxPos,
                m_mtxparams.ptr());

            m_pick.readFromDeviceToHostByNum(&m_pickedInfo);

            m_willPicklPixel = false;
        }
    }
}
