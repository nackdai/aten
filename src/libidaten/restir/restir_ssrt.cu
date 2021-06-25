#include "restir/restir.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__global__ void hitTestPrimaryRayInScreenSpace(
    idaten::TileDomain tileDomain,
    cudaSurfaceObject_t gbuffer,
    idaten::Path* paths,
    aten::Intersection* isects,
    int* hitbools,
    int width, int height,
    const aten::vec4 camPos,
    const aten::GeomParameter* __restrict__ geoms,
    const aten::PrimitiveParamter* __restrict__ prims,
    const aten::mat4* __restrict__ matrices,
    cudaTextureObject_t vtxPos)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= tileDomain.w || iy >= tileDomain.h) {
        return;
    }

    const auto idx = getIdx(ix, iy, tileDomain.w);

    paths->attrib[idx].isHit = false;

    hitbools[idx] = 0;

    if (paths->attrib[idx].isTerminate) {
        return;
    }

    ix += tileDomain.x;
    iy += tileDomain.y;

    // Sample data from texture.
    float4 data;
    surf2Dread(&data, gbuffer, ix * sizeof(float4), iy);

    // NOTE
    // x : objid
    // y : primid
    // zw : bary centroid

    int objid = __float_as_int(data.x);
    int primid = __float_as_int(data.y);

    isects[idx].objid = objid;
    isects[idx].primid = primid;

    // bary centroid.
    isects[idx].a = data.z;
    isects[idx].b = data.w;

    if (objid >= 0) {
        aten::PrimitiveParamter prim;
        prim.v0 = ((aten::vec4*)prims)[primid * aten::PrimitiveParamter_float4_size + 0];
        prim.v1 = ((aten::vec4*)prims)[primid * aten::PrimitiveParamter_float4_size + 1];

        isects[idx].mtrlid = prim.mtrlid;
        isects[idx].meshid = prim.gemoid;

        const auto* obj = &geoms[objid];

        float4 p0 = tex1Dfetch<float4>(vtxPos, prim.idx[0]);
        float4 p1 = tex1Dfetch<float4>(vtxPos, prim.idx[1]);
        float4 p2 = tex1Dfetch<float4>(vtxPos, prim.idx[2]);

        real a = data.z;
        real b = data.w;
        real c = 1 - a - b;

        // 重心座標系(barycentric coordinates).
        // v0基準.
        // p = (1 - a - b)*v0 + a*v1 + b*v2
        auto p = c * p0 + a * p1 + b * p2;
        aten::vec4 vp(p.x, p.y, p.z, 1.0f);

        if (obj->mtxid >= 0) {
            auto mtxL2W = matrices[obj->mtxid * 2 + 0];
            vp = mtxL2W.apply(vp);
        }

        isects[idx].t = (camPos - vp).length();

        paths->attrib[idx].isHit = true;
        hitbools[idx] = 1;
    }
    else {
        paths->attrib[idx].isHit = false;
        hitbools[idx] = 0;
    }
}

namespace idaten
{
    void ReSTIRPathTracing::onScreenSpaceHitTest(
        int width, int height,
        int bounce,
        cudaTextureObject_t texVtxPos)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        aten::vec4 campos = aten::vec4(m_camParam.origin, 1.0f);

        CudaGLResourceMapper<decltype(m_gbuffer)> rscmap(m_gbuffer);
        auto gbuffer = m_gbuffer.bind();

        hitTestPrimaryRayInScreenSpace << <grid, block >> > (
            m_tileDomain,
            gbuffer,
            m_paths.ptr(),
            m_isects.ptr(),
            m_hitbools.ptr(),
            width, height,
            campos,
            m_shapeparam.ptr(),
            m_primparams.ptr(),
            m_mtxparams.ptr(),
            texVtxPos);

        checkCudaKernel(hitTestPrimaryRayInScreenSpace);
    }
}
