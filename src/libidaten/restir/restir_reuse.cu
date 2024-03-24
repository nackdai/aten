#include "restir/restir.h"

#include "kernel/pt_common.h"
#include "kernel/device_scene_context.cuh"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"
#include "light/light_impl.h"
#include "renderer/pathtracing/pt_params.h"
#include "renderer/restir/restir_impl.h"

__global__ void computeTemporalReuse(
    idaten::Path paths,
    idaten::context ctxt,
    const aten::ObjectParameter* __restrict__ shapes,
    const aten::MaterialParameter* __restrict__ mtrls,
    const aten::LightParameter* __restrict__ lights,
    const aten::TriangleParameter* __restrict__ prims,
    const aten::mat4* __restrict__ matrices,
    const float4* __restrict__ aovTexclrMeshid,
    idaten::Reservoir* reservoirs,
    const idaten::Reservoir* __restrict__ prev_reservoirs,
    const idaten::ReSTIRInfo* __restrict__ curr_infos,
    const idaten::ReSTIRInfo* __restrict__ prev_infos,
    cudaSurfaceObject_t motionDetphBuffer,
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    auto idx = getIdx(ix, iy, width);

    if (paths.attrib[idx].isTerminate) {
        return;
    }

    ctxt.shapes = shapes;
    ctxt.mtrls = mtrls;
    ctxt.lights = lights;
    ctxt.prims = prims;
    ctxt.matrices = matrices;

    auto& sampler = paths.sampler[idx];

    const auto size = width * height;

    aten::const_span prev_reservoirs_as_span(prev_reservoirs, size);
    aten::const_span prev_resitr_infos(prev_infos, size);
    aten::const_span aov_texclr_meshid(aovTexclrMeshid, size);

    AT_NAME::restir::ApplyTemporalReuse(
        idx,
        width, height,
        ctxt,
        sampler,
        reservoirs[idx],
        curr_infos[idx],
        prev_reservoirs_as_span,
        prev_resitr_infos,
        aov_texclr_meshid, motionDetphBuffer);
}

__global__ void computeSpatialReuse(
    idaten::Path paths,
    idaten::context ctxt,
    const aten::ObjectParameter* __restrict__ shapes,
    const aten::MaterialParameter* __restrict__ mtrls,
    const aten::LightParameter* __restrict__ lights,
    const aten::TriangleParameter* __restrict__ prims,
    const aten::mat4* __restrict__ matrices,
    const float4* __restrict__ aovTexclrMeshid,
    const idaten::Reservoir* __restrict__ reservoirs,
    idaten::Reservoir* dst_reservoirs,
    const idaten::ReSTIRInfo* __restrict__ infos,
    int32_t width, int32_t height)
{
    int32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    auto idx = getIdx(ix, iy, width);

    if (paths.attrib[idx].isTerminate) {
        return;
    }

    ctxt.shapes = shapes;
    ctxt.mtrls = mtrls;
    ctxt.lights = lights;
    ctxt.prims = prims;
    ctxt.matrices = matrices;

    auto& sampler = paths.sampler[idx];

    const auto size = width * height;

    aten::const_span reservoirs_as_span(reservoirs, size);
    aten::const_span resitr_infos(infos, size);
    aten::const_span aov_texclr_meshid(aovTexclrMeshid, size);

    AT_NAME::restir::ApplySpatialReuse(
        idx,
        width, height,
        ctxt,
        sampler,
        dst_reservoirs[idx],
        reservoirs_as_span,
        resitr_infos,
        aov_texclr_meshid);
}

namespace idaten {
    aten::tuple<int32_t, int32_t> ReSTIRPathTracing::ComputelReuse(
        int32_t width, int32_t height,
        int32_t bounce)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

        const auto curr_reservoirs_idx = m_reservoirs.GetCurrParamsIdx();
        const auto dst_reservoirs_idx = m_reservoirs.GetDestinationParamsIdxForSpatialReuse();

        const auto curr_restir_info_idx = m_restir_infos.GetCurrParamsIdx();

        if (bounce == 0) {
            // NOTE
            // temporal reuse で利用する previous reservoir は
            // spatial reuse をする前のものでないといけない
            // spatial reuse はあくまでも現在フレームに対して行われるもので
            // 次フレームに影響を与えないようにする
            // e.g.
            //  - frame 1
            //     cur:0
            //     prev:N/A (最初なので temporal は skip)
            //     spatial_dst:1
            //     pos=0 -> pos=1(for next)
            //  - frame 2
            //     cur:1(=pos)
            //     prev:0
            //     spatial_dst:0
            //     pos=1 -> pos=0(for next)
            //     このとき prev は前フレームの cur となっている

            auto& curr_reservoirs = m_reservoirs.GetCurrParams();
            auto& prev_reservoirs = m_reservoirs.GetPreviousFrameParamsForTemporalReuse();
            auto& dst_reservoirs = m_reservoirs.GetDestinationParamsForSpatialReuse();
            m_reservoirs.Update();

            auto& curr_restir_infos = m_restir_infos.GetCurrParams();
            auto& prev_restir_infos = m_restir_infos.GetPreviousFrameParamsForTemporalReuse();
            m_restir_infos.Update();

            if (m_restirMode == ReSTIRMode::ReSTIR
                || m_restirMode == ReSTIRMode::TemporalReuse) {
                if (m_frame > 1) {
                    CudaGLResourceMapper<decltype(m_motionDepthBuffer)> rscmap(m_motionDepthBuffer);
                    auto motionDepthBuffer = m_motionDepthBuffer.bind();

                    computeTemporalReuse << <grid, block, 0, m_stream >> > (
                        path_host_->paths,
                        ctxt_host_.ctxt,
                        ctxt_host_.shapeparam.data(),
                        ctxt_host_.mtrlparam.data(),
                        ctxt_host_.lightparam.data(),
                        ctxt_host_.primparams.data(),
                        ctxt_host_.mtxparams.data(),
                        aov_.albedo_meshid().data(),
                        curr_reservoirs.data(),
                        prev_reservoirs.data(),
                        curr_restir_infos.data(),
                        prev_restir_infos.data(),
                        motionDepthBuffer,
                        width, height);

                    checkCudaKernel(computeTemporalReuse);
                }
            }

            if (m_restirMode == ReSTIRMode::ReSTIR
                || m_restirMode == ReSTIRMode::SpatialReuse) {
                computeSpatialReuse << <grid, block, 0, m_stream >> > (
                    path_host_->paths,
                    ctxt_host_.ctxt,
                    ctxt_host_.shapeparam.data(),
                    ctxt_host_.mtrlparam.data(),
                    ctxt_host_.lightparam.data(),
                    ctxt_host_.primparams.data(),
                    ctxt_host_.mtxparams.data(),
                    aov_.albedo_meshid().data(),
                    curr_reservoirs.data(),
                    dst_reservoirs.data(),
                    curr_restir_infos.data(),
                    width, height);

                checkCudaKernel(computeSpatialReuse);

                return aten::make_tuple(dst_reservoirs_idx, curr_restir_info_idx);
            }
        }

        return aten::make_tuple(curr_reservoirs_idx, curr_restir_info_idx);
    }
}
