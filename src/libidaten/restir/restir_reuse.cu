#include "restir/restir.h"

#include "kernel/pt_common.h"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

#if 1
__host__ __device__ void OnComputeTemporalReuse(
    int idx,
    aten::sampler* sampler,
    const idaten::Reservoir* cur_reservoirs,
    const idaten::Reservoir* prev_reservoirs,
    idaten::Reservoir* dst_reservoirs,
    const idaten::ReSTIRIntermedidate* intermediates,
    idaten::ReSTIRIntermedidate* dst_intermediates,
    const idaten::ReSTIRPathTracing::NormalMaterialStorage* cur_nml_mtrl_buf,
    const idaten::ReSTIRPathTracing::NormalMaterialStorage* prev_nml_mtrl_buf,
    float4 motionDepth,
    int width, int height)
{
    int ix = idx % width;
    int iy = idx / width;

    int reuse_idx = -1;
    auto new_reservoir = cur_reservoirs[idx];

    const auto& cur_nml = cur_nml_mtrl_buf[idx].normal;
    const auto& cur_mtrl_idx = cur_nml_mtrl_buf[idx].mtrl_idx;

    auto r = sampler[idx].nextSample();

    // �O�̃t���[���̃X�N���[�����W.
    int px = (int)(ix + motionDepth.x * width);
    int py = (int)(iy + motionDepth.y * height);

    if (AT_MATH_IS_IN_BOUND(px, 0, width - 1)
        && AT_MATH_IS_IN_BOUND(py, 0, height - 1))
    {
        int pidx = getIdx(px, py, width);

        const auto& prev_nml = prev_nml_mtrl_buf[pidx].normal;
        const auto& prev_mtrl_idx = prev_nml_mtrl_buf[pidx].mtrl_idx;

        // TODO
        // Compare normal and material type
        // Even if material index is different, if the material type is same, it's ok.

        {
            auto prev_reservoir = prev_reservoirs[pidx];

            if (prev_reservoir.w > 0.0f) {
                // NOTE
                // Prev frame�̏��͕����Ă����ƁAm �̐�������������̂ŁA�ǂ����ŗ}�����K�v.
                // �����ŁA�P���Ɉ�萔�ȏ�ɂȂ�����A�����őł��~�߂ɂ���.
                if (prev_reservoir.m > 20 * new_reservoir.m)
                {
                    prev_reservoir.w *= 20 * new_reservoir.m / prev_reservoir.m;
                    prev_reservoir.m = 20 * new_reservoir.m;
                }

                new_reservoir.w += prev_reservoir.w;
                new_reservoir.m += prev_reservoir.m;

                if (r <= prev_reservoir.w / new_reservoir.w) {
                    new_reservoir.light_pdf = prev_reservoir.light_pdf;
                    new_reservoir.light_idx = prev_reservoir.light_idx;
                    reuse_idx = pidx;
                }
            }
        }
    }

    dst_intermediates[idx] = intermediates[idx];

    if (reuse_idx >= 0) {
        dst_reservoirs[idx] = new_reservoir;

        dst_intermediates[idx].light_sample_nml = intermediates[reuse_idx].light_sample_nml;
        dst_intermediates[idx].light_color = intermediates[reuse_idx].light_color;
    }
    else {
        dst_reservoirs[idx] = cur_reservoirs[idx];
    }
}
#endif

__global__ void getMotionDepthBuffer(
    cudaSurfaceObject_t motionDetphBuffer,
    float4* dstMotionDepth,
    int width, int height)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    auto idx = getIdx(ix, iy, width);

    float4 motionDepth;
    surf2Dread(&motionDepth, motionDetphBuffer, ix * sizeof(float4), iy);

    dstMotionDepth[idx] = motionDepth;
}

__global__ void computeTemporalReuse(
    idaten::Path* paths,
    const idaten::Reservoir* __restrict__ cur_reservoirs,
    const idaten::Reservoir* __restrict__ prev_reservoirs,
    idaten::Reservoir* dst_reservoirs,
    const idaten::ReSTIRIntermedidate* __restrict__ intermediates,
    idaten::ReSTIRIntermedidate* dst_intermediates,
    const idaten::ReSTIRPathTracing::NormalMaterialStorage* __restrict__ cur_nml_mtrl_buf,
    const idaten::ReSTIRPathTracing::NormalMaterialStorage* __restrict__ prev_nml_mtrl_buf,
    cudaSurfaceObject_t motionDetphBuffer,
    int width, int height)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    auto idx = getIdx(ix, iy, width);

    float4 motionDepth;
    surf2Dread(&motionDepth, motionDetphBuffer, ix * sizeof(float4), iy);

    OnComputeTemporalReuse(
        idx,
        paths->sampler,
        cur_reservoirs, prev_reservoirs, dst_reservoirs,
        intermediates, dst_intermediates,
        cur_nml_mtrl_buf, prev_nml_mtrl_buf,
        motionDepth,
        width, height);
}

__host__ __device__ void OnComputeSpatialReuse(
    int idx,
    aten::sampler* sampler,
    const idaten::Reservoir* reservoirs,
    idaten::Reservoir* dst_reservoirs,
    const idaten::ReSTIRIntermedidate* intermediates,
    idaten::ReSTIRIntermedidate* dst_intermediates,
    int width, int height)
{
    int ix = idx % width;
    int iy = idx / width;

    int reuse_idx = -1;
    auto new_reservoir = reservoirs[idx];

    auto r = sampler->nextSample();

#pragma unroll
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            const auto xx = ix + x;
            const auto yy = iy + y;

            if (AT_MATH_IS_IN_BOUND(xx, 0, width - 1)
                && AT_MATH_IS_IN_BOUND(yy, 0, height - 1))
            {
                auto new_idx = getIdx(xx, yy, width);
                const auto& reservoir = reservoirs[new_idx];

                if (reservoir.w > 0.0f) {
                    new_reservoir.w += reservoir.w;
                    new_reservoir.m += reservoir.m;

                    if (r <= reservoir.w / new_reservoir.w) {
                        new_reservoir.light_pdf = reservoir.light_pdf;
                        new_reservoir.light_idx = reservoir.light_idx;
                        reuse_idx = new_idx;
                    }
                }
            }
        }
    }

    dst_intermediates[idx] = intermediates[idx];

    if (reuse_idx >= 0) {
        dst_reservoirs[idx] = new_reservoir;

        dst_intermediates[idx].light_sample_nml = intermediates[reuse_idx].light_sample_nml;
        dst_intermediates[idx].light_color = intermediates[reuse_idx].light_color;
    }
    else {
        dst_reservoirs[idx] = reservoirs[idx];
    }
}

__global__ void computeSpatialReuse(
    idaten::Path* paths,
    const idaten::Reservoir* __restrict__ reservoirs,
    idaten::Reservoir* dst_reservoirs,
    const idaten::ReSTIRIntermedidate* __restrict__ intermediates,
    idaten::ReSTIRIntermedidate* dst_intermediates,
    int width, int height)
{
    auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    auto iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= width || iy >= height) {
        return;
    }

    auto idx = getIdx(ix, iy, width);

    OnComputeSpatialReuse(
        idx,
        &paths->sampler[idx],
        reservoirs,
        dst_reservoirs,
        intermediates,
        dst_intermediates,
        width, height
    );
}

namespace idaten {
    std::tuple<int, int> ReSTIRPathTracing::computelReuse(
        int width, int height,
        int bounce)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

        constexpr int TEMPORAL_REUSE_RESERVOIR_SRC_IDX = 0;
        constexpr int TEMPORAL_REUSE_RESERVOIR_DST_IDX = 1;
        constexpr int TEMPORAL_REUSE_RESERVOIR_PREV_FRAME_IDX = 2;
        constexpr int SPATIAL_REUSE_RESERVOIR_DST_IDX = 2;

        int spatial_resue_reservoir_src_idx = 0;
        int spatial_resue_reservoir_dst_idx = SPATIAL_REUSE_RESERVOIR_DST_IDX;

        int spatial_resue_intermediate_src_idx = 0;
        int spatial_resue_intermediate_dst_idx = 1;

        if (bounce == 0) {
#if 0
            if (m_frame > 1) {
                idaten::TypedCudaMemory<float4> tmpMotionDepthBuf;
                tmpMotionDepthBuf.init(width * height);

                CudaGLResourceMapper<decltype(m_motionDepthBuffer)> rscmap(m_motionDepthBuffer);
                auto motionDepthBuffer = m_motionDepthBuffer.bind();

                getMotionDepthBuffer << <grid, block, 0, m_stream >> > (
                    motionDepthBuffer,
                    tmpMotionDepthBuf.ptr(),
                    width, height);

                std::vector<float4> motionDepthBuf;
                tmpMotionDepthBuf.readFromDeviceToHost(motionDepthBuf);

                std::vector<decltype(m_pathSampler)::type> sampler;
                m_pathSampler.readFromDeviceToHost(sampler);

                std::array<std::vector<Reservoir>, 3> reservoir;
                for (int i = 0; i < 3; i++) {
                    m_reservoirs[i].readFromDeviceToHost(reservoir[i]);
                }

                std::array<std::vector<ReSTIRIntermedidate>, 2> intermediate;
                for (int i = 0; i < 2; i++) {
                    m_intermediates[i].readFromDeviceToHost(intermediate[i]);
                }

                std::array<std::vector<NormalMaterialStorage>, 2> nml_mtrl_buf;
                for (int i = 0; i < 2; i++) {
                    m_bufNmlMtrl[i].readFromDeviceToHost(nml_mtrl_buf[i]);
                }

                int curBufNmlMtrlPos = getCurBufNmlMtrlPos();
                int prevBufNmlMtrlPos = getPrevBufNmlMtrlPos();

                constexpr int xx = 43;
                constexpr int yy = 719 - 718;

                for (int i = 0; i < motionDepthBuf.size(); i++) {
                    int ix = i % 1280;
                    int iy = i / 1280;

                    OnComputeTemporalReuse(
                        i,
                        sampler.data(),
                        reservoir[TEMPORAL_REUSE_RESERVOIR_SRC_IDX].data(),
                        reservoir[TEMPORAL_REUSE_RESERVOIR_PREV_FRAME_IDX].data(),
                        reservoir[TEMPORAL_REUSE_RESERVOIR_DST_IDX].data(),
                        intermediate[spatial_resue_intermediate_src_idx].data(),
                        intermediate[spatial_resue_intermediate_dst_idx].data(),
                        nml_mtrl_buf[curBufNmlMtrlPos].data(),
                        nml_mtrl_buf[prevBufNmlMtrlPos].data(),
                        motionDepthBuf[i],
                        width, height);
                }
            }
#elif 1
            if (m_frame > 1) {
                int curBufNmlMtrlPos = getCurBufNmlMtrlPos();
                int prevBufNmlMtrlPos = getPrevBufNmlMtrlPos();

                CudaGLResourceMapper<decltype(m_motionDepthBuffer)> rscmap(m_motionDepthBuffer);
                auto motionDepthBuffer = m_motionDepthBuffer.bind();

                computeTemporalReuse << <grid, block, 0, m_stream >> > (
                    m_paths.ptr(),
                    m_reservoirs[TEMPORAL_REUSE_RESERVOIR_SRC_IDX].ptr(),
                    m_reservoirs[TEMPORAL_REUSE_RESERVOIR_PREV_FRAME_IDX].ptr(),
                    m_reservoirs[TEMPORAL_REUSE_RESERVOIR_DST_IDX].ptr(),
                    m_intermediates[spatial_resue_intermediate_src_idx].ptr(),
                    m_intermediates[spatial_resue_intermediate_dst_idx].ptr(),
                    m_bufNmlMtrl[curBufNmlMtrlPos].ptr(),
                    m_bufNmlMtrl[prevBufNmlMtrlPos].ptr(),
                    motionDepthBuffer,
                    width, height);

                checkCudaKernel(computeTemporalReuse);

                updateCurBufNmlMtrlPos();

                spatial_resue_reservoir_src_idx = TEMPORAL_REUSE_RESERVOIR_DST_IDX;

                std::swap(
                    spatial_resue_intermediate_src_idx,
                    spatial_resue_intermediate_dst_idx);
            }
#endif

            computeSpatialReuse << <grid, block, 0, m_stream >> > (
                m_paths.ptr(),
                m_reservoirs[spatial_resue_reservoir_src_idx].ptr(),
                m_reservoirs[spatial_resue_reservoir_dst_idx].ptr(),
                m_intermediates[spatial_resue_intermediate_src_idx].ptr(),
                m_intermediates[spatial_resue_intermediate_dst_idx].ptr(),
                width, height);

            checkCudaKernel(computeSpatialReuse);
        }

        return std::make_tuple(
            spatial_resue_reservoir_dst_idx,
            spatial_resue_intermediate_dst_idx);
    }
}