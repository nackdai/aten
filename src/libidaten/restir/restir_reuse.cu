#include "restir/restir.h"

#include "kernel/pt_common.h"
#include "kernel/context.cuh"
#include "kernel/material.cuh"
#include "kernel/light.cuh"

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"

#include "aten4idaten.h"

__device__ float computeEnergyCost(
    idaten::Context& ctxt,
    const aten::vec4 albedo,
    const idaten::Reservoir neighbor_reservoir,
    const idaten::ReSTIRIntermedidate& cur_info,
    const idaten::ReSTIRIntermedidate& neighbor_info)
{
    aten::MaterialParameter mtrl;
    if (cur_info.is_mtrl_valid) {
        mtrl = ctxt.mtrls[cur_info.mtrl_idx];
    }
    else {
        mtrl = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
        mtrl.baseColor = aten::vec3(1.0f);
    }

    const aten::vec3 orienting_normal(
        cur_info.nml_x,
        cur_info.nml_y,
        cur_info.nml_z);

    auto dir_to_light = neighbor_info.light_pos - cur_info.p;
    auto dist_to_light = length(dir_to_light);
    dir_to_light /= dist_to_light;

    auto bsdf = sampleBSDF(
        &ctxt,
        &mtrl,
        orienting_normal,
        cur_info.wi,
        dir_to_light,
        cur_info.u, cur_info.v,
        albedo);

    const auto neighbor_light_idx = neighbor_reservoir.light_idx;

    aten::vec3 energy;
    computeLighting(
        energy,
        ctxt.lights[neighbor_light_idx],
        orienting_normal,
        neighbor_info.light_sample_nml,
        neighbor_reservoir.light_pdf,
        neighbor_info.light_color,
        dir_to_light,
        dist_to_light);

    energy = energy * bsdf;

    auto cost = (energy.x + energy.y + energy.z) / 3;

    return cost;
}

__global__ void computeTemporalReuse(
    idaten::Path* paths,
    const aten::LightParameter* __restrict__ lights,
    const aten::MaterialParameter* __restrict__ mtrls,
    cudaTextureObject_t* textures,
    const float4* __restrict__ aovTexclrMeshid,
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

    idaten::Context ctxt;
    {
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
        ctxt.textures = textures;
    }

    int reuse_idx = -1;
    auto new_reservoir = cur_reservoirs[idx];

    const auto& cur_nml = cur_nml_mtrl_buf[idx].normal;
    const auto& cur_mtrl_idx = cur_nml_mtrl_buf[idx].mtrl_idx;

    const auto& albedo_meshid = aovTexclrMeshid[idx];
    const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

    auto& sampler = paths->sampler[idx];

    // 前のフレームのスクリーン座標.
    int px = (int)(ix + motionDepth.x * width);
    int py = (int)(iy + motionDepth.y * height);

    if (AT_MATH_IS_IN_BOUND(px, 0, width - 1)
        && AT_MATH_IS_IN_BOUND(py, 0, height - 1))
    {
        int prev_idx = getIdx(px, py, width);

        const auto& prev_nml = prev_nml_mtrl_buf[prev_idx].normal;
        const auto& prev_mtrl_idx = prev_nml_mtrl_buf[prev_idx].mtrl_idx;

        // TODO
        // Compare normal and material type
        // Even if material index is different, if the material type is same, it's ok.

        {
            const auto& prev_reservoir = prev_reservoirs[prev_idx];

            if (prev_reservoir.light_idx > 0) {
                const auto& cur_info = intermediates[idx];
                const auto& prev_info = intermediates[prev_idx];

                aten::MaterialParameter mtrl;
                if (cur_info.is_mtrl_valid) {
                    mtrl = mtrls[cur_info.mtrl_idx];
                }
                else {
                    mtrl = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
                    mtrl.baseColor = aten::vec3(1.0f);
                }

                const aten::vec3 orienting_normal(
                    cur_info.nml_x,
                    cur_info.nml_y,
                    cur_info.nml_z);

                auto dir_to_light = prev_info.light_pos - cur_info.p;
                auto dist_to_light = length(dir_to_light);
                dir_to_light /= dist_to_light;

                auto bsdf = sampleBSDF(
                    &ctxt,
                    &mtrl,
                    orienting_normal,
                    cur_info.wi,
                    dir_to_light,
                    cur_info.u, cur_info.v,
                    albedo);

                const auto prev_light_idx = prev_reservoir.light_idx;

                aten::vec3 energy;
                computeLighting(
                    energy,
                    lights[prev_light_idx],
                    orienting_normal,
                    prev_info.light_sample_nml,
                    prev_reservoir.light_pdf,
                    prev_info.light_color,
                    dir_to_light,
                    dist_to_light);

                energy = energy * bsdf;

                auto cost = (energy.x + energy.y + energy.z) / 3;

                auto w_sum = new_reservoir.w + cost;

                if (w_sum > 0.0f
                    && sampler.nextSample() <= cost / w_sum)
                {
                    new_reservoir.w = w_sum;
                    new_reservoir.m += min(prev_reservoir.m, 20 * new_reservoir.m);
                    new_reservoir.selected_cost = cost;
                    new_reservoir.pdf = w_sum / (cost * new_reservoir.m);
                    new_reservoir.light_pdf = prev_reservoir.light_pdf;
                    new_reservoir.light_idx = prev_reservoir.light_idx;
                    reuse_idx = prev_idx;
                }
            }
        }
    }

    if (!isfinite(new_reservoir.pdf)) {
        new_reservoir.light_pdf = real(0);
        new_reservoir.light_idx = -1;
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

__global__ void computeSpatialReuse(
    idaten::Path* paths,
    const aten::LightParameter* __restrict__ lights,
    const aten::MaterialParameter* __restrict__ mtrls,
    cudaTextureObject_t* textures,
    const float4* __restrict__ aovTexclrMeshid,
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

    idaten::Context ctxt;
    {
        ctxt.mtrls = mtrls;
        ctxt.lights = lights,
        ctxt.textures = textures;
    }

    int reuse_idx = -1;
    auto new_reservoir = reservoirs[idx];

    const auto& cur_info = intermediates[idx];

    const auto& albedo_meshid = aovTexclrMeshid[idx];
    const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

    auto& sampler = paths->sampler[idx];

#pragma unroll
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            const auto xx = ix + x;
            const auto yy = iy + y;

            if (AT_MATH_IS_IN_BOUND(xx, 0, width - 1)
                && AT_MATH_IS_IN_BOUND(yy, 0, height - 1))
            {
                auto neighbor_idx = getIdx(xx, yy, width);
                const auto& neighbor_reservoir = reservoirs[neighbor_idx];

                const auto& neighbor_info = intermediates[neighbor_idx];

                auto cost = computeEnergyCost(
                    ctxt,
                    albedo,
                    neighbor_reservoir,
                    cur_info,
                    neighbor_info);

                auto w_sum = new_reservoir.w + cost;

                if (w_sum > 0.0f
                    && sampler.nextSample() <= cost / w_sum)
                {
                    new_reservoir.w = w_sum;
                    new_reservoir.m += neighbor_reservoir.m;
                    new_reservoir.selected_cost = cost;
                    new_reservoir.pdf = w_sum / (cost * new_reservoir.m);
                    new_reservoir.light_pdf = neighbor_reservoir.light_pdf;
                    new_reservoir.light_idx = neighbor_reservoir.light_idx;
                    reuse_idx = neighbor_idx;
                }
            }
        }
    }

    if (!isfinite(new_reservoir.pdf)) {
        new_reservoir.light_pdf = real(0);
        new_reservoir.light_idx = -1;
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
            if (m_restirMode == ReSTIRMode::ReSTIR) {
                if (m_frame > 1) {
                    int curBufNmlMtrlPos = getCurBufNmlMtrlPos();
                    int prevBufNmlMtrlPos = getPrevBufNmlMtrlPos();

                    CudaGLResourceMapper<decltype(m_motionDepthBuffer)> rscmap(m_motionDepthBuffer);
                    auto motionDepthBuffer = m_motionDepthBuffer.bind();

                    computeTemporalReuse << <grid, block, 0, m_stream >> > (
                        m_paths.ptr(),
                        m_lightparam.ptr(),
                        m_mtrlparam.ptr(),
                        m_tex.ptr(),
                        m_aovTexclrMeshid.ptr(),
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
            }

            computeSpatialReuse << <grid, block, 0, m_stream >> > (
                m_paths.ptr(),
                m_lightparam.ptr(),
                m_mtrlparam.ptr(),
                m_tex.ptr(),
                m_aovTexclrMeshid.ptr(),
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
