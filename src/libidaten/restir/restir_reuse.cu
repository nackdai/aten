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

__global__ void computeTemporalReuse(
    idaten::Path paths,
    const aten::ObjectParameter* __restrict__ shapes,
    const aten::MaterialParameter* __restrict__ mtrls,
    const aten::LightParameter* __restrict__ lights, int32_t lightnum,
    const aten::TriangleParameter* __restrict__ prims,
    cudaTextureObject_t vtxPos,
    cudaTextureObject_t vtxNml,
    const aten::mat4* __restrict__ matrices,
    cudaTextureObject_t* textures,
    const float4* __restrict__ aovTexclrMeshid,
    idaten::Reservoir* reservoirs,
    const idaten::Reservoir* __restrict__ prev_reservoirs,
    const idaten::ReSTIRInfo* __restrict__ infos,
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

    idaten::context ctxt;
    {
        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lightnum = lightnum;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.vtxPos = vtxPos;
        ctxt.vtxNml = vtxNml;
        ctxt.matrices = matrices;
        ctxt.textures = textures;
    }

    const auto& self_info = infos[idx];
    aten::MaterialParameter mtrl;
    gatherMaterialInfo(
        mtrl,
        &ctxt,
        self_info.mtrl_idx,
        self_info.is_voxel);

    const auto& normal = self_info.nml;

    auto& sampler = paths.sampler[idx];

    const auto& albedo_meshid = aovTexclrMeshid[idx];
    const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

    auto& comibined_reservoir = reservoirs[idx];

    float selected_target_density = comibined_reservoir.isValid()
        ? comibined_reservoir.target_density_
        : 0.0f;

    // NOTE
    // In this case, self reservoir's M should be number of number of light sampling.
    const auto maxM = 20 * comibined_reservoir.m_;

    float4 motionDepth;
    surf2Dread(&motionDepth, motionDetphBuffer, ix * sizeof(float4), iy);

    // 前のフレームのスクリーン座標.
    int32_t px = (int32_t)(ix + motionDepth.x * width);
    int32_t py = (int32_t)(iy + motionDepth.y * height);

    bool is_acceptable = AT_MATH_IS_IN_BOUND(px, 0, width - 1)
        && AT_MATH_IS_IN_BOUND(py, 0, height - 1);

    if (is_acceptable)
    {
        aten::LightSampleResult lightsample;

        auto neighbor_idx = getIdx(px, py, width);
        const auto& neighbor_reservoir = prev_reservoirs[neighbor_idx];

        auto m = std::min(neighbor_reservoir.m_, maxM);

        if (neighbor_reservoir.isValid()) {
            const auto& neighbor_info = infos[neighbor_idx];

            const auto& neighbor_normal = neighbor_info.nml;

            aten::MaterialParameter neightbor_mtrl;
            auto is_valid_mtrl = gatherMaterialInfo(
                neightbor_mtrl,
                &ctxt,
                neighbor_info.mtrl_idx,
                neighbor_info.is_voxel);

            // Check how close with neighbor pixel.
            is_acceptable = is_valid_mtrl
                && (mtrl.type == neightbor_mtrl.type)
                && (dot(normal, neighbor_normal) >= 0.95f);

            if (is_acceptable) {
                const auto light_pos = neighbor_reservoir.light_idx_;

                const auto& light = ctxt.lights[light_pos];

                sampleLight(&lightsample, &ctxt, &light, self_info.p, neighbor_normal, &sampler, 0);

                aten::vec3 nmlLight = lightsample.nml;
                aten::vec3 dirToLight = normalize(lightsample.dir);

                auto pdf = AT_NAME::material::samplePDF(
                    &neightbor_mtrl,
                    normal,
                    self_info.wi, dirToLight,
                    self_info.u, self_info.v);
                auto brdf = AT_NAME::material::sampleBSDFWithExternalAlbedo(
                    &neightbor_mtrl,
                    normal,
                    self_info.wi, dirToLight,
                    self_info.u, self_info.v,
                    albedo,
                    self_info.pre_sampled_r);
                brdf /= pdf;

                auto cosShadow = dot(normal, dirToLight);
                auto cosLight = dot(nmlLight, -dirToLight);
                auto dist2 = aten::squared_length(lightsample.dir);

                auto energy = brdf * lightsample.finalColor;

                cosShadow = aten::abs(cosShadow);

                if (cosShadow > 0 && cosLight > 0) {
                    if (light.attrib.isSingular) {
                        energy = energy * cosShadow * cosLight;
                    }
                    else {
                        energy = energy * cosShadow * cosLight / dist2;
                    }
                }
                else {
                    energy.x = energy.y = energy.z = 0.0f;
                }

                auto target_density = (energy.x + energy.y + energy.z) / 3; // p_hat

                auto weight = target_density * neighbor_reservoir.pdf_ * m;

                auto r = sampler.nextSample();

                if (comibined_reservoir.update(lightsample, light_pos, weight, m, r)) {
                    selected_target_density = target_density;
                }
            }
        }
        else {
            comibined_reservoir.update(lightsample, -1, 0.0f, m, 0.0f);
        }
    }

    if (selected_target_density > 0.0f) {
        comibined_reservoir.target_density_ = selected_target_density;
        // NOTE
        // 1/p_hat(xz) * (1/M * w_sum) = w_sum / (p_hat(xi) * M)
        comibined_reservoir.pdf_ = comibined_reservoir.w_sum_ / (comibined_reservoir.target_density_ * comibined_reservoir.m_);
    }

    if (!isfinite(comibined_reservoir.pdf_)) {
        comibined_reservoir.clear();
    }
}

__global__ void computeSpatialReuse(
    idaten::Path paths,
    const aten::ObjectParameter* __restrict__ shapes,
    const aten::MaterialParameter* __restrict__ mtrls,
    const aten::LightParameter* __restrict__ lights, int32_t lightnum,
    const aten::TriangleParameter* __restrict__ prims,
    cudaTextureObject_t vtxPos,
    cudaTextureObject_t vtxNml,
    const aten::mat4* __restrict__ matrices,
    cudaTextureObject_t* textures,
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

    idaten::context ctxt;
    {
        ctxt.shapes = shapes;
        ctxt.mtrls = mtrls;
        ctxt.lightnum = lightnum;
        ctxt.lights = lights;
        ctxt.prims = prims;
        ctxt.vtxPos = vtxPos;
        ctxt.vtxNml = vtxNml;
        ctxt.matrices = matrices;
        ctxt.textures = textures;
    }

    const auto& self_info = infos[idx];
    aten::MaterialParameter mtrl;
    gatherMaterialInfo(
        mtrl,
        &ctxt,
        self_info.mtrl_idx,
        self_info.is_voxel);

    const auto& normal = self_info.nml;

    auto& sampler = paths.sampler[idx];

    const auto& albedo_meshid = aovTexclrMeshid[idx];
    const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

    static const int32_t offset_x[] = {
        -1,  0,  1,
        -1,  1,
        -1,  0,  1,
    };
    static const int32_t offset_y[] = {
        -1, -1, -1,
         0,  0,
         1,  1,  1,
    };

    auto& comibined_reservoir = dst_reservoirs[idx];
    comibined_reservoir.clear();

    const auto& reservoir = reservoirs[idx];

    float selected_target_density = 0.0f;

    if (reservoir.isValid()) {
        comibined_reservoir = reservoir;
        selected_target_density = reservoir.target_density_;
    }

#pragma unroll
    for (int32_t i = 0; i < AT_COUNTOF(offset_x); i++) {
        const int32_t xx = ix + offset_x[i];
        const int32_t yy = iy + offset_y[i];

        bool is_acceptable = AT_MATH_IS_IN_BOUND(xx, 0, width - 1)
            && AT_MATH_IS_IN_BOUND(yy, 0, height - 1);

        if (is_acceptable)
        {
            aten::LightSampleResult lightsample;

            auto neighbor_idx = getIdx(xx, yy, width);
            const auto& neighbor_reservoir = reservoirs[neighbor_idx];

            if (neighbor_reservoir.isValid()) {
                const auto& neighbor_info = infos[neighbor_idx];

                const auto& neighbor_normal = neighbor_info.nml;

                aten::MaterialParameter neightbor_mtrl;
                auto is_valid_mtrl = gatherMaterialInfo(
                    neightbor_mtrl,
                    &ctxt,
                    neighbor_info.mtrl_idx,
                    neighbor_info.is_voxel);

                // Check how close with neighbor pixel.
                is_acceptable = is_valid_mtrl
                    && (mtrl.type == neightbor_mtrl.type)
                    && (dot(normal, neighbor_normal) >= 0.95f);

                if (is_acceptable) {
                    const auto light_pos = neighbor_reservoir.light_idx_;

                    const auto& light = ctxt.lights[light_pos];

                    sampleLight(&lightsample, &ctxt, &light, self_info.p, neighbor_normal, &sampler, 0);

                    aten::vec3 nmlLight = lightsample.nml;
                    aten::vec3 dirToLight = normalize(lightsample.dir);

                    auto pdf = AT_NAME::material::samplePDF(
                        &neightbor_mtrl,
                        normal,
                        self_info.wi, dirToLight,
                        self_info.u, self_info.v);
                    auto brdf = AT_NAME::material::sampleBSDFWithExternalAlbedo(
                        &neightbor_mtrl,
                        normal,
                        self_info.wi, dirToLight,
                        self_info.u, self_info.v,
                        albedo,
                        self_info.pre_sampled_r);
                    brdf /= pdf;

                    auto cosShadow = dot(normal, dirToLight);
                    auto cosLight = dot(nmlLight, -dirToLight);
                    auto dist2 = aten::squared_length(lightsample.dir);

                    auto energy = brdf * lightsample.finalColor;

                    cosShadow = aten::abs(cosShadow);

                    if (cosShadow > 0 && cosLight > 0) {
                        if (light.attrib.isSingular) {
                            energy = energy * cosShadow * cosLight;
                        }
                        else {
                            energy = energy * cosShadow * cosLight / dist2;
                        }
                    }
                    else {
                        energy.x = energy.y = energy.z = 0.0f;
                    }

                    auto target_density = (energy.x + energy.y + energy.z) / 3; // p_hat

                    auto m = neighbor_reservoir.m_;
                    auto weight = target_density * neighbor_reservoir.pdf_ * m;

                    auto r = sampler.nextSample();

                    if (comibined_reservoir.update(lightsample, light_pos, weight, m, r)) {
                        selected_target_density = target_density;
                    }
                }
            }
            else {
                comibined_reservoir.update(lightsample, -1, 0.0f, neighbor_reservoir.m_, 0.0f);
            }
        }
    }

    if (selected_target_density > 0.0f) {
        comibined_reservoir.target_density_ = selected_target_density;
        // NOTE
        // 1/p_hat(xz) * (1/M * w_sum) = w_sum / (p_hat(xi) * M)
        comibined_reservoir.pdf_ = comibined_reservoir.w_sum_ / (comibined_reservoir.target_density_ * comibined_reservoir.m_);
    }

    if (!isfinite(comibined_reservoir.pdf_)) {
        comibined_reservoir.clear();
    }
}

namespace idaten {
    int32_t ReSTIRPathTracing::computelReuse(
        int32_t width, int32_t height,
        int32_t bounce,
        cudaTextureObject_t texVtxPos,
        cudaTextureObject_t texVtxNml)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);

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

            const auto cur_idx = m_curReservoirPos;
            const auto prev_idx = (m_curReservoirPos + 1) & 0x01;
            const auto dst_idx = (m_curReservoirPos + 1) & 0x01;

            m_curReservoirPos = (m_curReservoirPos + 1) & 0x01;
            if (m_restirMode == ReSTIRMode::ReSTIR
                || m_restirMode == ReSTIRMode::TemporalReuse) {
                if (m_frame > 1) {
                    CudaGLResourceMapper<decltype(m_motionDepthBuffer)> rscmap(m_motionDepthBuffer);
                    auto motionDepthBuffer = m_motionDepthBuffer.bind();

                    computeTemporalReuse << <grid, block, 0, m_stream >> > (
                        m_paths,
                        m_shapeparam.ptr(),
                        m_mtrlparam.ptr(),
                        m_lightparam.ptr(), m_lightparam.num(),
                        m_primparams.ptr(),
                        texVtxPos, texVtxNml,
                        m_mtxparams.ptr(),
                        m_tex.ptr(),
                        aov_.albedo_meshid().ptr(),
                        m_reservoirs[cur_idx].ptr(),
                        m_reservoirs[prev_idx].ptr(),
                        m_restir_infos.ptr(),
                        motionDepthBuffer,
                        width, height);

                    checkCudaKernel(computeTemporalReuse);
                }
            }

            if (m_restirMode == ReSTIRMode::ReSTIR
                || m_restirMode == ReSTIRMode::SpatialReuse) {
                computeSpatialReuse << <grid, block, 0, m_stream >> > (
                    m_paths,
                    m_shapeparam.ptr(),
                    m_mtrlparam.ptr(),
                    m_lightparam.ptr(), m_lightparam.num(),
                    m_primparams.ptr(),
                    texVtxPos, texVtxNml,
                    m_mtxparams.ptr(),
                    m_tex.ptr(),
                    aov_.albedo_meshid().ptr(),
                    m_reservoirs[cur_idx].ptr(),
                    m_reservoirs[dst_idx].ptr(),
                    m_restir_infos.ptr(),
                    width, height);

                checkCudaKernel(computeSpatialReuse);

                return dst_idx;
            }
        }

        return m_curReservoirPos;
    }
}
