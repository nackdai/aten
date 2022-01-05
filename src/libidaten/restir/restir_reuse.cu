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

#if 0
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
#endif

#if 0
__host__ void computeSpatialReuse(
    int ix, int iy,
    aten::sampler* samplers,
    const aten::LightParameter* lights,
    const aten::MaterialParameter* mtrls,
    const float4* aovTexclrMeshid,
    const idaten::Reservoir* reservoirs,
    idaten::Reservoir* dst_reservoirs,
    const idaten::ReSTIRInfo* infos,
    int width, int height)
{
    auto idx = getIdx(ix, iy, width);

    idaten::Context ctxt;
    {
        ctxt.mtrls = mtrls;
        ctxt.lights = lights;
    }

    auto& sampler = samplers[idx];

    const auto& albedo_meshid = aovTexclrMeshid[idx];
    const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

    static const int offset_x[] = {
        -1,  0,  1,
        -1,  1,
        -1,  0,  1,
    };
    static const int offset_y[] = {
        -1, -1, -1,
         0,  0,
         1,  1,  1,
    };

    auto& comibined_reservoir = dst_reservoirs[idx];
    comibined_reservoir.clear();

    const auto& reservoir = reservoirs[idx];
    const auto& self_info = infos[idx];

    float selected_target_density = 0.0f;

    if (reservoir.isValid()) {
        comibined_reservoir = reservoir;
        selected_target_density = reservoir.target_density_;
    }

    const auto& normal = self_info.nml;

#pragma unroll
    for (int i = 0; i < AT_COUNTOF(offset_x); i++) {
        const auto xx = ix + offset_x[i];
        const auto yy = iy + offset_y[i];

        bool is_acceptable = AT_MATH_IS_IN_BOUND(xx, 0, width - 1)
            && AT_MATH_IS_IN_BOUND(yy, 0, height - 1);

        if (is_acceptable)
        {
            auto r = sampler.nextSample();

            auto neighbor_idx = getIdx(xx, yy, width);
            const auto& neighbor_reservoir = reservoirs[neighbor_idx];

            auto m = neighbor_reservoir.m_;

            aten::LightSampleResult lightsample;

            if (neighbor_reservoir.isValid()) {
                const auto& neighbor_info = infos[neighbor_idx];

                const auto light_pos = neighbor_reservoir.light_idx_;

                const auto& light = ctxt.lights[light_pos];

                //sampleLight(&lightsample, ctxt, &light, org, normal, sampler, lod);

                // TODO
                // Only point light
                AT_NAME::PointLight::sample(&light, self_info.p, &sampler, &lightsample);

                aten::vec3 nmlLight = lightsample.nml;
                aten::vec3 dirToLight = normalize(lightsample.dir);

                // TODO
                // Only lambert
                auto pdf = AT_NAME::lambert::pdf(normal, dirToLight);
                auto brdf = AT_NAME::lambert::bsdf(&mtrls[self_info.mtrl_idx], albedo);
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

                if (comibined_reservoir.update(lightsample, light_pos, weight, m, r)) {
                    selected_target_density = target_density;
                }
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
#else
__global__ void computeSpatialReuse(
    idaten::Path* paths,
    const aten::LightParameter* __restrict__ lights,
    const aten::MaterialParameter* __restrict__ mtrls,
    cudaTextureObject_t* textures,
    const float4* __restrict__ aovTexclrMeshid,
    const idaten::Reservoir* __restrict__ reservoirs,
    idaten::Reservoir* dst_reservoirs,
    const idaten::ReSTIRInfo* __restrict__ infos,
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
        ctxt.lights = lights;
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

    auto& sampler = paths->sampler[idx];

    const auto& albedo_meshid = aovTexclrMeshid[idx];
    const aten::vec4 albedo(albedo_meshid.x, albedo_meshid.y, albedo_meshid.z, 1.0f);

    static const int offset_x[] = {
        -1,  0,  1,
        -1,  1,
        -1,  0,  1,
    };
    static const int offset_y[] = {
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
    for (int i = 0; i < AT_COUNTOF(offset_x); i++) {
        const auto xx = ix + offset_x[i];
        const auto yy = iy + offset_y[i];

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
                gatherMaterialInfo(
                    neightbor_mtrl,
                    &ctxt,
                    neighbor_info.mtrl_idx,
                    neighbor_info.is_voxel);

                // Check how close with neighbor pixel.
                is_acceptable = (mtrl.type == neightbor_mtrl.type)
                    && (dot(normal, neighbor_normal) >= 0.95f);

                if (is_acceptable) {
                    const auto light_pos = neighbor_reservoir.light_idx_;

                    const auto& light = ctxt.lights[light_pos];

                    sampleLight(&lightsample, &ctxt, &light, self_info.p, neighbor_normal, &sampler, 0);

                    aten::vec3 nmlLight = lightsample.nml;
                    aten::vec3 dirToLight = normalize(lightsample.dir);

                    auto pdf = samplePDF(
                        &ctxt, &neightbor_mtrl,
                        normal,
                        self_info.wi, dirToLight,
                        self_info.u, self_info.v);
                    auto brdf = sampleBSDF(
                        &ctxt, &neightbor_mtrl,
                        normal,
                        self_info.wi, dirToLight,
                        self_info.u, self_info.v,
                        albedo);
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
#endif

namespace idaten {
    std::tuple<int, int> ReSTIRPathTracing::computelReuse(
        int width, int height,
        int bounce)
    {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (m_tileDomain.w + block.x - 1) / block.x,
            (m_tileDomain.h + block.y - 1) / block.y);
#if 0
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
#elif 0
        decltype(m_pathSampler)::vector_type samplers;
        m_pathSampler.readFromDeviceToHost(samplers);

        decltype(m_lightparam)::vector_type lights;
        m_lightparam.readFromDeviceToHost(lights);

        decltype(m_mtrlparam)::vector_type mtrls;
        m_mtrlparam.readFromDeviceToHost(mtrls);

        decltype(m_aovTexclrMeshid)::vector_type aov;
        m_aovTexclrMeshid.readFromDeviceToHost(aov);

        decltype(m_reservoirs)::value_type::vector_type reservoirs;
        m_reservoirs[0].readFromDeviceToHost(reservoirs);

        decltype(m_reservoirs)::value_type::vector_type dst_reservoirs;
        dst_reservoirs.resize(reservoirs.size());

        decltype(m_restir_infos)::value_type::vector_type infos;
        m_restir_infos[0].readFromDeviceToHost(infos);

        for (int iy = 0; iy < height; iy++) {
            for (int ix = 0; ix < width; ix++) {
                computeSpatialReuse(
                    ix, iy,
                    samplers.data(),
                    lights.data(),
                    mtrls.data(),
                    aov.data(),
                    reservoirs.data(),
                    dst_reservoirs.data(),
                    infos.data(),
                    width, height);
            }
        }

        return std::make_tuple(0, 0);
#else
        if (bounce == 0) {
            computeSpatialReuse << <grid, block, 0, m_stream >> > (
                m_paths.ptr(),
                m_lightparam.ptr(),
                m_mtrlparam.ptr(),
                m_tex.ptr(),
                m_aovTexclrMeshid.ptr(),
                m_reservoirs[0].ptr(),
                m_reservoirs[1].ptr(),
                m_restir_infos[0].ptr(),
                width, height);

            checkCudaKernel(computeSpatialReuse);

            return std::make_tuple(1, 0);
        }

        return std::make_tuple(0, 0);
#endif
    }
}
