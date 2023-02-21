#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"
#include "kernel/context.cuh"
#include "material/sample_texture.h"
#include "aten4idaten.h"

__device__ void sampleMaterial(
    AT_NAME::MaterialSampling* result,
    const idaten::Context* ctxt,
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& orgnormal,
    aten::sampler* sampler,
    real pre_sampled_r,
    float u, float v);

__device__ void sampleMaterial(
    AT_NAME::MaterialSampling* result,
    const idaten::Context* ctxt,
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& orgnormal,
    aten::sampler* sampler,
    real pre_sampled_r,
    float u, float v,
    const aten::vec4& externalAlbedo);

__device__ real samplePDF(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v);

__device__ aten::vec3 sampleDirection(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    real u, real v,
    aten::sampler* sampler,
    real pre_sampled_r);

__device__ aten::vec3 sampleBSDF(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v,
    real pre_sampled_r);

__device__ aten::vec3 sampleBSDF(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v,
    const aten::vec4& externalAlbedo,
    real pre_sampled_r);

__device__ real applyNormal(
    const aten::MaterialParameter* mtrl,
    const int normalMapIdx,
    const aten::vec3& orgNml,
    aten::vec3& newNml,
    real u, real v,
    const aten::vec3& wi,
    aten::sampler* sampler);

__device__ real computeFresnel(
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real outsideIor = 1);

#ifndef __AT_DEBUG__
#include "kernel/material_impl.cuh"
#endif

inline __device__ bool gatherMaterialInfo(
    aten::MaterialParameter& dst_mtrl,
    const idaten::Context* ctxt,
    const int mtrl_id,
    const bool is_voxel)
{
    bool is_valid_mtrl = mtrl_id >= 0;

    if (is_valid_mtrl) {
        dst_mtrl = ctxt->mtrls[mtrl_id];

        if (is_voxel) {
            // Replace to lambert.
            const auto& albedo = ctxt->mtrls[mtrl_id].baseColor;
            dst_mtrl = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
            dst_mtrl.baseColor = albedo;
        }
    }
    else {
        // TODO
        dst_mtrl = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
        dst_mtrl.baseColor = aten::vec3(1.0f);
    }

    return is_valid_mtrl;
}
