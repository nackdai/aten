#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"
#include "kernel/context.cuh"
#include "aten4idaten.h"

__device__ void sampleMaterial(
    AT_NAME::MaterialSampling* result,
    const idaten::Context* ctxt,
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& orgnormal,
    aten::sampler* sampler,
    float u, float v);

__device__ void sampleMaterial(
    AT_NAME::MaterialSampling* result,
    const idaten::Context* ctxt,
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& orgnormal,
    aten::sampler* sampler,
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
    aten::sampler* sampler);

__device__ aten::vec3 sampleBSDF(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v);

__device__ aten::vec3 sampleBSDF(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v,
    const aten::vec4& externalAlbedo);


__device__ real computeFresnel(
    const aten::MaterialParameter* dst_mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real outsideIor = 1);

#ifndef __AT_DEBUG__
#include "kernel/material_impl.cuh"
#endif

inline __device__ void gatherMaterialInfo(
    aten::MaterialParameter& dst_mtrl,
    const idaten::Context* ctxt,
    const int mtrl_id,
    const bool is_voxel)
{
    if (mtrl_id >= 0) {
        dst_mtrl = ctxt->mtrls[mtrl_id];

        if (is_voxel) {
            // Replace to lambert.
            const auto& albedo = ctxt->mtrls[mtrl_id].baseColor;
            dst_mtrl = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
            dst_mtrl.baseColor = albedo;
        }

        if (dst_mtrl.type != aten::MaterialType::Layer) {
            dst_mtrl.albedoMap = (int)(dst_mtrl.albedoMap >= 0 ? ctxt->textures[dst_mtrl.albedoMap] : -1);
            dst_mtrl.normalMap = (int)(dst_mtrl.normalMap >= 0 ? ctxt->textures[dst_mtrl.normalMap] : -1);
            dst_mtrl.roughnessMap = (int)(dst_mtrl.roughnessMap >= 0 ? ctxt->textures[dst_mtrl.roughnessMap] : -1);
        }
    }
    else {
        // TODO
        dst_mtrl = aten::MaterialParameter(aten::MaterialType::Lambert, MaterialAttributeLambert);
        dst_mtrl.baseColor = aten::vec3(1.0f);
    }
}
