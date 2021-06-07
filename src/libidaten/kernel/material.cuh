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
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& orgnormal,
    aten::sampler* sampler,
    float u, float v);

__device__ void sampleMaterial(
    AT_NAME::MaterialSampling* result,
    const idaten::Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& orgnormal,
    aten::sampler* sampler,
    float u, float v,
    const aten::vec4& externalAlbedo);

__device__ real samplePDF(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v);

__device__ aten::vec3 sampleDirection(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    real u, real v,
    aten::sampler* sampler);

__device__ aten::vec3 sampleBSDF(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v);

__device__ aten::vec3 sampleBSDF(
    const idaten::Context* ctxt,
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real u, real v,
    const aten::vec4& externalAlbedo);


__device__ real computeFresnel(
    const aten::MaterialParameter* mtrl,
    const aten::vec3& normal,
    const aten::vec3& wi,
    const aten::vec3& wo,
    real outsideIor = 1);

#ifndef __AT_DEBUG__
#include "kernel/material_impl.cuh"
#endif
