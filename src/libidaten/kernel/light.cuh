#pragma once

#include "kernel/context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_params.h"

#include "aten4idaten.h"

__device__ void sampleLight(
    aten::LightSampleResult* result,
    idaten::Context* ctxt,
    const aten::LightParameter* light,
    const aten::vec3& org,
    const aten::vec3& normal,
    aten::sampler* sampler,
    int lod = 0);

class ComputeBrdfFunctor {
public:
    __device__ ComputeBrdfFunctor(
        idaten::Context& ctxt,
        const aten::MaterialParameter& mtrl,
        const aten::vec3& orienting_normal,
        const aten::vec3& ray_dir,
        float u, float v,
        const aten::vec4& albedo)
        : ctxt_(ctxt), mtrl_(mtrl), orienting_normal_(orienting_normal),
        ray_dir_(ray_dir), u_(u), v_(v), albedo_(albedo) {}

    __device__ aten::vec3 operator()(const aten::vec3& dir_to_light);

private:
    idaten::Context& ctxt_;
    const aten::MaterialParameter& mtrl_;
    const aten::vec3& orienting_normal_;
    const aten::vec3& ray_dir_;
    float u_;
    float v_;
    const aten::vec4& albedo_;
};

__device__ int sampleLightWithReservoirRIP(
    aten::LightSampleResult* result,
    idaten::Reservoir& reservoir,
    aten::LightParameter* target_light,
    ComputeBrdfFunctor& compute_brdf,
    idaten::Context* ctxt,
    const aten::vec3& org,
    const aten::vec3& normal,
    aten::sampler* sampler,
    int lod = 0);

#ifndef __AT_DEBUG__
#include "kernel/compute_brdf_functor_impl.cuh"
#include "kernel/light_impl.cuh"
#endif
