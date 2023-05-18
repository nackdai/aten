#pragma once

#include "kernel/device_scene_context.cuh"
#include "kernel/intersect.cuh"
#include "kernel/pt_params.h"

#include "aten4idaten.h"

__device__ void sampleLight(
    aten::LightSampleResult* result,
    idaten::context* ctxt,
    const aten::LightParameter* light,
    const aten::vec3& org,
    const aten::vec3& normal,
    aten::sampler* sampler,
    int32_t lod = 0);

#ifndef __AT_DEBUG__
#include "kernel/light_impl.cuh"
#endif
