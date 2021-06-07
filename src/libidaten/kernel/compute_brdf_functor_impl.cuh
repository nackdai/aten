#pragma once

#include "kernel/idatendefs.cuh"
#include "kernel/material.cuh"

AT_CUDA_INLINE __device__ aten::vec3 ComputeBrdfFunctor::operator()(const aten::vec3& dir_to_light) {
    return sampleBSDF(
        &ctxt_, &mtrl_, orienting_normal_, ray_dir_, dir_to_light, u_, v_, albedo_);
}
