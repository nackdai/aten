#pragma once

#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"
#include "cuda/cudamemory.h"
#include "kernel/context.cuh"

inline __device__ bool gatherMaterialInfo(
    aten::MaterialParameter& dst_mtrl,
    const idaten::context* ctxt,
    const int32_t mtrl_id,
    const bool is_voxel)
{
    bool is_valid_mtrl = mtrl_id >= 0;

    if (is_valid_mtrl) {
        dst_mtrl = ctxt->GetMaterial(static_cast<uint32_t>(mtrl_id));

        if (is_voxel) {
            // Replace to lambert.
            const auto& albedo = ctxt->GetMaterial(static_cast<uint32_t>(mtrl_id)).baseColor;
            dst_mtrl = aten::MaterialParameter(aten::MaterialType::Lambert, aten::MaterialAttributeLambert);
            dst_mtrl.baseColor = albedo;
        }

        dst_mtrl.albedoMap = (int)(dst_mtrl.albedoMap >= 0 ? ctxt->textures[dst_mtrl.albedoMap] : -1);
        dst_mtrl.normalMap = (int)(dst_mtrl.normalMap >= 0 ? ctxt->textures[dst_mtrl.normalMap] : -1);
        dst_mtrl.roughnessMap = (int)(dst_mtrl.roughnessMap >= 0 ? ctxt->textures[dst_mtrl.roughnessMap] : -1);
    }
    else {
        // TODO
        dst_mtrl = aten::MaterialParameter(aten::MaterialType::Lambert, aten::MaterialAttributeLambert);
        dst_mtrl.baseColor = aten::vec3(1.0f);
    }

    return is_valid_mtrl;
}
