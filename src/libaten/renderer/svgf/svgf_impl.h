#pragma once

#include "material/material.h"
#include "math/mat4.h"
#include "misc/span.h"
#include "renderer/pathtracing/pathtracing_impl.h"

namespace AT_NAME
{
    namespace _detail
    {
        template <bool NeedCheckSingularMtrlBounce>
        inline AT_DEVICE_MTRL_API bool NeedFillAOVBySingularMtrl(
            const int32_t idx,
            const int32_t bounce,
            const AT_NAME::Path& paths)
        {
            if constexpr (NeedCheckSingularMtrlBounce) {
                //return bounce == 1 && paths.attrib[idx].mtrlType == aten::MaterialType::Specular;
                return bounce == 1 && paths.attrib[idx].isSingular;
            }
            return false;
        }
    }

    template <bool NeedCheckSingularMtrlBounce, bool NeedOverrideMeshIdByMtrlId, bool IsExternalAlbedo>
    inline AT_DEVICE_MTRL_API bool FillAOVs(
        const int32_t idx,
        const int32_t bounce,
        const AT_NAME::Path& paths,
        const aten::hitrecord& rec,
        const aten::Intersection& isect,
        const aten::mat4& mtxW2C,
        const aten::vec3& normal,
        aten::MaterialParameter& mtrl,
        aten::span<aten::vec4>& aov_normal_depth,
        aten::span<aten::vec4>& aov_texclr_meshid)
    {
        if (bounce == 0
            || _detail::NeedFillAOVBySingularMtrl<NeedCheckSingularMtrlBounce>(idx, bounce, paths))
        {
            // texture color
            auto texcolor = AT_NAME::sampleTexture(mtrl.albedoMap, rec.u, rec.v, aten::vec4(1.0f));

            AT_NAME::FillBasicAOVs(
                aov_normal_depth[idx], normal, rec, mtxW2C,
                aov_texclr_meshid[idx], texcolor, isect);

            if constexpr (NeedOverrideMeshIdByMtrlId) {
                aov_texclr_meshid[idx].w = static_cast<real>(isect.mtrlid);
            }

            if constexpr (IsExternalAlbedo) {
                // For exporting separated albedo.
                mtrl.albedoMap = -1;
            }
            return true;
        }

        return false;
    }
}
