#pragma once

#include <optional>

#include "material/material.h"
#include "math/mat4.h"
#include "misc/span.h"
#include "misc/tuple.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/pathtracing/pt_params.h"

namespace AT_NAME {
namespace svgf {
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

        inline AT_DEVICE_MTRL_API int32_t GetIdx(int32_t x, int32_t y, int32_t pitch)
        {
            return x + y * pitch;
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
        aten::span<AT_NAME::_detail::v4>& aov_normal_depth,
        aten::span<AT_NAME::_detail::v4>& aov_texclr_meshid)
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

    template <bool IsFirstFrameExecution>
    inline AT_DEVICE_MTRL_API AT_NAME::_detail::v4 PrepareForDenoise(
        const int32_t idx,
        const AT_NAME::Path& paths,
        aten::span<AT_NAME::_detail::v4> temporary_color_buffer,
        aten::span<AT_NAME::_detail::v4> aov_color_variance = nullptr,
        aten::span<AT_NAME::_detail::v4> aov_moment_temporalweight = nullptr)
    {
        const auto& c = paths.contrib[idx].v;

        AT_NAME::_detail::v4 contrib;
        AT_NAME::_detail::CopyVec(contrib, c);
        contrib /= c.w; // w is number of sample.

        if constexpr (IsFirstFrameExecution) {
            float lum = AT_NAME::color::luminance(contrib.x, contrib.y, contrib.z);

            aov_moment_temporalweight[idx].x += lum * lum;
            aov_moment_temporalweight[idx].y += lum;
            aov_moment_temporalweight[idx].z += 1;

            aov_color_variance[idx] = AT_NAME::_detail::MakeVec4(
                contrib.x, contrib.y, contrib.z, aov_color_variance[idx].w);
        }

        // In order not to chnage the values in paths for the next step, keep color in another buffer.
        temporary_color_buffer[idx] = c;

        return contrib;
    }

    inline AT_DEVICE_MTRL_API aten::tuple<AT_NAME::_detail::v3, real, int32_t, AT_NAME::_detail::v4> ExtractCenterPixel(
        int32_t idx,
        const aten::span<AT_NAME::_detail::v4>& contribs,
        const aten::span<AT_NAME::_detail::v4>& curr_aov_normal_depth,
        const aten::span<AT_NAME::_detail::v4>& curr_aov_texclr_meshid)
    {
        auto nml_depth = curr_aov_normal_depth[idx];
        auto texclr_meshid = curr_aov_texclr_meshid[idx];

        const float center_depth = nml_depth.w;
        const int32_t center_meshid = static_cast<int32_t>(texclr_meshid.w);

        // Pixel color of this frame.
        const auto& contrib = contribs[idx];

        auto curr_color = AT_NAME::_detail::MakeVec4<AT_NAME::_detail::v4>(contrib.x, contrib.y, contrib.z, 1.0f);
        curr_color /= contrib.w;

        auto center_normal = AT_NAME::_detail::MakeVec3(nml_depth.x, nml_depth.y, nml_depth.z);

        return aten::make_tuple(center_normal, center_depth, center_meshid, curr_color);
    }

    inline AT_DEVICE_MTRL_API std::optional<AT_NAME::_detail::v4> CheckIfPixelIsBackground(
        const int32_t idx,
        const AT_NAME::_detail::v4& curr_color,
        const int32_t center_meshid,
        aten::span<AT_NAME::_detail::v4>& curr_aov_color_variance,
        aten::span<AT_NAME::_detail::v4>& curr_aov_moment_temporalweight)
    {
        if (center_meshid < 0) {
            // This case can be treated as background.
            curr_aov_color_variance[idx] = curr_color;

            curr_aov_moment_temporalweight[idx] = AT_NAME::_detail::MakeVec4(1, 1, 1, curr_aov_moment_temporalweight[idx].w);

            return curr_color;
        }

        return std::nullopt;
    }

    inline AT_DEVICE_MTRL_API void AccumulateMoments(
        const int32_t idx,
        const float weight,
        const aten::span<AT_NAME::_detail::v4>& curr_aov_color_variance,
        aten::span<AT_NAME::_detail::v4>& curr_aov_moment_temporalweight,
        const aten::span<AT_NAME::_detail::v4>& prev_aov_moment_temporalweight)
    {
        const auto& color_variance = curr_aov_color_variance[idx];
        auto curr_color = AT_NAME::_detail::MakeVec3(color_variance.x, color_variance.y, color_variance.z);

        // TODO
        // 現フレームと過去フレームが同率で加算されるため、どちらかに強い影響がでると影響が弱まるまでに非常に時間がかかる.
        // Average of the current frame and the past frames is computed with the same ratio.
        // So, if either is significantly larger than the others, it might takes a very long time to attenuate its effect.
        // ex)
        // f0 = 100, f1 = 0, f2 = 0
        // avg = (f0 + f1 + f2) / 3 = 33.3 <- 非常に大きい値が残り続ける.

        // accumulate moments.
        float lum = AT_NAME::color::luminance(curr_color.x, curr_color.y, curr_color.z);
        auto center_moment = AT_NAME::_detail::MakeVec3(lum * lum, lum, 0);

        // 積算フレーム数のリセット.
        // Reset the accumurate frame count.
        int32_t frame = 1;

        if (weight > 0.0f) {
            auto moment_temporalweight = prev_aov_moment_temporalweight[idx];;
            auto prev_moment = AT_NAME::_detail::MakeVec3(moment_temporalweight.x, moment_temporalweight.y, moment_temporalweight.z);

            // 積算フレーム数を１増やす.
            // Advance the accumurate frame count.
            frame = static_cast<int32_t>(prev_moment.z + 1);

            center_moment += prev_moment;
        }

        center_moment.z = static_cast<float>(frame);

        curr_aov_moment_temporalweight[idx].x = center_moment.x;
        curr_aov_moment_temporalweight[idx].y = center_moment.y;
        curr_aov_moment_temporalweight[idx].z = center_moment.z;
    }

    template <typename BufferForMotionDepth>
    inline float AT_DEVICE_MTRL_API TemporalReprojection(
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        const float threshold_normal,
        const float threshold_depth,
        const AT_NAME::_detail::v3& center_normal,
        const float center_depth,
        const int32_t center_meshid,
        AT_NAME::_detail::v4& curr_color,
        aten::span<AT_NAME::_detail::v4>& curr_aov_color_variance,
        aten::span<AT_NAME::_detail::v4>& curr_aov_moment_temporalweight,
        const aten::span<AT_NAME::_detail::v4>& prev_aov_normal_depth,
        const aten::span<AT_NAME::_detail::v4>& prev_aov_texclr_meshid,
        const aten::span<AT_NAME::_detail::v4>& prev_aov_color_variance,
        BufferForMotionDepth& motion_detph_buffer)
    {
        const auto idx = _detail::GetIdx(ix, iy, width);

        auto sum = AT_NAME::_detail::MakeVec4(0, 0, 0, 0);
        float weight = 0.0f;

#pragma unroll
        for (int32_t y = -1; y <= 1; y++) {
            for (int32_t x = -1; x <= 1; x++) {
                int32_t xx = clamp(ix + x, 0, static_cast<int32_t>(width - 1));
                int32_t yy = clamp(iy + y, 0, static_cast<int32_t>(height - 1));

                AT_NAME::_detail::v4 motion_depth;
                if constexpr (std::is_class_v<std::remove_reference_t<decltype(motion_detph_buffer)>>) {
                    motion_depth = motion_detph_buffer[idx];
                }
                else {
                    surf2Dread(&motion_depth, motion_detph_buffer, ix * sizeof(motion_depth), iy);
                }

                // 前のフレームのスクリーン座標.
                // Screen position for the previous frame.
                int32_t prev_x = static_cast<int32_t>(xx + motion_depth.x * width);
                int32_t prev_y = static_cast<int32_t>(yy + motion_depth.y * height);

                prev_x = clamp(prev_x, 0, static_cast<int32_t>(width - 1));
                prev_y = clamp(prev_y, 0, static_cast<int32_t>(height - 1));

                int32_t prev_idx = _detail::GetIdx(prev_x, prev_y, width);

                const auto& nml_depth = prev_aov_normal_depth[prev_idx];
                const auto& texclr_meshid = prev_aov_texclr_meshid[prev_idx];

                const float prev_depth = nml_depth.w;
                const int32_t prev_meshid = (int32_t)texclr_meshid.w;
                auto prev_normal = AT_NAME::_detail::MakeVec3(nml_depth.x, nml_depth.y, nml_depth.z);

                // TODO
                // 同じメッシュ上でもライトのそばの明るくなったピクセルを拾ってしまう場合の対策が必要.
                // Even if the picked pixels are on the same mesh, the radiance of each pixel might be very different.
                // The countermeaure for such kind of the situation might necessary.

                float Wz = clamp((threshold_depth - abs(1 - center_depth / prev_depth)) / threshold_depth, 0.0f, 1.0f);
                float Wn = clamp((dot(center_normal, prev_normal) - threshold_normal) / (1.0f - threshold_normal), 0.0f, 1.0f);
                float Wm = center_meshid == prev_meshid ? 1.0f : 0.0f;

                // 前のフレームのピクセルカラーを取得.
                // Pixel color of the previous frame.
                const auto& prev_color = prev_aov_color_variance[prev_idx];

                float W = Wz * Wn * Wm;
                sum += prev_color * W;
                weight += W;
            }
        }

        if (weight > 0.0f) {
            sum /= weight;
            weight /= 9;

            curr_color = 0.2f * curr_color + 0.8f * sum;
        }

        curr_aov_moment_temporalweight[idx].w = weight;

        curr_aov_color_variance[idx].x = curr_color.x;
        curr_aov_color_variance[idx].y = curr_color.y;
        curr_aov_color_variance[idx].z = curr_color.z;

        return weight;
    }
}   // namespace svgf
}   // namespace AT_NAME
