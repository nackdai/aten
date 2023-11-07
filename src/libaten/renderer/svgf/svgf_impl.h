#pragma once

#include <optional>

#include "material/material.h"
#include "math/mat4.h"
#include "misc/span.h"
#include "misc/const_span.h"
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

    template <bool WillDevideColorByW = true, typename Span_v4>
    inline AT_DEVICE_MTRL_API aten::tuple<AT_NAME::_detail::v3, real, int32_t, AT_NAME::_detail::v4> ExtractCenterPixel(
        int32_t idx,
        const aten::const_span<AT_NAME::_detail::v4>& contribs,
        Span_v4& curr_aov_normal_depth,
        Span_v4& curr_aov_texclr_meshid)
    {
        const auto& nml_depth = curr_aov_normal_depth[idx];
        const auto& texclr_meshid = curr_aov_texclr_meshid[idx];

        const float center_depth = nml_depth.w;
        const int32_t center_meshid = static_cast<int32_t>(texclr_meshid.w);

        // Pixel color of this frame.
        const auto& contrib = contribs[idx];

        auto curr_color{ AT_NAME::_detail::MakeVec4<AT_NAME::_detail::v4>(contrib.x, contrib.y, contrib.z, 1.0f) };
        if constexpr (WillDevideColorByW) {
            curr_color /= contrib.w;
        }

        auto center_normal = AT_NAME::_detail::MakeVec3(nml_depth.x, nml_depth.y, nml_depth.z);

        return aten::make_tuple(center_normal, center_depth, center_meshid, curr_color);
    }

    inline AT_DEVICE_MTRL_API std::optional<AT_NAME::_detail::v4> CheckIfBackgroundPixel(
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
        aten::span<AT_NAME::_detail::v4>& curr_aov_color_variance,
        aten::span<AT_NAME::_detail::v4>& curr_aov_moment_temporalweight,
        const aten::const_span<AT_NAME::_detail::v4>& prev_aov_moment_temporalweight)
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
    inline AT_DEVICE_MTRL_API float TemporalReprojection(
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
        const aten::const_span<AT_NAME::_detail::v4>& prev_aov_normal_depth,
        const aten::const_span<AT_NAME::_detail::v4>& prev_aov_texclr_meshid,
        const aten::const_span<AT_NAME::_detail::v4>& prev_aov_color_variance,
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
                const auto prev_normal{ AT_NAME::_detail::MakeVec3(nml_depth.x, nml_depth.y, nml_depth.z) };

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

    inline AT_DEVICE_MTRL_API AT_NAME::_detail::v4 EstimateVariance(
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        const aten::mat4& mtx_C2V,
        const float camera_distance,
        const aten::const_span<AT_NAME::_detail::v4>& aov_normal_depth,
        aten::span<AT_NAME::_detail::v4>& aov_texclr_meshid,
        aten::span<AT_NAME::_detail::v4>& aov_color_variance,
        aten::span<AT_NAME::_detail::v4>& aov_moment_temporalweight)
    {
        const int32_t idx = _detail::GetIdx(ix, iy, width);

        const auto& normal_depth = aov_normal_depth[idx];
        const auto& texclr_meshid = aov_texclr_meshid[idx];
        const auto& moment_temporalweight = aov_moment_temporalweight[idx];

        const auto& center_color = aov_color_variance[idx];

        const float center_depth = aov_normal_depth[idx].w;
        const int32_t center_meshid = static_cast<int32_t>(texclr_meshid.w);

        if (center_meshid < 0) {
            // Thie pixle can be treated as the background.
            // The variance is also treated as zero.
            aov_moment_temporalweight[idx].x = 0;
            aov_moment_temporalweight[idx].y = 0;
            aov_moment_temporalweight[idx].z = 1;

            return AT_NAME::_detail::MakeVec4(0, 0, 0, 0);
        }

        const float pixel_distance_ratio = (center_depth / camera_distance) * height;

        auto center_moment{ AT_NAME::_detail::MakeVec3(moment_temporalweight.x, moment_temporalweight.y, moment_temporalweight.z) };

        int32_t frame = static_cast<int32_t>(center_moment.z);

        center_moment /= center_moment.z;

        float variance = 0.0f;
        auto color = center_color;

        if (frame < 4) {
            // 積算フレーム数が４未満 or Disoccludedされている.
            // 7x7birateral filterで輝度を計算.
            // Accumulated farme is less than 4 or the pixel is disoccluded.
            // Compute the luminance by 7x7birateral filter.

            auto center_normal{ AT_NAME::_detail::MakeVec3(normal_depth.x, normal_depth.y, normal_depth.z) };

            auto moment_sum{ AT_NAME::_detail::MakeVec3(center_moment.x, center_moment.y, center_moment.z) };
            float weight = 1.0f;

            int32_t radius = frame > 1 ? 2 : 3;

            for (int32_t v = -radius; v <= radius; v++)
            {
                for (int32_t u = -radius; u <= radius; u++)
                {
                    if (u != 0 || v != 0) {
                        int32_t xx = clamp(ix + u, 0, width - 1);
                        int32_t yy = clamp(iy + v, 0, height - 1);

                        int32_t sample_idx = _detail::GetIdx(xx, yy, width);
                        const auto& normal_depth = aov_normal_depth[sample_idx];
                        const auto& texclr_meshid = aov_texclr_meshid[sample_idx];
                        const auto& moment_temporalweight = aov_moment_temporalweight[sample_idx];

                        const auto sample_nml{ AT_NAME::_detail::MakeVec3(normal_depth.x, normal_depth.y, normal_depth.z) };
                        const float sample_depth = normal_depth.w;
                        const int32_t sample_meshid = static_cast<int32_t>(texclr_meshid.w);
                        const auto& sample_color = aov_color_variance[sample_idx];

                        auto moment{ AT_NAME::_detail::MakeVec3(moment_temporalweight.x, moment_temporalweight.y, moment_temporalweight.z) };
                        moment /= moment.z;

                        const auto uv_length = aten::sqrt(static_cast<real>(u * u + v * v));

                        const float Wz = aten::abs(sample_depth - center_depth) / (pixel_distance_ratio * uv_length + 1e-2f);
                        const float Wn = aten::pow(aten::cmpMax(0.0f, dot(sample_nml, center_normal)), 128.0f);

                        const float Wm = center_meshid == sample_meshid ? 1.0f : 0.0f;

                        const float W = exp(-Wz) * Wn * Wm;

                        moment_sum += moment * W;
                        color += sample_color * W;
                        weight += W;
                    }
                }
            }

            moment_sum /= weight;
            color /= weight;

            variance = 1.0f + 3.0f * (1.0f - frame / 4.0f) * aten::cmpMax(0.0f, moment_sum.y - moment_sum.x * moment_sum.x);
        }
        else {
            variance = aten::cmpMax(0.0f, center_moment.x - center_moment.y * center_moment.y);
        }

        color.w = variance;
        aov_color_variance[idx] = color;

        return AT_NAME::_detail::MakeVec4(variance, variance, variance, 1);
    }

    inline AT_DEVICE_MTRL_API float ExecGaussFilter3x3(
        int32_t ix, int32_t iy,
        int32_t w, int32_t h,
        const aten::const_span<AT_NAME::_detail::v4>& color_variance_buffer)
    {
        static constexpr float kernel_array[] = {
            1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
            1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
            1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
        };
        static constexpr aten::const_span<float> kernel(kernel_array);

        static constexpr int32_t offsetx_array[] = {
            -1, 0, 1,
            -1, 0, 1,
            -1, 0, 1,
        };
        static constexpr aten::const_span<int32_t> offsetx(offsetx_array);

        static constexpr int32_t offsety_array[] = {
            -1, -1, -1,
            0, 0, 0,
            1, 1, 1,
        };
        static constexpr aten::const_span<int32_t> offsety(offsety_array);

        float sum = 0;

        int32_t pos = 0;

#pragma unroll
        for (int32_t i = 0; i < 9; i++) {
            int32_t xx = clamp(ix + offsetx[i], 0, w - 1);
            int32_t yy = clamp(iy + offsety[i], 0, h - 1);

            int32_t idx = _detail::GetIdx(xx, yy, w);

            float tmp = color_variance_buffer[idx].w;

            sum += kernel[pos] * tmp;

            pos++;
        }

        return sum;
    }

    inline AT_DEVICE_MTRL_API real ComputeGaussFiltereredVariance(
        const bool is_first_iter,
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        const aten::const_span<AT_NAME::_detail::v4>& aov_color_variance,
        const aten::const_span<AT_NAME::_detail::v4>& color_variance_buffer)
    {
        // 3x3 Gauss filter.
        float gauss_filtered_variance;

        if (is_first_iter) {
            gauss_filtered_variance = ExecGaussFilter3x3(ix, iy, width, height, aov_color_variance);
        }
        else {
            gauss_filtered_variance = ExecGaussFilter3x3(ix, iy, width, height, color_variance_buffer);
        }

        return gauss_filtered_variance;
    }

    inline AT_DEVICE_MTRL_API std::optional<std::optional<AT_NAME::_detail::v4>> CheckIfBackgroundPixelForAtrous(
        const bool is_final_iter,
        const int32_t idx,
        const int32_t center_meshid,
        const AT_NAME::_detail::v4& center_color,
        const aten::const_span<AT_NAME::_detail::v4>& aov_texclr_meshid,
        aten::span< AT_NAME::_detail::v4>& next_color_variance_buffer)
    {
        std::optional<AT_NAME::_detail::v4> result;

        if (center_meshid < 0) {
            // If mesh id is negative, the pixel is the background.
            // So, just output the background pixel.
            next_color_variance_buffer[idx] = AT_NAME::_detail::MakeVec4(center_color.x, center_color.y, center_color.z, 0.0f);

            if (is_final_iter) {
                // In the finaly iteration, apply albedo.
                auto texclr_meshid{ aov_texclr_meshid[idx] };
                // NOTE:
                // center_color is constant. So, multiply to texclr_meshid, and return it as the albedo applied color.
                texclr_meshid *= center_color;
                result = std::make_optional(texclr_meshid);
            }

            return result;
        }

        return std::nullopt;
    }

    inline AT_DEVICE_MTRL_API AT_NAME::_detail::v4 ExecAtrousWaveletFilter(
        bool is_first_iter, bool is_final_iter,
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        float gauss_filtered_variance,
        const AT_NAME::_detail::v3& center_normal,
        const float center_depth,
        const int32_t center_meshid,
        const AT_NAME::_detail::v4& center_color,
        const aten::const_span<AT_NAME::_detail::v4>& aov_normal_depth,
        const aten::const_span<AT_NAME::_detail::v4>& aov_texclr_meshid,
        const aten::const_span<AT_NAME::_detail::v4>& aov_color_variance,
        const aten::const_span<AT_NAME::_detail::v4>& color_variance_buffer,
        const int32_t step_scale,
        const float camera_distance)
    {
        static constexpr float sigmaZ = 1.0f;
        static constexpr float sigmaN = 128.0f;
        static constexpr float sigmaL = 4.0f;

        static constexpr float h_array[] = {
            2.0 / 3.0,  2.0 / 3.0,  2.0 / 3.0,  2.0 / 3.0,
            1.0 / 6.0,  1.0 / 6.0,  1.0 / 6.0,  1.0 / 6.0,
            4.0 / 9.0,  4.0 / 9.0,  4.0 / 9.0,  4.0 / 9.0,
            1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
            1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        };
        static constexpr aten::const_span<float> h(h_array);

        static constexpr int32_t offsetx_array[] = {
            1,  0, -1, 0,
            2,  0, -2, 0,
            1, -1, -1, 1,
            1, -1, -1, 1,
            2, -2, -2, 2,
            2, -2, -2, 2,
        };
        static constexpr aten::const_span<int32_t> offsetx(offsetx_array);

        static constexpr int32_t offsety_array[] = {
            0, 1,  0, -1,
            0, 2,  0, -2,
            1, 1, -1, -1,
            2, 2, -2, -2,
            1, 1, -1, -1,
            2, 2, -2, -2,
        };
        static constexpr aten::const_span<int32_t> offsety(offsety_array);

        const int32_t idx = _detail::GetIdx(ix, iy, width);

        float sqrt_gauss_filtered_variance = sqrt(gauss_filtered_variance);

        auto center_luminance = AT_NAME::color::luminance(center_color.x, center_color.y, center_color.z);
        auto sumC = center_color;
        auto sumV = center_color.w;
        auto weight = 1.0f;

        int32_t pos = 0;

        const auto pixel_distance_ratio = (center_depth / camera_distance) * height;

#pragma unroll
        for (int32_t i = 0; i < 24; i++)
        {
            int32_t scaled_offset_x = offsetx[i] * step_scale;
            int32_t scaled_offset_y = offsety[i] * step_scale;

            int32_t xx = clamp(ix + scaled_offset_x, 0, width - 1);
            int32_t yy = clamp(iy + scaled_offset_y, 0, height - 1);

            const auto u_length = aten::sqrt(scaled_offset_x * scaled_offset_x + scaled_offset_y * scaled_offset_y);

            const int32_t qidx = _detail::GetIdx(xx, yy, width);

            const auto& normal_depth = aov_normal_depth[qidx];
            const auto& texclr_meshid = aov_texclr_meshid[qidx];

            const auto normal{ AT_NAME::_detail::MakeVec3(normal_depth.x, normal_depth.y, normal_depth.z) };

            const auto depth = normal_depth.w;
            const int32_t meshid = static_cast<int32_t>(texclr_meshid.w);

            const auto& color = is_first_iter ? aov_color_variance[qidx] : color_variance_buffer[qidx];
            const auto variance = color.w;

            float lum = AT_NAME::color::luminance(color.x, color.y, color.z);

            float Wz = 3.0f * fabs(center_depth - depth) / (pixel_distance_ratio * u_length + 0.000001f);

            float Wn = powf(aten::cmpMax(0.0f, dot(center_normal, normal)), sigmaN);

            float Wl = aten::cmpMin(expf(-fabs(center_luminance - lum) / (sigmaL * sqrt_gauss_filtered_variance + 0.000001f)), 1.0f);

            float Wm = meshid == center_meshid ? 1.0f : 0.0f;

            float W = expf(-Wl * Wl - Wz) * Wn * Wm * h[i];

            sumC += W * color;
            sumV += W * W * variance;

            weight += W;

            pos++;
        }

        sumC /= weight;
        sumV /= (weight * weight);

        return AT_NAME::_detail::MakeVec4(sumC.x, sumC.y, sumC.z, sumV);
    }

    inline AT_DEVICE_MTRL_API std::optional<AT_NAME::_detail::v4> PostProcessForAtrousFilter(
        bool is_first_iter, bool is_final_iter,
        const int32_t idx,
        const AT_NAME::_detail::v4& filtered_color_variance,
        const aten::const_span<AT_NAME::_detail::v4>& aov_texclr_meshid,
        aten::span< AT_NAME::_detail::v4>& temporary_color_buffer,
        aten::span< AT_NAME::_detail::v4>& next_color_variance_buffer)
    {
        next_color_variance_buffer[idx] = filtered_color_variance;

        if (is_first_iter) {
            // Store color temporarily.
            temporary_color_buffer[idx].x = filtered_color_variance.x;
            temporary_color_buffer[idx].y = filtered_color_variance.y;
            temporary_color_buffer[idx].z = filtered_color_variance.z;
        }

        if (is_final_iter) {
            // In the finaly iteration, apply albedo.
            auto texclr_meshid{ aov_texclr_meshid[idx] };

            // NOTE:
            // filtered_color_variance is constant. So, multiply to texclr_meshid, and return it as the albedo applied color.
            texclr_meshid *= filtered_color_variance;
            return texclr_meshid;
        }
        return std::nullopt;
    }
}   // namespace svgf
}   // namespace AT_NAME
