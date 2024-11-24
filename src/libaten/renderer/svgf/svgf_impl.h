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
        inline AT_DEVICE_API bool NeedFillAOVBySingularMtrl(
            const int32_t idx,
            const int32_t bounce,
            const AT_NAME::Path& paths)
        {
            if constexpr (NeedCheckSingularMtrlBounce) {
                //return bounce == 1 && paths.attrib[idx].mtrlType == aten::MaterialType::Specular;
                return bounce == 1 && paths.attrib[idx].is_singular;
            }
            return false;
        }

        inline AT_DEVICE_API int32_t GetIdx(int32_t x, int32_t y, int32_t pitch)
        {
            return x + y * pitch;
        }
    }

    /**
     * @brief Fill AOV buffers.
     *
     * @tparam NeedCheckSingularMtrlBounce Whether the case that the material is singular is checked.
     * @tparam NeedOverrideMeshIdByMtrlId Whether the mesh id is overwritten by the material id in AOV buffer.
     * @tparam IsExternalAlbedo Whether albedo map texture id in the material parameter is reset with -1.
     * @param[in] idx Index to the pixel.
     * @param[in] bounce Count of bounce.
     * @param[in] paths Information of paths.
     * @param[in] isect Scene intersection information.
     * @param[in] mtxW2C Matrix to compute from world cooridnate to clip coordinate.
     * @param[in] normal Normal to be stored as AOV.
     * @param[in,out] mtrl Material parameter at hit point. If IsExternalAlbedo is true, mtrl.albedoMap is reset with -1.
     * @param[out] aov_normal_depth Destination buffer to store normal and depth.
     * @param[out] aov_texclr_meshid Destination buffer to store albedo color and mesh id.
     * @return If AOV is filled in this API, returns true. Otherwise, returns false.
     */
    template <bool NeedCheckSingularMtrlBounce, bool NeedOverrideMeshIdByMtrlId, bool IsExternalAlbedo>
    inline AT_DEVICE_API bool FillAOVs(
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
                aov_texclr_meshid[idx].w = static_cast<float>(isect.mtrlid);
            }

            if constexpr (IsExternalAlbedo) {
                // For exporting separated albedo.
                mtrl.albedoMap = -1;
            }
            return true;
        }

        return false;
    }

    /**
     * @brief Prepare to execute denoise.
     *
     * @tparam IsFirstFrameExecution Whether this API is executed in the first frame count.
     * @param[in] idx Index to the pixel.
     * @param[in] paths Information of paths.
     * @param[out] temporary_color_buffer Destination buffer to store the processed color.
     * @param[out] aov_color_variance Destination buffer to store contribution and variance. If IsFirstFrameExecution is false, this buffer isn't used.
     * @param[out] aov_moment_temporalweight Destination buffer to store moment of color luminance and temporal weight. If IsFirstFrameExecution is false, this buffer isn't used.
     * @return Contribution as RGB.
     */
    template <bool IsFirstFrameExecution>
    inline AT_DEVICE_API AT_NAME::_detail::v4 PrepareForDenoise(
        const int32_t idx,
        const AT_NAME::Path& paths,
        aten::span<AT_NAME::_detail::v4>& temporary_color_buffer,
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
            aov_moment_temporalweight[idx].z += 1;  // frame count,

            aov_color_variance[idx] = AT_NAME::_detail::MakeVec4(
                contrib.x, contrib.y, contrib.z, aov_color_variance[idx].w);
        }

        // NOTE:
        // Path.contrib just store the path tracing contribution. It doesn't store the filtered color.
        // Therefore, we have another buffer to store the fitered color. It's temporary_color_buffer in this API.
        temporary_color_buffer[idx] = c;

        return contrib;
    }

    /**
     * @brief Extract center pixel data from AOB buffers.
     *
     * @tparam WillDivideContribByW Whether contribution value will be divided bt its w element value.
     * @tparam Span_v4 Type to specify span type which contains 4 elements vector type. This is for accepting both const span type or non-const span type.
     * @param idx Index to the pixel.
     * @param[in] contribs Buffer of contribution.
     * @param[in] aov_normal_depth AOV buffer to store normal and depth.
     * @param[in] aov_texclr_meshid AOV buffer to store albedo color and mesh id.
     * @return Tuple for the excracted center pixle data, normal, depth, mesh id, contribution.
     */
    template <bool WillDivideContribByW = true, class Span_v4>
    inline AT_DEVICE_API aten::tuple<AT_NAME::_detail::v3, float, int32_t, AT_NAME::_detail::v4> ExtractCenterPixel(
        int32_t idx,
        const aten::const_span<AT_NAME::_detail::v4>& contribs,
        Span_v4& aov_normal_depth,
        Span_v4& aov_texclr_meshid)
    {
        const auto& nml_depth = aov_normal_depth[idx];
        const auto& texclr_meshid = aov_texclr_meshid[idx];

        const float center_depth = nml_depth.w;
        const int32_t center_meshid = static_cast<int32_t>(texclr_meshid.w);

        // Pixel color of this frame.
        const auto& contrib = contribs[idx];

        auto curr_color{ AT_NAME::_detail::MakeVec4<AT_NAME::_detail::v4>(contrib.x, contrib.y, contrib.z, 1.0f) };
        if constexpr (WillDivideContribByW) {
            curr_color /= contrib.w;
        }

        auto center_normal = AT_NAME::_detail::MakeVec3(nml_depth.x, nml_depth.y, nml_depth.z);

        return aten::make_tuple(center_normal, center_depth, center_meshid, curr_color);
    }

    /**
     * @brief Update AOV buffer if the pixle is background.
     *
     * @param[in] idx Index to the pixel.
     * @param[in] color Pixel color.
     * @param[in] center_meshid Mesh id at the pixel.
     * @param[out] aov_color_variance Destination buffer to store contribution and variance. If the pixle is background, only contribution is updated with color.
     * @param[out] aov_moment_temporalweight Destination buffer to store moment of color luminance and temporal weight. If the pixle is background, only moments is updated with 1.
     * @return If the pixel is background, returns the color in the arguments directly. Otherwise, returns nullopt.
     */
    inline AT_DEVICE_API std::optional<AT_NAME::_detail::v4> UpdateAOVIfBackgroundPixel(
        const int32_t idx,
        const AT_NAME::_detail::v4& color,
        const int32_t center_meshid,
        aten::span<AT_NAME::_detail::v4>& aov_color_variance,
        aten::span<AT_NAME::_detail::v4>& aov_moment_temporalweight)
    {
        if (center_meshid < 0) {
            // This case can be treated as background.
            aov_color_variance[idx] = color;

            aov_moment_temporalweight[idx] = AT_NAME::_detail::MakeVec4(1, 1, 1, aov_moment_temporalweight[idx].w);

            return color;
        }

        return std::nullopt;
    }

    /**
     * @brief Accumulate moments.
     *
     * @param[in] idx Index to the pixel.
     * @param[in] weight Weight which is computed in temporal reprojection.
     * @param[in] curr_aov_color_variance AOV buffer to store contribution and variance for current frame.
     * @param[out] curr_aov_moment_temporalweight AOV buffer to store moments and temporal weight for current frame.
     * @param[in] prev_aov_moment_temporalweight AOV buffer to store moments and temporal weight for previous frame.
     */
    inline AT_DEVICE_API void AccumulateMoments(
        const int32_t idx,
        const float weight,
        const aten::const_span<AT_NAME::_detail::v4>& curr_aov_color_variance,
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
        // avg = (f0 + f1 + f2) / 3 = 33.3 <- 非常に大きい値が残り続ける. The effect of the large value continues to be remaining.

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
            // Advance the frame count to accumulate.
            frame = static_cast<int32_t>(prev_moment.z + 1);

            center_moment += prev_moment;
        }

        center_moment.z = static_cast<float>(frame);

        curr_aov_moment_temporalweight[idx].x = center_moment.x;
        curr_aov_moment_temporalweight[idx].y = center_moment.y;
        curr_aov_moment_temporalweight[idx].z = center_moment.z;
    }

    /**
     * @brief Compute temporal reprojection.
     *
     * @tparam BufferForMotionDepth Type of motion depth buffer.
     * @param[in] ix X position to the pixel in the screen coordinate.
     * @param[in] iy Y position to the pixel in the screen coordinate.
     * @param[in] width Screen width.
     * @param[in] height Screen height.
     * @param[in] threshold_normal Threhold to accept the different for the inner product of normals.
     * @param[in] threshold_depth Threhold to accept the different for depth values.
     * @param[in] center_normal Normal at the center pixel. The center pixel locates at (ix, iy).
     * @param[in] center_depth Depth at the center pixel. The center pixel locates at (ix, iy).
     * @param[in] center_meshid Mesh id at the center pixel. The center pixel locates at (ix, iy).
     * @param[in] center_color Color at the center pixel.
     * @param[out] curr_aov_color_variance AOV buffer to store contribution and variance for current frame.
     * @param[out] curr_aov_moment_temporalweight AOV buffer to store moments and temporal weight for current frame.
     * @param[in] prev_aov_normal_depth AOV buffer to store normal and depth for previous frame.
     * @param[in] prev_aov_texclr_meshid AOV buffer to store albedo color and mesh id for previous frame.
     * @param[in] prev_aov_color_variance AOV buffer to store contribution and variance for previous frame.
     * @param[in] motion_detph_buffer Motion depth buffer.
     * @return Tuple for weight to merge the colors and the reporjected color.
     */
    template <class BufferForMotionDepth>
    inline AT_DEVICE_API aten::tuple<float, AT_NAME::_detail::v4> TemporalReprojection(
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        const float threshold_normal,
        const float threshold_depth,
        const AT_NAME::_detail::v3& center_normal,
        const float center_depth,
        const int32_t center_meshid,
        const AT_NAME::_detail::v4& center_color,
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
        auto curr_color{ center_color };

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
                // For example, a pixel is affected by a light but others are not.
                // The countermeaure for such kind of the situation might be necessary.

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

        return aten::make_tuple(weight, curr_color);
    }

    /**
     * @brief Recompute temporal weight from the 3x3 surrounding pixels.
     *
     * @param[in] ix X position to the pixel in the screen coordinate.
     * @param[in] iy Y position to the pixel in the screen coordinate.
     * @param[in] width Screen width.
     * @param[in] height Screen height.
     * @param[in] aov_texclr_meshid AOV buffer to store albedo color and mesh id,
     * @param[in] ov_moment_temporalweight AOV buffer to store moments and temporal weight.
     * @return Temporal weight which is affected with the propagation from the surrounding pixels. If the center pixle is background, returns nullopt.
     */
    inline AT_DEVICE_API std::optional<float> RecomputeTemporalWeightFromSurroundingPixels(
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        const aten::const_span<AT_NAME::_detail::v4>& aov_texclr_meshid,
        const aten::const_span<AT_NAME::_detail::v4>& aov_moment_temporalweight)
    {
        const auto idx = _detail::GetIdx(ix, iy, width);

        const int32_t center_meshId = static_cast<int32_t>(aov_texclr_meshid[idx].w);

        if (center_meshId < 0) {
            // This pixel is background, so nothing is done.
            return std::nullopt;
        }

        float temporal_weight = aov_moment_temporalweight[idx].w;

        for (int32_t y = -1; y <= 1; y++) {
            for (int32_t x = -1; x <= 1; x++) {
                int32_t xx = ix + x;
                int32_t yy = iy + y;

                if ((0 <= xx) && (xx < width)
                    && (0 <= yy) && (yy < height))
                {
                    int32_t pidx = _detail::GetIdx(xx, yy, width);
                    float w = aov_moment_temporalweight[pidx].w;
                    temporal_weight = aten::min(temporal_weight, w);
                }
            }
        }

        return temporal_weight;
    }

    /**
     * @brief Estimate variance.
     *
     * @param[in] ix X position to the pixel in the screen coordinate.
     * @param[in] iy Y position to the pixel in the screen coordinate.
     * @param[in] width Screen width.
     * @param[in] height Screen height.
     * @param[in] mtx_C2V Matrix to tranfrom from clip coordinate to view coordinate.
     * @param[in] aov_normal_depth AOV buffer to store normal and depth.
     * @param[in] aov_texclr_meshid AOV buffer to store albedo color and mesh id.
     * @param[in,out] aov_color_variance AOV buffer to store contribution and variance.
     * @param[in,out] aov_moment_temporalweight AOV buffer to store moments and temporal weight.
     * @return Estivated variance. If the pixel is background, returns zero vector.
     */
    inline AT_DEVICE_API AT_NAME::_detail::v4 EstimateVariance(
        const int32_t ix, const int32_t iy,
        const int32_t width, const int32_t height,
        const aten::mat4& mtx_C2V,
        const float camera_distance,
        const aten::const_span<AT_NAME::_detail::v4>& aov_normal_depth,
        const aten::const_span<AT_NAME::_detail::v4>& aov_texclr_meshid,
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

        AT_ASSERT(center_moment.z > 0);
        center_moment /= center_moment.z;

        float variance = 0.0f;
        auto color = center_color;

        if (frame < 4) {
            // 積算フレーム数が４未満 or Disoccludedされている.
            // 7x7 birateral filterで輝度を計算.
            // If the accumulated frame count is less than 4 or the pixel is disoccluded,
            // compute the luminance by 7x7 birateral filter.
            // If the pixel is disoccluded, the frame count is reset as 1.
            // So, just checking whether the frame count is less than 4 satisfies the both condition.

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

                        const auto uv_length = aten::sqrt(static_cast<float>(u * u + v * v));

                        const float Wz = aten::abs(sample_depth - center_depth) / (pixel_distance_ratio * uv_length + 1e-2f);
                        const float Wn = aten::pow(aten::max(0.0f, dot(sample_nml, center_normal)), 128.0f);

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

            variance = aten::max(0.0f, moment_sum.x - moment_sum.y * moment_sum.y);
        }
        else {
            variance = aten::max(0.0f, center_moment.x - center_moment.y * center_moment.y);
        }

        color.w = variance;
        aov_color_variance[idx] = color;

        return AT_NAME::_detail::MakeVec4(variance, variance, variance, 1);
    }

    /**
     * @brief Execute 3x3 gaussian filter for one element of vecto4 value.
     *
     * @tparam[in] MemberVar Pointer to member variable in vector4 type.
     * @param[in] ix X position to the pixel in the screen coordinate.
     * @param[in] iy Y position to the pixel in the screen coordinate.
     * @param[in] width Screen width.
     * @param[in] height Screen height.
     * @param[in] buffer Buffer to store vector 4 value.
     * @return 3x3 gaussian filtered value.
     */
    template <class MemberVar>
    inline AT_DEVICE_API float Exec3x3GaussFilter(
        int32_t ix, int32_t iy,
        int32_t width, int32_t height,
        const aten::const_span<AT_NAME::_detail::v4>& buffer,
        MemberVar member_var)
    {
        static constexpr float kernel_array[] = {
            1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
            1.0 / 8.0,  1.0 / 4.0, 1.0 / 8.0,
            1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0,
        };
        static constexpr aten::const_span kernel(kernel_array);

        static constexpr int32_t offsetx_array[] = {
            -1, 0, 1,
            -1, 0, 1,
            -1, 0, 1,
        };
        static constexpr aten::const_span offsetx(offsetx_array);

        static constexpr int32_t offsety_array[] = {
            -1, -1, -1,
            0, 0, 0,
            1, 1, 1,
        };
        static constexpr aten::const_span offsety(offsety_array);

        float sum = 0;

        int32_t pos = 0;

#pragma unroll
        for (int32_t i = 0; i < 9; i++) {
            int32_t xx = clamp(ix + offsetx[i], 0, width - 1);
            int32_t yy = clamp(iy + offsety[i], 0, height - 1);

            int32_t idx = _detail::GetIdx(xx, yy, width);

            // NOTE:
            // https://stackoverflow.com/questions/58111915/access-member-variables-using-templates
            float tmp = buffer[idx].*member_var;

            sum += kernel[pos] * tmp;

            pos++;
        }

        return sum;
    }

    /**
     * @brief Check the pixel is background for A-trous filter.
     *
     * @param[in] is_final_iter Whether this API is called as the final iteration of A-trous filter.
     * @param[in] idx Index to the pixel.
     * @param[in] center_color Color of the filtering center pixel.
     * @param[in] aov_texclr_meshid AOV buffer to store albedo color and mesh id.
     * @param[out] color_variance_buffer Buffer to store contribute and variance.
     * @param If the pixel is background and this API is called as the final iteration, returns the computed pixel color.
     *        If the pixel is background but this API is not calles as the final iteration, returns the valid optiaonal but no value.
     *        If the pixel is not background, returns nullopt.
     */
    inline AT_DEVICE_API std::optional<std::optional<AT_NAME::_detail::v4>> CheckIfBackgroundPixelForAtrous(
        const bool is_final_iter,
        const int32_t idx,
        const AT_NAME::_detail::v4& center_color,
        const aten::const_span<AT_NAME::_detail::v4>& aov_texclr_meshid,
        aten::span< AT_NAME::_detail::v4>& color_variance_buffer)
    {
        std::optional<AT_NAME::_detail::v4> result;

        const auto center_meshid = aov_texclr_meshid[idx].w;

        if (center_meshid < 0) {
            // If mesh id is negative, the pixel is the background.
            // So, just output the background pixel.
            color_variance_buffer[idx] = AT_NAME::_detail::MakeVec4(center_color.x, center_color.y, center_color.z, 0.0f);

            // TODO:
            // Is the following necessary?
            if (is_final_iter) {
                // In the final iteration, apply albedo.
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

    /**
     * @brief Execute A-trous wavelet filter.
     *
     * @param[in] is_first_iter Whether this API is called as the 1st A-trous wavelt filter iteration.
     * @param[in] ix X position to the pixel in the screen coordinate.
     * @param[in] iy Y position to the pixel in the screen coordinate.
     * @param[in] width Screen width.
     * @param[in] height Screen height.
     * @param[in] gauss_filtered_variance Gaussian filtered variance.
     * @param[in] center_normal Normal of the filtering center pixel.
     * @param[in] center_depth Depth of the filtering center pixel.
     * @param[in] center_meshid Mesh id of the filtering center pixel.
     * @param[in] center_color Color of the filtering center pixel.
     * @param[in] aov_normal_depth AOV buffer to store normal and depth.
     * @param[in] aov_texclr_meshid AOV buffer to store albedo color and mesh id.
     * @param[in] aov_color_variance AOV buffer to store contribution and variance.
     * @param[in] aov_moment_temporalweight AOV buffer to store moments and temporal weight.
     * @param[in] filter_iter_count Current A-trous wavelet filter iterantion count.
     * @param[in] camera_distance Distance from camera origin to the screen.
     * @return A-trous wavelet filtered color.
     */
    inline AT_DEVICE_API AT_NAME::_detail::v4 ExecAtrousWaveletFilter(
        bool is_first_iter,
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
        const int32_t filter_iter_count,
        const float camera_distance)
    {
        static constexpr float sigmaZ = 1.0f;
        static constexpr float sigmaN = 128.0f;
        static constexpr float sigmaL = 4.0f;

        static constexpr float h_array[] = {
            2.0f / 3.0f,  2.0f / 3.0f,  2.0f / 3.0f,  2.0f / 3.0f,
            1.0f / 6.0f,  1.0f / 6.0f,  1.0f / 6.0f,  1.0f / 6.0f,
            4.0f / 9.0f,  4.0f / 9.0f,  4.0f / 9.0f,  4.0f / 9.0f,
            1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,
            1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,
            1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
        };
        // NOTE:
        // std::declval<decltype(h_array)>() -> Make value of const float [24]
        // std::declval<decltype(h_array)>()[0] -> One value in const float [24]
        // decltype(std::declval<decltype(h_array)>()[0]) -> Type of the value in const float [24] -> const float&
        using element_of_h_array = std::remove_const_t<std::remove_reference_t<decltype(std::declval<decltype(h_array)>()[0])>>;
        static constexpr aten::const_span h(h_array);

        static constexpr int32_t offsetx_array[] = {
            1,  0, -1, 0,
            2,  0, -2, 0,
            1, -1, -1, 1,
            1, -1, -1, 1,
            2, -2, -2, 2,
            2, -2, -2, 2,
        };
        using element_of_offsetx_array = std::remove_const_t<std::remove_reference_t<decltype(std::declval<decltype(offsetx_array)>()[0])>>;
        static constexpr aten::const_span offsetx(offsetx_array);

        static constexpr int32_t offsety_array[] = {
            0, 1,  0, -1,
            0, 2,  0, -2,
            1, 1, -1, -1,
            2, 2, -2, -2,
            1, 1, -1, -1,
            2, 2, -2, -2,
        };
        using element_of_offsety_array = std::remove_const_t<std::remove_reference_t<decltype(std::declval<decltype(offsety_array)>()[0])>>;
        static constexpr aten::const_span offsety(offsety_array);

        const int32_t step_scale = 1 << filter_iter_count;

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

            const auto u_length = aten::sqrt(static_cast<float>(scaled_offset_x * scaled_offset_x + scaled_offset_y * scaled_offset_y));

            const int32_t qidx = _detail::GetIdx(xx, yy, width);

            const auto& normal_depth = aov_normal_depth[qidx];
            const auto& texclr_meshid = aov_texclr_meshid[qidx];

            const auto normal{ AT_NAME::_detail::MakeVec3(normal_depth.x, normal_depth.y, normal_depth.z) };

            const auto depth = normal_depth.w;
            const int32_t meshid = static_cast<int32_t>(texclr_meshid.w);

            const auto& color = is_first_iter ? aov_color_variance[qidx] : color_variance_buffer[qidx];
            const auto variance = color.w;

            float lum = AT_NAME::color::luminance(color.x, color.y, color.z);

            float Wz = 3.0f * fabs(center_depth - depth) / (sigmaZ * (pixel_distance_ratio * u_length) + 0.000001f);

            float Wn = powf(aten::max(0.0f, dot(center_normal, normal)), sigmaN);

            float Wl = aten::min(expf(-fabs(center_luminance - lum) / (sigmaL * sqrt_gauss_filtered_variance + 0.000001f)), 1.0f);

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

    /**
     * @brief Execute post process per A-trous wavelet filter iteration.
     *
     * @param[in] is_first_iter Whether this API is called as the 1st A-trous wavelt filter iteration.
     * @param[in] is_final_iter Whether this API is called as the final A-trous wavelt filter iteration.
     * @param[in] idx Index to the pixel.
     * @param[in] filtered_color_variance A-trous wavlet filtered color and variance.
     * @param[in] aov_texclr_meshid AOV buffer to store albedo color and mesh id. This is used only for the final iteration.
     * @param[out] temporary_color_buffer Buffer to store the result of the 1st A-trous wavelt filter iteration.
     * @param[out] temporary_color_buffer Buffer to store the filtered_color_variance as the 1st A-trous wavelt filter iteration.
     * @param[out] next_color_variance_buffer Buffer to store the filtered_color_variance for the next A-trous wavelt filter iteration.
     * @return If is_final_iter is true, returns the albedo multiplied color as the SVGF result for displaying. Otherwise, returns nullopt.
     */
    inline AT_DEVICE_API std::optional<AT_NAME::_detail::v4> PostProcessForAtrousFilter(
        bool is_first_iter, bool is_final_iter,
        const int32_t idx,
        const AT_NAME::_detail::v4& filtered_color_variance,
        const aten::const_span<AT_NAME::_detail::v4>& aov_texclr_meshid,
        aten::span< AT_NAME::_detail::v4>& temporary_color_buffer,
        aten::span< AT_NAME::_detail::v4>& next_color_variance_buffer)
    {
        next_color_variance_buffer[idx] = filtered_color_variance;

        if (is_first_iter) {
            // NOTE:
            // Store the 1st filter iteration color.
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

    /**
     * @brief Copy vector type value with specified number of its elements.
     *
     * @tparam CopyElementNumPerItem Number of elements to be copied.
     * @param idx Index to be copied in the buffer.
     * @param src Source buffer.
     * @param dst Destination buffer to store the copied value from the source buffer.
     */
    template <int32_t CopyElementNumPerItem>
    inline AT_DEVICE_API constexpr void CopyVectorBuffer(
        const int32_t idx,
        const aten::const_span<AT_NAME::_detail::v4>& src,
        aten::span<AT_NAME::_detail::v4>& dst)
    {
        const auto& src_v = src[idx];
        auto& dst_v = dst[idx];

        constexpr auto num = CopyElementNumPerItem > 4 ? 4 : CopyElementNumPerItem;

        if constexpr (num <= 4) {
            if constexpr (num >= 4) {
                dst_v.w = src_v.w;
            }
            if constexpr (num >= 3) {
                dst_v.z = src_v.z;
            }
            if constexpr (num >= 2) {
                dst_v.y = src_v.y;
            }
            if constexpr (num >= 1) {
                dst_v.x = src_v.x;
            }
        }
    }
}   // namespace svgf
}   // namespace AT_NAME
