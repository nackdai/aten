#include <map>

#include "material/toon.h"

#include "light/light.h"
#include "light/light_impl.h"
#include "material/diffuse.h"
#include "material/ggx.h"
#include "material/material.h"
#include "material/material_impl.h"
#include "material/sample_texture.h"
#include "material/toon_specular.h"
#include "misc/color.h"
#include "renderer/pathtracing/pathtracing_nee_impl.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif

namespace AT_NAME
{
    bool Toon::edit(aten::IMaterialParamEditor* editor)
    {
        auto is_updated = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, roughness, 0.01F, 1.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, 0.01F, 10.0F);

        // Toon type.
        constexpr std::array toon_type_str = {
            "Diffuse",
            "Specular",
        };
        constexpr std::array toon_types = {
            aten::MaterialType::Diffuse,
            aten::MaterialType::Specular,
        };

        int32_t toon_type = 0;

        const auto it = std::find(toon_types.begin(), toon_types.end(), m_param.toon.toon_type);
        AT_ASSERT(it != toon_types.end());
        if (it != toon_types.end()) {
            toon_type = static_cast<int32_t>(std::distance(toon_types.begin(), it));
        }

        is_updated |= editor->edit("toon_type", toon_type_str.data(), toon_type_str.size(), toon_type);
        m_param.toon.toon_type = toon_types[toon_type];

        // Stylized highlight.
        if (editor->CollapsingHeader("Stylized Highlight")) {
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highligt_translation_dt, -1.0F, 1.0F);
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highligt_translation_db, -1.0F, 1.0F);
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highligt_scale_t, 0.0F, 1.0F);
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highligt_scale_b, 0.0F, 1.0F);
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highlight_split_t, 0.0F, 1.0F);
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highlight_split_b, 0.0F, 1.0F);
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highlight_square_sharp, 0.0F, 1.0F);
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highlight_square_magnitude, 0.0F, 1.0F);
        }

        // Rim light.
        if (editor->CollapsingHeader("Rim Light")) {
            bool enable_rim_light = m_param.toon.enable_rim_light;
            is_updated |= editor->edit("enable_rim_light", enable_rim_light);
            m_param.toon.enable_rim_light = enable_rim_light;

            is_updated |= AT_EDIT_MATERIAL_PARAM(editor, m_param.toon, rim_light_color);
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, rim_light_width, 0.0F, 1.0F);
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, rim_light_softness, 0.0F, 1.0F);
            is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, rim_light_spread, 0.0F, 1.0F);
        }

        return is_updated;
    }

    AT_DEVICE_API aten::vec3 Toon::ComputeBRDF(
        const AT_NAME::context& ctxt,
        const aten::MaterialParameter& param,
        const aten::LightSampleResult* sampled_light,
        aten::sampler& sampler,
        const aten::vec3& hit_pos,
        const aten::vec3& normal,
        const aten::vec3& wi,
        float u, float v)
    {
        // The target light is sepcified beforehand and it is only one.
        // So, light selected pdf is always 1.
        constexpr float light_selected_pdf = 1.0F;

        aten::vec3 toon_term{ 0.0F };

        if (sampled_light) {
            // Diffuse.
            aten::MaterialParameter base_mtrl = param;
            if (param.toon.toon_type == aten::MaterialType::Diffuse) {
                base_mtrl.type = aten::MaterialType::Diffuse;
            }
            else {
                base_mtrl.type = aten::MaterialType::ToonSpecular;
            }

            // Compute radiance.
            // The target light is singular, and then MIS in ComputeRadianceNEE is always 1.0.
            auto res = ComputeRadianceNEE(
                aten::vec3(1.0F),
                wi, normal,
                base_mtrl, 0.0F, u, v,
                light_selected_pdf,
                *sampled_light);

            if (res) {
                auto radiance = res.value();

                float lum_y = aten::clamp(color::luminance(radiance), 0.0F, 1.0F);

                if (!aten::isClose(lum_y, 0.0F, 1e-9F, 1e-3F)) {
                    const auto hsv = color::RGBtoHSV(radiance);
                    const auto color = color::HSVtoRGB(aten::vec3(hsv.r, hsv.g, 1));

                    lum_y = pow(lum_y, 1.0F / 2.2F);

                    const auto remap = AT_NAME::sampleTexture(param.toon.remap_texture, 0.5F, lum_y, aten::vec4(1.0F));

                    toon_term = remap * color;
                }
            }
        }

        const auto rim_light_term = ComputeRimLight(ctxt, param, hit_pos, normal, wi);

        return toon_term + rim_light_term;
    }

    namespace _detail {
        AT_DEVICE_API float bezier_smoothstep(float edge0, float edge1, float mid, float t, float s)
        {
            if (t <= edge0) {
                return 0;
            }
            else if (t >= edge1) {
                return 1;
            }

            t = (t - edge0) / (edge1 - edge0);
            t *= s;

            float B0 = 0.0F;
            float B1 = mid;
            float B2 = 1.0F;
            float P = (B0 - 2 * B1 + B2) * t * t + (-2 * B0 + 2 * B1) * t + B0;
            return P;
        }

    }

    AT_DEVICE_API aten::vec3 Toon::ComputeRimLight(
        const AT_NAME::context& ctxt,
        const aten::MaterialParameter& param,
        const aten::vec3& hit_pos,
        const aten::vec3& normal,
        const aten::vec3& wi)
    {
        aten::vec3 post_processed_additional_color(0.0F);

        const auto V = -wi;
        const auto N = normal;

        // Rim light.
        if (param.toon.enable_rim_light) {
            const auto NdotV = dot(V, N);

            float rim = 0;

            // NOTE:
            // width is larger, rim light is thicker. If width is smaller, rim light is thinner.
            // As smoothstep, less than edge0 is zero.
            // In that case, if width is large, the result of smoothstep might be more zero.
            // It means, if width is larger, rim light is thinner. It's fully opposite what widthe means.
            // Therefore, width need to be invert as smoothstep edge0 argument.
            if (NdotV > 0) {
                rim = _detail::bezier_smoothstep(
                    1.0F - param.toon.rim_light_width,
                    1.0F,
                    (1 - param.toon.rim_light_softness) * 0.5F,
                    1 - NdotV,
                    param.toon.rim_light_spread);

                post_processed_additional_color += rim * param.toon.rim_light_color;
            }
        }

        return post_processed_additional_color;
    }

    AT_DEVICE_API float ToonSpecular::ComputePDF(
        const aten::MaterialParameter& param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        const auto V = -wi;
        const auto L = wo;
        const auto N = normal;

        const auto H = ComputeHalfVector(param, N, V, L);
        const auto pdf = MicrofacetGGX::ComputePDFWithHalfVector(param.standard.roughness, N, H, L);
        return pdf;
    }

    AT_DEVICE_API aten::vec3 ToonSpecular::ComputeBRDF(
        const aten::MaterialParameter& param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        const auto V = -wi;
        const auto L = wo;
        const auto N = normal;

        const auto H = ComputeHalfVector(param, N, V, L);

        const auto brdf = MicrofacetGGX::ComputeBRDFWithHalfVector(
            param.standard.roughness,
            param.standard.ior,
            N, V, L, H);

        return brdf;
    }

    AT_DEVICE_API aten::vec3 ToonSpecular::ComputeHalfVector(
        const aten::MaterialParameter& param,
        const aten::vec3& N,
        const aten::vec3& V,
        const aten::vec3& L)
    {
        // NOTE:
        // Stylized hightlight for Cartoon Rendering and Animation.
        // http://ivizlab.sfu.ca/arya/Papers/IEEE/C%20G%20&%20A/2003/July/Stylized%20Highlights%20for%20Cartoon%20Rendering.pdf

        auto H = normalize(L + V);

        // Stylized hightlight.
        aten::vec3 t, b;
        aten::tie(t, b) = aten::GetTangentCoordinate(N);

        // Translation.
        H = H + param.toon.highligt_translation_dt * t + param.toon.highligt_translation_db * b;
        H = normalize(H);

        // Direction scale.
        H = H - param.toon.highligt_scale_t * dot(H, t) * t;
        H = normalize(H);
        H = H - param.toon.highligt_scale_b * dot(H, b) * b;
        H = normalize(H);

        // Split.
        H = H - param.toon.highlight_split_t * aten::sign(dot(H, t)) * t - param.toon.highlight_split_b * aten::sign(dot(H, b)) * b;
        H = normalize(H);

        // Square.
        const auto sqrnorm_t = sin(aten::pow(aten::acos(dot(H, t)), param.toon.highlight_square_sharp));
        const auto sqrnorm_b = sin(aten::pow(aten::acos(dot(H, b)), param.toon.highlight_square_sharp));
        H = H - param.toon.highlight_square_magnitude * (sqrnorm_t * dot(H, t) * t + sqrnorm_b * dot(H, b) * b);
        H = normalize(H);

        return H;
    }
}

//////////////////////////////////////////

namespace AT_NAME
{
    AT_DEVICE_API aten::vec3 StylizedBrdf::ComputeBRDF(
        const AT_NAME::context& ctxt,
        const aten::MaterialParameter& param,
        const aten::LightSampleResult* sampled_light,
        aten::sampler& sampler,
        const aten::vec3& hit_pos,
        const aten::vec3& normal,
        const aten::vec3& wi,
        float u, float v)
    {
        // The target light is sepcified beforehand and it is only one.
        // So, light selected pdf is always 1.
        constexpr float light_selected_pdf = 1.0F;

        aten::vec3 toon_term{ 0.0F };

        if (sampled_light) {
            // Diffuse.
            aten::MaterialParameter base_mtrl = param;
            if (param.toon.toon_type == aten::MaterialType::Diffuse) {
                base_mtrl.type = aten::MaterialType::Diffuse;
            }
            else {
                base_mtrl.type = aten::MaterialType::ToonSpecular;
            }

            const auto pdf = material::samplePDF(&base_mtrl, normal, wi, sampled_light->dir, u, v);

            // Compute radiance.
            // The target light is singular, and then MIS in ComputeRadianceNEE is always 1.0.
            auto res = ComputeRadianceNEE(
                aten::vec3(1.0F),
                wi, normal,
                base_mtrl, 0.0F, u, v,
                light_selected_pdf,
                *sampled_light);

            if (res) {
                // NOTE:
                // Global Illumination-Aware Stylised Shading
                // https://diglib.eg.org/server/api/core/bitstreams/d84134e0-af8c-4db6-a13a-dc854294f6aa/content

                // TODO
                constexpr float W_MIN = 0.01F;

                const auto radiance = res.value();

                // Convert RGB to XYZ.
                const auto xyz = color::sRGBtoXYZ(radiance);
                const auto y = xyz.y;

                // To avoid too dark, compare with the minimum weight.
                const auto weight = aten::max(y, W_MIN);

                const auto y_min = aten::max(0.0F, aten::min(param.toon.stylized_y_min, param.toon.stylized_y_max));
                const auto y_max = aten::max(param.toon.stylized_y_min, param.toon.stylized_y_max);

                // Compute texture coord (1D, vertical) for remap texture.
                auto remap_v = 0.0F;
                if (y_max <= y) {
                    remap_v = 1.0F;
                }
                else if (y <= y_min) {
                    remap_v = 0.0F;
                }
                else {
                    remap_v = (y - y_min) / (y_max - y_min);
                }

                const auto remap = AT_NAME::sampleTexture(param.toon.remap_texture, 0.5F, remap_v, aten::vec4(radiance));

                toon_term = weight * remap * pdf;
            }
        }

        const auto rim_light_term = ComputeRimLight(ctxt, param, hit_pos, normal, wi);

        return toon_term + rim_light_term;
    }

    bool StylizedBrdf::edit(aten::IMaterialParamEditor* editor)
    {
        bool is_updated = Toon::edit(editor);

        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, stylized_y_min, 0.0F, 10.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, stylized_y_max, 0.0F, 10.0F);

        const auto tmp_min = aten::min(m_param.toon.stylized_y_min, m_param.toon.stylized_y_max);
        const auto tmp_max = aten::max(m_param.toon.stylized_y_min, m_param.toon.stylized_y_max);

        m_param.toon.stylized_y_min = tmp_min;
        m_param.toon.stylized_y_max = tmp_max;

        return is_updated;
    }
}
