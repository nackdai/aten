#include "material/toon.h"

#include "light/light.h"
#include "light/light_impl.h"
#include "material/diffuse.h"
#include "material/ggx.h"
#include "material/material.h"
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

        // Stylized highlight.
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highligt_translation_dt, -1.0F, 1.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highligt_translation_db, -1.0F, 1.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highligt_scale_t, 0.0F, 1.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highligt_scale_b, 0.0F, 1.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highlight_split_t, 0.0F, 1.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highlight_split_b, 0.0F, 1.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highlight_square_sharp, 0.0F, 1.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, highlight_square_magnitude, 0.0F, 1.0F);

        // Rim light.
        {
            bool enable_rim_light = m_param.toon.enable_rim_light;
            is_updated |= editor->edit("enable_rim_light", enable_rim_light);
            m_param.toon.enable_rim_light = enable_rim_light;
        }
        is_updated |= AT_EDIT_MATERIAL_PARAM(editor, m_param.toon, rim_light_color);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, rim_light_width, 0.0F, 1.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, rim_light_softness, 0.0F, 1.0F);
        is_updated |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.toon, rim_light_spread, 0.0F, 1.0F);

        return is_updated;
    }

    AT_DEVICE_API aten::vec3 Toon::ComputeBRDF(
        const AT_NAME::context& ctxt,
        const aten::MaterialParameter& param,
        const aten::LightSampleResult* light_sample,
        aten::sampler& sampler,
        const aten::vec3& hit_pos,
        const aten::vec3& normal,
        const aten::vec3& wi,
        float u, float v,
        float& pdf)
    {
        // TODO
        constexpr float w_min = 0.01F;
        constexpr float y_min = 0.0F;
        constexpr float y_max = 1.0F;

        aten::vec3 radiance(0.0F);
        pdf = 1.0F;

        if (light_sample) {
            // TODO
            // How can we configure base material type.
            aten::MaterialParameter base_mtrl = param;
#if 0
            base_mtrl.type = aten::MaterialType::Diffuse;
#else
            base_mtrl.type = aten::MaterialType::ToonSpecular;
#endif

            // The target light is sepcified beforehand and it is only one.
            // So, light selected pdf is always 1.
            constexpr float light_selected_pdf = 1.0F;

            // Compute radiance.
            // The target light is singular, and then MIS in ComputeRadianceNEE is always 1.0.
            auto res = ComputeRadianceNEE(
                wi, normal,
                base_mtrl, 0.0F, u, v,
                light_selected_pdf, *light_sample);
            if (res) {
                radiance = res.value();
#if 0
                pdf = Diffuse::ComputePDF(normal, light_sample->dir);
#else
                pdf = ToonSpecular::ComputePDF(param, normal, wi, light_sample->dir, u, v);
#endif
            }
        }

        // NOTE:
        // ACP
        // https://diglib.eg.org/server/api/core/bitstreams/d84134e0-af8c-4db6-a13a-dc854294f6aa/content

        // Convert RGB to XYZ.
        const auto xyz = color::RGBtoXYZ(radiance);
        const auto y = xyz.y;

        // To avoid too dark, compare with the minimum weight.
        const auto weight = aten::saturate(aten::cmpMax(y, w_min));

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

        // TODO
        // According to the paper, weight is necessary.
        // But, it causes the gradation, color change etc from ramp color...
        aten::vec3 bsdf = weight * remap * pdf;

        return bsdf;
    }

    namespace _detail {
        inline AT_DEVICE_API float bezier(float B0, float B1, float B2, float t)
        {
            float P = (B0 - 2 * B1 + B2) * t * t + (-2 * B0 + 2 * B1) * t + B0;
            return P;
        }

        inline AT_DEVICE_API float bezier_smoothstep(float edge0, float edge1, float mid, float t, float s)
        {
            if (t <= edge0) {
                return 0;
            }
            else if (t >= edge1) {
                return 1;
            }

            t = (t - edge0) / (edge1 - edge0);
            t *= s;

            float P = bezier(0, mid, 1, t);
            return P;
        }

    }

    AT_DEVICE_API aten::vec3 Toon::PostProcess(
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
                    1.0 - param.toon.rim_light_width,
                    1.0,
                    (1 - param.toon.rim_light_softness) * 0.5,
                    1 - NdotV,
                    param.toon.rim_light_spread);

                post_processed_additional_color += rim * param.toon.rim_light_color;
            }
        }

        return post_processed_additional_color;
    }

    namespace blinn {
        inline  AT_DEVICE_API float ComputeDistribution(
            const float shininess,
            const aten::vec3& N,
            const aten::vec3& H)
        {
            const auto a = shininess;
            const auto NdotH = dot(N, H);

            // NOTE
            // http://simonstechblog.blogspot.jp/2011/12/microfacet-brdf.html
            // https://agraphicsguynotes.com/posts/sample_microfacet_brdf/
            auto D = (a + 2) / (2 * AT_MATH_PI);
            D *= aten::pow(NdotH, a);
            D *= (NdotH > 0 ? 1 : 0);

            return D;
        }

        inline  AT_DEVICE_API float ComputePDF(
            const aten::MaterialParameter& param,
            const aten::vec3& N,
            const aten::vec3& L,
            const aten::vec3& H)
        {
            const auto costheta = dot(N, H);

            const auto D = ComputeDistribution(param.standard.shininess, N, H);

            // For Jacobian |dwh/dwo|
            const auto denom = 4 * aten::abs(dot(L, H));

            const auto pdf = denom > 0 ? (D * costheta) / denom : 0;

            return pdf;
        }

        inline AT_DEVICE_API aten::vec3 ComputeBRDF(
            const aten::MaterialParameter& param,
            const aten::vec3& N,
            const aten::vec3& V,
            const aten::vec3& L,
            const aten::vec3& H)
        {
            // Assume index of refraction of the medie on the incident side is vacuum.
            const auto ni = 1.0F;
            const auto nt = param.standard.ior;

            auto NdotH = aten::abs(dot(N, H));
            auto VdotH = aten::abs(dot(V, H));
            auto NdotL = aten::abs(dot(N, L));
            auto NdotV = aten::abs(dot(N, V));

            const auto F = material::ComputeSchlickFresnel(ni, nt, L, H);

            auto denom = 4 * NdotL * NdotV;

            const auto D = ComputeDistribution(param.standard.shininess, N, H);

            // Compute G.
            auto G{ 1.0F };
            {
                // Cook-Torrance geometry function.
                // http://simonstechblog.blogspot.jp/2011/12/microfacet-brdf.html

                auto G1 = 2 * NdotH * NdotL / VdotH;
                auto G2 = 2 * NdotH * NdotV / VdotH;
                G = aten::cmpMin(1.0F, aten::cmpMin(G1, G2));
            }

            const auto brdf = denom > AT_MATH_EPSILON ? F * G * D / denom : 0.0F;

            return aten::vec3(brdf);
        }
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

#if 0
        const auto pdf = blinn::ComputePDF(param, N, L, H);
#else
        const auto pdf = MicrofacetGGX::ComputePDFWithHalfVector(param.standard.roughness, N, H, L);
#endif
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

#if 0
        const auto brdf = blinn::ComputeBRDF(
            param,
            N, V, L, H);
#else
        const auto brdf = MicrofacetGGX::ComputeBRDFWithHalfVector(
            param.standard.roughness,
            param.standard.ior,
            N, V, L, H);
#endif


        return brdf;
    }

    AT_DEVICE_API aten::vec3 ToonSpecular::ComputeHalfVector(
        const aten::MaterialParameter& param,
        const aten::vec3& N,
        const aten::vec3& V,
        const aten::vec3& L)
    {
        auto H = normalize(L + V);

        // Stylized hightlight.
        aten::vec3 t, b;
        aten::tie(t, b) = aten::GetTangentCoordinate(N);

#if 0
        // NOTE:
        // The result from GetTangentCoordinate is:
        //    +------>b
        //   /|
        //  / |
        // n  t
        // It's fully right hand coordiante as expected.
        // But, this is not intuitive to translate the half vector.
        // To translate the half vector intuitively like the following:
        //    b
        //    |
        //    |
        //    +------>t
        //   /
        //  /
        // n
        // We rotate t and b around n.
        aten::mat4 rot;
        rot.asRotateByZ(AT_MATH_PI * 0.5F);
        t = rot.apply(t);
        b = rot.apply(b);
#endif

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
