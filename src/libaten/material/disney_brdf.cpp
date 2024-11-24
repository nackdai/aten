#include "math/math.h"
#include "material/disney_brdf.h"
#include "material/diffuse.h"
#include "material/ggx.h"
#include "misc/color.h"

namespace AT_NAME
{
    // NOTE:
    // http://project-asura.com/blog/archives/1972
    // https://rayspace.xyz/CG/contents/Disney_principled_BRDF/
    // https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
    // https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/renderer/modeling/bsdf/disneybrdf.cpp
    // https://schuttejoe.github.io/post/disneybsdf/
    // Physically-Based Shading at Disney
    // https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf

    bool DisneyBRDF::edit(aten::IMaterialParamEditor* editor)
    {
        bool is_update = false;
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, subsurface, 0, 1);
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, metallic, 0, 1);
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, specular, 0, 1);
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, specularTint, 0, 1);
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, roughness, 0, 1);
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, anisotropic, 0, 1);
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, sheen, 0, 1);
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, sheenTint, 0, 1);
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, clearcoat, 0, 1);
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, clearcoatGloss, 0, 1);
        is_update |= AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, float(0.01), float(10));
        is_update |= AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

        return is_update;
    }

    namespace DisneyBrdfUtil {
        static inline AT_DEVICE_API aten::vec3 ComputeCtint(const aten::vec3& base_color)
        {
            const auto Y = dot(base_color, aten::vec3(0.3F, 0.6F, 0.1F));
            const auto Ctint = Y > 0 ? base_color / Y : aten::vec3(1.0F);
            return Ctint;
        }

        static inline AT_DEVICE_API float SchlickFresnel(const float u)
        {
            const float m = aten::clamp(1.0F - u, 0.0F, 1.0F);
            const float m2 = m * m;
            return m2 * m2 * m;
        }
    };

    class DisneyBrdfDiffuse {
    public:
        static inline AT_DEVICE_API aten::vec3 EvalBRDF(
            const aten::vec3& base_color,
            const float roughness,
            const float subsurface,
            const aten::vec3& V,
            const aten::vec3& L,
            const aten::vec3& N)
        {
            const auto H = normalize(V + L);

            const auto LdotH = dot(L, H);
            const auto NdotV = dot(V, N);
            const auto NdotL = dot(L, N);

            const auto FV = DisneyBrdfUtil::SchlickFresnel(dot(V, N));
            const auto FL = DisneyBrdfUtil::SchlickFresnel(dot(L, N));

            // Diffuse factor is available.
            const auto Fd90 = 0.5F + 2 * LdotH * LdotH * roughness;
            auto fd = aten::mix(1.0F, Fd90, FL) * aten::mix(1.0F, Fd90, FV);

            // Subsurface factor is available.
            const auto Fss90 = LdotH * LdotH * roughness;
            const auto Fss = aten::mix(1.0F, Fss90, FL) * aten::mix(1.0F, Fss90, FV);
            const auto ss = 1.25F * (Fss * (1.0F / (NdotL + NdotV) - 0.5F) + 0.5F);

            fd = aten::mix(fd, ss, subsurface);

            return base_color / AT_MATH_PI * fd;
        }

        static inline AT_DEVICE_API float EvalPDF(
            const aten::vec3& L,
            const aten::vec3& N)
        {
            return Diffuse::ComputePDF(N, L);
        }

        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2,
            const aten::vec3& N)
        {
            const auto L = Diffuse::SampleDirection(N, r1, r2);
            return L;
        }
    };

    class DisneyBrdfSheen {
    public:
        static inline AT_DEVICE_API aten::vec3 EvalBRDF(
            const aten::vec3& base_color,
            const float sheen,
            const float sheen_tint,
            const aten::vec3& V,
            const aten::vec3& L,
            const aten::vec3& N)
        {
            const auto H = normalize(V + L);

            // Luminance approximmation.
            const auto Ctint = DisneyBrdfUtil::ComputeCtint(base_color);
            const auto Csheen = aten::mix(aten::vec3(1.0F), Ctint, sheen_tint);

            const float FH = DisneyBrdfUtil::SchlickFresnel(dot(L, H));

            const auto Fsheen = sheen * Csheen * FH;
            return Fsheen;
        }

        static inline AT_DEVICE_API float EvalPDF(
            const aten::vec3& L,
            const aten::vec3& N)
        {
            // Uniform hemisphere sampling.
            return 1 / (AT_MATH_PI);

            //return Diffuse::ComputePDF(N, L);
        }

        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2,
            const aten::vec3& N)
        {
            const auto L = Diffuse::SampleDirection(N, r1, r2);
            return L;
        }
    };

    class DisneyBrdfClearcoat {
    public:
        static inline AT_DEVICE_API float D_GTR1(
            const float roughness,
            const float NdotH)
        {
            const auto a = roughness;

            if (a >= 1) {
                return 1 / AT_MATH_PI;
            }

            const auto a2 = a * a;
            const auto t = 1.0F + (a2 - 1.0F) * NdotH * NdotH;
            return (a2 - 1) / (AT_MATH_PI * aten::log(a2) * t);
        }

        static inline AT_DEVICE_API aten::vec3 EvalBRDF(
            const aten::vec3& base_color,
            const float clearcoat,
            const float clearcoat_gloss,
            const aten::vec3& V,
            const aten::vec3& L,
            const aten::vec3& N)
        {
#if 0
            const auto H = normalize(V + L);

            const auto NdotH = dot(N, H);
            const auto LdotH = dot(L, H);

            const auto a_clearcoat = aten::mix(0.1F, 0.001F, clearcoat_gloss);

            const auto D = D_GTR1(a_clearcoat, NdotH);

            // clearcoat (ior = 1.5 -> F0 = 0.04).
            const auto FH = DisneyBrdfUtil::SchlickFresnel(aten::abs(LdotH));
            const auto F = aten::mix(0.04F, 1.0F, FH);

            const auto G = MicrofacetGGX::ComputeG2Smith(0.25F, V, L, N);

            const auto denom = 4 * aten::abs(dot(L, N)) * aten::abs(dot(V, N));

            const auto f_clearcoat = denom > 0.0F ? 0.25F * clearcoat * F * D * G : 0.0F;
            return aten::vec3(f_clearcoat);
#else
            const auto H = normalize(V + L);

            const auto LdotH = dot(L, H);

            const auto FH = DisneyBrdfUtil::SchlickFresnel(aten::abs(LdotH));
            const auto F = aten::mix(0.04F, 1.0F, FH);

            const auto f_clearcoat = 0.25F * clearcoat * F;
            return aten::vec3(f_clearcoat);
#endif
        }

        static inline AT_DEVICE_API float EvalPDF(
            const float clearcoat_gloss,
            const aten::vec3& V,
            const aten::vec3& L,
            const aten::vec3& N)
        {
            const auto H = normalize(V + L);

            const auto NdotH = dot(N, H);

            const auto a_clearcoat = aten::mix(0.1F, 0.001F, clearcoat_gloss);

            const auto D = D_GTR1(a_clearcoat, NdotH);

            const auto costheta = aten::abs(dot(H, N));

            // For Jacobian |dwh/dwo|
            const auto denom = 4 * aten::abs(dot(L, H));

            const auto pdf = denom > 0 ? (D * costheta) / denom : 0;

            return pdf;
        }

        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2,
            const float clearcoat_gloss,
            const aten::vec3& V,
            const aten::vec3& N)
        {
            const auto a_clearcoat = aten::mix(0.1F, 0.001F, clearcoat_gloss);

            const auto m = MicrofacetGGX::SampleMicrosurfaceNormal(a_clearcoat, N, r1, r2);

            // We can assume ideal reflection on each micro surface.
            // So, compute ideal reflection vector based on micro surface normal.
            const auto wo = material::ComputeReflectVector(-V, m);

            return wo;
        }
    };

    // TODO:
    // Anistoropic:
    // https://kinakomoti321.hatenablog.com/entry/2022/01/29/011529#%E7%95%B0%E6%96%B9%E6%80%A7%E6%9C%89%E3%82%8A%E6%B3%95%E7%B7%9A%E5%88%86%E5%B8%83%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AA%E3%83%B3%E3%82%B0
    // e.g.
    // const aten::vec3 X = aten::getOrthoVector(N);
    // const aten::vec3 Y = normalize(cross(N, X));
    // const auto NdotH = dot(N, H);
    // const auto HdotX = dot(H, X);
    // const auto HdotY = dot(H, Y);

    class DisneyBrdfSpecular {
    public:
        static inline AT_DEVICE_API aten::vec3 EvalBRDF(
            const aten::vec3& base_color,
            const float roughness,
            const float metallic,
            const float specular,
            const float specular_tint,
            const aten::vec3& V,
            const aten::vec3& L,
            const aten::vec3& N)
        {
            const auto H = normalize(V + L);

            const auto Ctint = DisneyBrdfUtil::ComputeCtint(base_color);
            const auto Cspec = aten::mix(aten::vec3(1), Ctint, specular_tint);

            const auto F_s0 = aten::mix(0.08F * specular * Cspec, base_color, metallic);

            const auto F = aten::mix(F_s0, aten::vec3(1.0F), dot(L, H));

            const auto D = MicrofacetGGX::ComputeDistribution(H, N, roughness);
            const auto G = MicrofacetGGX::ComputeG2Smith(roughness, V, L, N);

            const auto NdotL = aten::abs(dot(N, L));
            const auto NdotV = aten::abs(dot(N, V));
            const auto denom = 4 * NdotV * NdotL;

            const auto f_spec = denom > 0 ? F * D * G / denom : aten::vec3(0.0F);

            return f_spec;
        }

        static inline AT_DEVICE_API float EvalPDF(
            const float roughness,
            const aten::vec3& V,
            const aten::vec3& L,
            const aten::vec3& N)
        {
            const auto H = normalize(V + L);
            const auto pdf = MicrofacetGGX::ComputePDFWithHalfVector(roughness, N, H, L);
            return pdf;
        }

        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2,
            const float roughness,
            const aten::vec3& V,
            const aten::vec3& N)
        {
            const auto wo = MicrofacetGGX::SampleDirection(r1, r2, roughness, -V, N);
            return wo;
        }
    };

    AT_DEVICE_API void DisneyBRDF::ComputeWeights(
        std::array<float, Component::Num>& weights,
        const aten::vec3& base_color,
        const float metalic,
        const float sheen,
        const float specular,
        const float clearcoat)
    {
        const auto base_color_lum = AT_NAME::color::luminance(base_color.x, base_color.y, base_color.z);

        weights[Component::Diffuse] = base_color_lum * (1 - metalic);
        weights[Component::Sheen] = sheen * (1 - metalic);
        weights[Component::Specular] = aten::mix(specular, 1.0F, metalic);
        weights[Component::Clearcoat] = 0.25F * clearcoat;

        float norm = 0.0F;
        for (auto w : weights) {
            norm += w;
        }

        if (norm > 0) {
            for (auto& w : weights) {
                w /= norm;
            }
        }
    }

    AT_DEVICE_API void DisneyBRDF::GetCDF(
        const std::array<float, Component::Num>& weights,
        std::array<float, Component::Num>& cdf)
    {
        cdf[Component::Diffuse] = weights[Component::Diffuse];
        cdf[Component::Sheen] = cdf[Component::Diffuse] + weights[Component::Sheen];
        cdf[Component::Specular] = cdf[Component::Sheen] + weights[Component::Specular];
        cdf[Component::Clearcoat] = cdf[Component::Specular] + weights[Component::Clearcoat];
    }

    AT_DEVICE_API float DisneyBRDF::pdf(
        const aten::MaterialParameter& mtrl,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        const auto roughness = mtrl.standard.roughness;
        const auto metalic = mtrl.standard.metallic;
        const auto sheen = mtrl.standard.sheen;
        const auto specular = mtrl.standard.specular;
        const auto clearcoat = mtrl.standard.clearcoat;
        const auto clearcoat_gloss = mtrl.standard.clearcoatGloss;

        std::array<float, Component::Num> weights;
        ComputeWeights(weights, mtrl.baseColor, metalic, sheen, specular, clearcoat);

        const auto V = -wi;
        const auto L = wo;
        const auto N = n;
        const auto H = normalize(V + L);

        float pdf = 0.0F;
        pdf += weights[Component::Diffuse] * DisneyBrdfDiffuse::EvalPDF(L, N);
        pdf += weights[Component::Sheen] * DisneyBrdfSheen::EvalPDF(L, N);
        pdf += weights[Component::Specular] * DisneyBrdfSpecular::EvalPDF(roughness, V, L, N);
        pdf += weights[Component::Clearcoat] * DisneyBrdfClearcoat::EvalPDF(clearcoat_gloss, V, L, N);

        return pdf;
    }

    AT_DEVICE_API AT_NAME::MaterialSampling DisneyBRDF::bsdf(
        const aten::MaterialParameter& mtrl,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        const auto roughness = mtrl.standard.roughness;
        const auto metalic = mtrl.standard.metallic;
        const auto sheen = mtrl.standard.sheen;
        const auto specular = mtrl.standard.specular;
        const auto clearcoat = mtrl.standard.clearcoat;
        const auto clearcoat_gloss = mtrl.standard.clearcoatGloss;

        const auto subsurface = mtrl.standard.subsurface;
        const auto sheen_tint = mtrl.standard.sheenTint;
        const auto specular_tint = mtrl.standard.specularTint;

        std::array<float, Component::Num> weights;
        ComputeWeights(weights, mtrl.baseColor, metalic, sheen, specular, clearcoat);

        const auto V = -wi;
        const auto N = n;

        float pdf{ 0.0F };

        aten::vec3 diffuse_brdf{ 0.0F };
        aten::vec3 sheen_brdf{ 0.0F };
        aten::vec3 specular_brdf{ 0.0F };
        aten::vec3 clearcoat_brdf{ 0.0F };

        if (weights[Component::Diffuse] > 0.0F) {
            // Diffuse.
            diffuse_brdf = DisneyBrdfDiffuse::EvalBRDF(mtrl.baseColor, roughness, subsurface, V, wo, N);
            const auto prob = DisneyBrdfDiffuse::EvalPDF(wo, N);
            pdf += prob * weights[Component::Diffuse];
        }
        if (weights[Component::Sheen] > 0.0F) {
            // Sheen.
            sheen_brdf = DisneyBrdfSheen::EvalBRDF(mtrl.baseColor, sheen, sheen_tint, V, wo, N);
            const auto prob = DisneyBrdfSheen::EvalPDF(wo, N);
            pdf += prob * weights[Component::Sheen];
        }
        if (weights[Component::Specular] > 0.0F) {
            // Specular.
            specular_brdf = DisneyBrdfSpecular::EvalBRDF(mtrl.baseColor, roughness, metalic, specular, specular_tint, V, wo, N);
            const auto prob = DisneyBrdfSpecular::EvalPDF(roughness, V, wo, N);
            pdf += prob * weights[Component::Specular];
        }
        if (weights[Component::Clearcoat] > 0.0F) {
            // Cleaercoat.
            clearcoat_brdf = DisneyBrdfClearcoat::EvalBRDF(mtrl.baseColor, clearcoat, clearcoat_gloss, V, wo, N);
            const auto prob = DisneyBrdfClearcoat::EvalPDF(roughness, V, wo, N);
            pdf += prob * weights[Component::Clearcoat];
        }

        AT_NAME::MaterialSampling result;

        result.bsdf = (1 - metalic) * (diffuse_brdf + sheen_brdf) + specular_brdf + clearcoat_brdf;
        result.pdf = pdf;

        return result;
    }

    AT_DEVICE_API void DisneyBRDF::sample(
        AT_NAME::MaterialSampling& result,
        const aten::MaterialParameter& mtrl,
        const aten::vec3& n,
        const aten::vec3& wi,
        aten::sampler* sampler,
        float u, float v)
    {
        const auto r1 = sampler->nextSample();
        const auto r2 = sampler->nextSample();
        const auto r3 = sampler->nextSample();

        const auto roughness = mtrl.standard.roughness;
        const auto metalic = mtrl.standard.metallic;
        const auto sheen = mtrl.standard.sheen;
        const auto specular = mtrl.standard.specular;
        const auto clearcoat = mtrl.standard.clearcoat;
        const auto clearcoat_gloss = mtrl.standard.clearcoatGloss;

        const auto subsurface = mtrl.standard.subsurface;
        const auto sheen_tint = mtrl.standard.sheenTint;
        const auto specular_tint = mtrl.standard.specularTint;

        std::array<float, Component::Num> weights;
        ComputeWeights(weights, mtrl.baseColor, metalic, sheen, specular, clearcoat);

        std::array<float, Component::Num> cdf;
        GetCDF(weights, cdf);

        const auto V = -wi;
        const auto N = n;

        aten::vec3 wo;
        float pdf = 0;

        aten::vec3 diffuse_brdf{ 0.0F };
        aten::vec3 sheen_brdf{ 0.0F };
        aten::vec3 specular_brdf{ 0.0F };
        aten::vec3 clearcoat_brdf{ 0.0F };

        if (r3 < cdf[Component::Diffuse]) {
            // Diffuse.
            wo = DisneyBrdfDiffuse::SampleDirection(r1, r2, N);
            diffuse_brdf = DisneyBrdfDiffuse::EvalBRDF(mtrl.baseColor, roughness, subsurface, V, wo, N);
            pdf = DisneyBrdfDiffuse::EvalPDF(wo, N);
            pdf *= weights[Component::Diffuse];

            // Diffuse is already sampled. So, no need to evaluate with the sample outgoing vector anymore.
            weights[Component::Diffuse] = 0.0F;
        }
        else if (r3 < cdf[Component::Sheen]) {
            // Sheen.
            wo = DisneyBrdfSheen::SampleDirection(r1, r2, N);
            sheen_brdf = DisneyBrdfSheen::EvalBRDF(mtrl.baseColor, sheen, sheen_tint, V, wo, N);
            pdf = DisneyBrdfSheen::EvalPDF(wo, N);
            pdf *= weights[Component::Sheen];

            // Sheen is already sampled. So, no need to evaluate with the sample outgoing vector anymore.
            weights[Component::Sheen] = 0.0F;
        }
        else if (r3 < cdf[Component::Specular]) {
            // Specular.
            wo = DisneyBrdfSpecular::SampleDirection(r1, r2, roughness, V, N);
            specular_brdf = DisneyBrdfSpecular::EvalBRDF(mtrl.baseColor, roughness, metalic, specular, specular_tint, V, wo, N);
            pdf = DisneyBrdfSpecular::EvalPDF(roughness, V, wo, N);
            pdf *= weights[Component::Specular];

            // Specular is already sampled. So, no need to evaluate with the sample outgoing vector anymore.
            weights[Component::Specular] = 0.0F;
        }
        else {
            // Cleaercoat.
            wo = DisneyBrdfClearcoat::SampleDirection(r1, r2, clearcoat_gloss, V, N);
            clearcoat_brdf = DisneyBrdfClearcoat::EvalBRDF(mtrl.baseColor, clearcoat, clearcoat_gloss, V, wo, N);
            pdf = DisneyBrdfClearcoat::EvalPDF(roughness, V, wo, N);
            pdf *= weights[Component::Clearcoat];

            // Clearcoat is already sampled. So, no need to evaluate with the sample outgoing vector anymore.
            weights[Component::Clearcoat] = 0.0F;
        }

        // Evaluate with the sampled outgoing vector.
        // If weight is zero, it means brdf and pdf are already sampled and no need to evaluate.

        if (weights[Component::Diffuse] > 0.0F) {
            // Diffuse.
            diffuse_brdf = DisneyBrdfDiffuse::EvalBRDF(mtrl.baseColor, roughness, subsurface, V, wo, N);
            const auto prob = DisneyBrdfDiffuse::EvalPDF(wo, N);
            pdf += weights[Component::Diffuse] * prob;
        }
        if (weights[Component::Sheen] > 0.0F) {
            // Sheen.
            sheen_brdf = DisneyBrdfSheen::EvalBRDF(mtrl.baseColor, sheen, sheen_tint, V, wo, N);
            const auto prob = DisneyBrdfSheen::EvalPDF(wo, N);
            pdf += weights[Component::Sheen] * prob;
        }
        if (weights[Component::Specular] > 0.0F) {
            // Specular.
            specular_brdf = DisneyBrdfSpecular::EvalBRDF(mtrl.baseColor, roughness, metalic, specular, specular_tint, V, wo, N);
            const auto prob = DisneyBrdfSpecular::EvalPDF(roughness, V, wo, N);
            pdf += weights[Component::Specular] * prob;
        }
        if (weights[Component::Clearcoat] > 0.0F) {
            // Clearcoat.
            clearcoat_brdf = DisneyBrdfClearcoat::EvalBRDF(mtrl.baseColor, clearcoat, clearcoat_gloss, V, wo, N);
            const auto prob = DisneyBrdfClearcoat::EvalPDF(roughness, V, wo, N);
            pdf += weights[Component::Clearcoat] * prob;
        }

        result.pdf = pdf;
        result.bsdf = (1 - metalic) * (diffuse_brdf + sheen_brdf) + specular_brdf + clearcoat_brdf;
        result.dir = wo;
    }
}
