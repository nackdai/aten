#include "math/math.h"
#include "material/disney_brdf.h"
#include "material/diffuse.h"
#include "material/ggx.h"

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

    AT_DEVICE_API float DisneyBRDF::pdf(
        const aten::MaterialParameter& mtrl,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        const auto pdf = ComputePDF(mtrl, wi, wo, n);
        return pdf;
    }

    AT_DEVICE_API aten::vec3 DisneyBRDF::sampleDirection(
        const aten::MaterialParameter& mtrl,
        const aten::vec3& n,
        const aten::vec3& wi,
        float u, float v,
        aten::sampler* sampler)
    {
        const auto r1 = sampler->nextSample();
        const auto r2 = sampler->nextSample();
        const auto r3 = sampler->nextSample();

        const auto dir = SampleDirection(r1, r2, r3, mtrl, wi, n);
        return dir;
    }

    AT_DEVICE_API aten::vec3 DisneyBRDF::bsdf(
        const aten::MaterialParameter& mtrl,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        const auto bsdf = ComputeBRDF(mtrl, n, wi, wo);
        return bsdf;
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

        result.dir = SampleDirection(r1, r2, r3, mtrl, wi, n);
        const auto& wo = result.dir;

        result.pdf = ComputePDF(mtrl, wi, wo, n);
        result.bsdf = ComputeBRDF(mtrl, n, wi, wo);
    }

    bool DisneyBRDF::edit(aten::IMaterialParamEditor* editor)
    {
        bool b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, subsurface, 0, 1);
        bool b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, metallic, 0, 1);
        bool b2 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, specular, 0, 1);
        bool b3 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, specularTint, 0, 1);
        bool b4 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, roughness, 0, 1);
        bool b5 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, anisotropic, 0, 1);
        bool b6 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, sheen, 0, 1);
        bool b7 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, sheenTint, 0, 1);
        bool b8 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, clearcoat, 0, 1);
        bool b9 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, clearcoatGloss, 0, 1);
        bool b10 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, float(0.01), float(10));
        bool b11 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

        return b0 || b1 || b2 || b3 || b4 || b5 || b6 || b7 || b8 || b9 || b10 || b11;
    }

    namespace DisneyBrdfUtil {
        static inline AT_DEVICE_API aten::vec3 ComputeCtint(const aten::vec3& base_color)
        {
            const auto Y = dot(base_color, aten::vec3(0.3F, 0.6F, 0.1F));
            const auto Ctint = Y > 0 ? base_color / Y : aten::vec3(1.0F);
            return Ctint;
        }
    };

    class DisneyBrdfDiffuse {
    public:
        static inline AT_DEVICE_API float SchlickFresnel(const float Fd90, const float cos_theta)
        {
            const auto c = aten::saturate(1.0F - cos_theta);
            const auto c5 = c * c * c * c * c;
            return (1.0F + (Fd90 - 1.0F) * c5);
        }

        static inline AT_DEVICE_API aten::vec3 EvalBRDF(
            const aten::vec3& base_color,
            const float roughness,
            const float subsurface,
            const aten::vec3& V,
            const aten::vec3& L,
            const aten::vec3& N,
            const aten::vec3& H)
        {
            const auto LdotH = dot(L, H);

            float fd = 0.0F;

            // Diffuse factor is available.
            if (subsurface < 1.0F) {
                const auto Fd90 = 0.5F + 2 * LdotH * LdotH * roughness;

                const auto FV = SchlickFresnel(Fd90, dot(V, N));
                const auto FL = SchlickFresnel(Fd90, dot(L, N));

                fd = FV * FL;
            }

            float ss = 0.0F;

            // Subsurface factor is available.
            if (subsurface > 0.0F) {
                const auto Fss90 = LdotH * LdotH * roughness;

                const auto FV = SchlickFresnel(Fss90, dot(V, N));
                const auto FL = SchlickFresnel(Fss90, dot(L, N));

                const auto NdotV = dot(V, N);
                const auto NdotL = dot(L, N);

                ss = 1.25F * (FV * FL * (1 / (NdotL + NdotV) - 0.5F) + 0.5F);
            }

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
            const aten::vec3& N,
            const aten::vec3& H)
        {
            // Luminance approximmation.
            const auto Ctint = DisneyBrdfUtil::ComputeCtint(base_color);
            const auto Csheen = aten::mix(aten::vec3(1.0F), Ctint, sheen_tint);

            const auto FH = material::ComputeSchlickFresnelWithF0AndCosTheta(0.0F, dot(L, H));

            const auto Fsheen = sheen * Csheen * FH;
            return Fsheen;
        }

        static inline AT_DEVICE_API float EvalPDF(
            const aten::vec3& L,
            const aten::vec3& N)
        {
            // Uniform hemisphere sampling.
            //return 1 / (AT_MATH_PI_2);

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

    class DisneyBrdfClearcoat {
    public:
        static inline AT_DEVICE_API float D_GTR1(
            const float roughness,
            const aten::vec3& N,
            const aten::vec3& H)
        {
            // NOTE:
            // If roughness is 1, a2 is 1.
            // Then, log(a2) is log(1) and it's zero.
            // It causes zero divide to calculate D.
            // To avoid it, clamp roughness less than 1.

            // Equation (4).
            const auto a = aten::cmpMin(roughness, 0.99F);
            const auto NdotH = aten::abs(dot(N, H));

            const auto a2 = a * a;
            const auto denom = 1 + (a2 - 1) * NdotH * NdotH;

            const auto D = (a2 - 1) / (AT_MATH_PI * aten::log(a2)) / denom;
            return D;
        }

        static inline AT_DEVICE_API aten::vec3 EvalBRDF(
            const aten::vec3& base_color,
            const float clearcoat,
            const float clearcoat_gloss,
            const aten::vec3& V,
            const aten::vec3& L,
            const aten::vec3& N,
            const aten::vec3& H)
        {
            const auto a_clearcoat = clearcoat_gloss;

            const auto D = D_GTR1(a_clearcoat, N, H);

            // clearcoat (ior = 1.5 -> F0 = 0.04).
            const auto FH = material::ComputeSchlickFresnelWithF0AndCosTheta(0.04F, dot(L, H));

            const auto G = MicrofacetGGX::ComputeG2Smith(0.25F, V, L, N);

            const auto f_clearcoat = 0.25F * clearcoat * FH * D * G;
            return aten::vec3(f_clearcoat);
        }

        static inline AT_DEVICE_API float EvalPDF(
            const float clearcoat_gloss,
            const aten::vec3& L,
            const aten::vec3& N,
            const aten::vec3& H)
        {
            const auto a_clearcoat = clearcoat_gloss;

            const auto D = D_GTR1(a_clearcoat, N, H);

            const auto costheta = aten::abs(dot(H, N));

            // For Jacobian |dwh/dwo|
            const auto denom = 4 * aten::abs(dot(L, N));

            const auto pdf = denom > 0 ? (D * costheta) / denom : 0;

            return pdf;
        }

        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2,
            const float clearcoat_gloss,
            const aten::vec3& V,
            const aten::vec3& N)
        {
            const auto a_clearcoat = clearcoat_gloss;

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
            const aten::vec3& N,
            const aten::vec3& H)
        {
            const auto Ctint = DisneyBrdfUtil::ComputeCtint(base_color);
            const auto Cspec = aten::mix(aten::vec3(1), Ctint, specular_tint);

            const auto F_s0 = aten::mix(0.08F * specular * Cspec, base_color, metallic);

            const auto F = material::ComputeSchlickFresnelWithF0AndCosTheta(F_s0, dot(L, H));

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
            const aten::vec3& L,
            const aten::vec3& N,
            const aten::vec3& H)
        {
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
        const float metalic,
        const float sheen,
        const float specular,
        const float clearcoat)
    {
        weights[Component::Diffuse] = 1 - metalic;
        weights[Component::Sheen] = sheen * (1 - metalic);
        weights[Component::Specular] = specular * metalic;
        weights[Component::Clearcoat] = 0.25F * clearcoat;

        float norm = 0.0F;
        for (auto w : weights) {
            norm += w;
        }

        for (auto& w : weights) {
            w /= norm;
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

    AT_DEVICE_API float DisneyBRDF::ComputePDF(
        const aten::MaterialParameter& param,
        const aten::vec3& wi,
        const aten::vec3& wo,
        const aten::vec3& n)
    {
        const auto roughness = param.standard.roughness;
        const auto metalic = param.standard.metallic;
        const auto sheen = param.standard.sheen;
        const auto specular = param.standard.specular;
        const auto clearcoat = param.standard.clearcoat;
        const auto clearcoat_gloss = param.standard.clearcoatGloss;

        std::array<float, Component::Num> weights;
        ComputeWeights(weights, metalic, sheen, specular, clearcoat);

        const auto H = normalize(-wi + wo);

        float pdf = 0.0F;

        if (weights[Component::Diffuse] > 0.0F) {
            pdf += weights[Component::Diffuse] * DisneyBrdfDiffuse::EvalPDF(wo, n);
        }
        if (weights[Component::Sheen] > 0.0F) {
            pdf += weights[Component::Sheen] * DisneyBrdfSheen::EvalPDF(wo, n);
        }

        if (weights[Component::Specular] > 0.0F) {
            pdf += weights[Component::Specular] * DisneyBrdfSpecular::EvalPDF(roughness, wo, n, H);
        }
        if (weights[Component::Clearcoat] > 0.0F) {
            pdf += weights[Component::Clearcoat] * DisneyBrdfClearcoat::EvalPDF(clearcoat_gloss, wo, n, H);
        }

        return pdf;
    }

    AT_DEVICE_API aten::vec3 DisneyBRDF::SampleDirection(
        const float r1, const float r2, const float r3,
        const aten::MaterialParameter& param,
        const aten::vec3& wi,
        const aten::vec3& n)
    {
        const auto roughness = param.standard.roughness;
        const auto metalic = param.standard.metallic;
        const auto sheen = param.standard.sheen;
        const auto specular = param.standard.specular;
        const auto clearcoat = param.standard.clearcoat;
        const auto clearcoat_gloss = param.standard.clearcoatGloss;

        std::array<float, Component::Num> weights;
        ComputeWeights(weights, metalic, sheen, specular, clearcoat);

        std::array<float, Component::Num> cdf;
        GetCDF(weights, cdf);

        const auto V = -wi;
        const auto N = n;

        aten::vec3 wo;

        if (r3 < cdf[Component::Diffuse]) {
            wo = DisneyBrdfDiffuse::SampleDirection(r1, r2, N);
        }
        else if (r3 < cdf[Component::Sheen]) {
            wo = DisneyBrdfSheen::SampleDirection(r1, r2, N);
        }
        else if (r3 < cdf[Component::Specular]) {
            wo = DisneyBrdfSpecular::SampleDirection(r1, r2, roughness, V, N);
        }
        else {
            // Cleaercoat.
            wo = DisneyBrdfClearcoat::SampleDirection(r1, r2, clearcoat_gloss, V, N);
        }

        return wo;
    }

    AT_DEVICE_API aten::vec3 DisneyBRDF::ComputeBRDF(
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo)
    {
        const auto roughness = param.standard.roughness;
        const auto metalic = param.standard.metallic;
        const auto sheen = param.standard.sheen;
        const auto specular = param.standard.specular;
        const auto clearcoat = param.standard.clearcoat;
        const auto clearcoat_gloss = param.standard.clearcoatGloss;

        const auto subsurface = param.standard.subsurface;
        const auto sheen_tint = param.standard.sheenTint;
        const auto specular_tint = param.standard.specularTint;

        const auto V = -wi;
        const auto L = wo;
        const auto N = n;
        const auto H = normalize(V + L);

        const auto f_d = DisneyBrdfDiffuse::EvalBRDF(
            param.baseColor,
            roughness, subsurface,
            V, L, N, H);

        const auto f_sheen = DisneyBrdfSheen::EvalBRDF(
            param.baseColor,
            sheen, sheen_tint,
            V, L, N, H);

        const auto f_spec = DisneyBrdfSpecular::EvalBRDF(
            param.baseColor,
            roughness, metalic,
            specular, specular_tint,
            V, L, N, H);

        const auto f_clearcoat = DisneyBrdfClearcoat::EvalBRDF(
            param.baseColor,
            clearcoat, clearcoat_gloss,
            V, L, N, H);

        const auto f = (f_d + f_sheen) * (1 - metalic) + f_spec + f_clearcoat;

        return f;
    }
}
