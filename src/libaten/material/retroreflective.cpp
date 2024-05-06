#include "material/retroreflective.h"
#include "material/sample_texture.h"
#include "material/beckman.h"
#include "material/lambert.h"

namespace AT_NAME
{
    AT_DEVICE_API float Retroreflective::pdf(
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        const auto pdf = ComputePDF(param, wi, wo, n);
        return pdf;
    }

    AT_DEVICE_API aten::vec3 Retroreflective::sampleDirection(
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi,
        float u, float v,
        aten::sampler* sampler)
    {
        const auto r1 = sampler->nextSample();
        const auto r2 = sampler->nextSample();
        const auto r3 = sampler->nextSample();

        const auto dir = SampleDirection(r1, r2, r3, param, wi, n);
        return dir;
    }

    AT_DEVICE_API aten::vec3 Retroreflective::bsdf(
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo,
        float u, float v)
    {
        const auto bsdf = ComputeBRDF(param, n, wi, wo);
        return bsdf;
    }

    AT_DEVICE_API void Retroreflective::sample(
        AT_NAME::MaterialSampling& result,
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi,
        aten::sampler* sampler,
        float u, float v)
    {
        const auto r1 = sampler->nextSample();
        const auto r2 = sampler->nextSample();
        const auto r3 = sampler->nextSample();

        result.dir = SampleDirection(r1, r2, r3, param, wi, n);
        const auto& wo = result.dir;

        result.pdf = ComputePDF(param, wi, wo, n);
        result.bsdf = ComputeBRDF(param, n, wi, wo);
    }

    bool Retroreflective::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, roughness, 0.01F, 1.0F);
        auto b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, 1.0F, 10.0F);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

        return b0 || b1;
    }

    class RetroreflectiveSurfaceReflection {
    public:
        static inline AT_DEVICE_API aten::vec3 EvalBRDF(
            const float roughness,
            const float ior,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo)
        {
            const auto brdf = MicrofacetBeckman::ComputeBRDF(roughness, ior, n, wi, wo);
            return brdf;
        }

        static inline AT_DEVICE_API float EvalPDF(
            const float roughness,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo)
        {
            const auto pdf = MicrofacetBeckman::ComputePDF(roughness, n, wi, wo);
            return pdf;
        }

        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2,
            const float roughness,
            const aten::vec3& wi,
            const aten::vec3& n)
        {
            const auto wo = MicrofacetBeckman::SampleDirection(roughness, wi, n, r1, r2);
            return wo;
        }
    };

    class RetroreflectiveRetroreflection {
    public:
        static inline AT_DEVICE_API float GetEffectiveRetroreflectiveArea(
            const aten::vec3& into_prismatic_sheet_dir,
            const aten::vec3& surface_normal)
        {
            constexpr std::array ERATable = {
                std::array{aten::Deg2Rad(0.00000F), 0.64754F},
                std::array{aten::Deg2Rad(0.90000F), 0.65542F},
                std::array{aten::Deg2Rad(1.80000F), 0.65597F},
                std::array{aten::Deg2Rad(2.70000F), 0.65809F},
                std::array{aten::Deg2Rad(3.60000F), 0.65676F},
                std::array{aten::Deg2Rad(4.50000F), 0.65617F},
                std::array{aten::Deg2Rad(5.40000F), 0.65473F},
                std::array{aten::Deg2Rad(6.30000F), 0.65207F},
                std::array{aten::Deg2Rad(7.20000F), 0.64913F},
                std::array{aten::Deg2Rad(8.10000F), 0.64519F},
                std::array{aten::Deg2Rad(9.00000F), 0.64118F},
                std::array{aten::Deg2Rad(9.90000F), 0.63707F},
                std::array{aten::Deg2Rad(10.80000F), 0.63161F},
                std::array{aten::Deg2Rad(11.70000F), 0.62889F},
                std::array{aten::Deg2Rad(12.60000F), 0.62211F},
                std::array{aten::Deg2Rad(13.50000F), 0.61503F},
                std::array{aten::Deg2Rad(14.40000F), 0.60473F},
                std::array{aten::Deg2Rad(15.30000F), 0.59359F},
                std::array{aten::Deg2Rad(16.20000F), 0.58159F},
                std::array{aten::Deg2Rad(17.10000F), 0.56907F},
                std::array{aten::Deg2Rad(18.00000F), 0.55633F},
                std::array{aten::Deg2Rad(18.90000F), 0.54344F},
                std::array{aten::Deg2Rad(19.80000F), 0.53001F},
                std::array{aten::Deg2Rad(20.70000F), 0.51531F},
                std::array{aten::Deg2Rad(21.60000F), 0.49711F},
                std::array{aten::Deg2Rad(22.50000F), 0.47744F},
                std::array{aten::Deg2Rad(23.40000F), 0.45818F},
                std::array{aten::Deg2Rad(24.30000F), 0.43884F},
                std::array{aten::Deg2Rad(25.20000F), 0.41917F},
                std::array{aten::Deg2Rad(26.10000F), 0.39954F},
                std::array{aten::Deg2Rad(27.00000F), 0.37793F},
                std::array{aten::Deg2Rad(27.90000F), 0.35501F},
                std::array{aten::Deg2Rad(28.80000F), 0.33171F},
                std::array{aten::Deg2Rad(29.70000F), 0.30684F},
                std::array{aten::Deg2Rad(30.60000F), 0.28187F},
                std::array{aten::Deg2Rad(31.50000F), 0.25732F},
                std::array{aten::Deg2Rad(32.40000F), 0.22999F},
                std::array{aten::Deg2Rad(33.30000F), 0.20212F},
                std::array{aten::Deg2Rad(34.20000F), 0.17373F},
                std::array{aten::Deg2Rad(35.10000F), 0.14399F},
                std::array{aten::Deg2Rad(36.00000F), 0.11725F},
                std::array{aten::Deg2Rad(36.90000F), 0.09801F},
                std::array{aten::Deg2Rad(37.80000F), 0.08237F},
                std::array{aten::Deg2Rad(38.70000F), 0.06934F},
                std::array{aten::Deg2Rad(39.60000F), 0.05785F},
                std::array{aten::Deg2Rad(40.50000F), 0.04836F},
                std::array{aten::Deg2Rad(41.40001F), 0.03978F},
                std::array{aten::Deg2Rad(42.30000F), 0.03220F},
                std::array{aten::Deg2Rad(43.20000F), 0.02613F},
                std::array{aten::Deg2Rad(44.10000F), 0.02063F},
                std::array{aten::Deg2Rad(45.00000F), 0.01595F},
                std::array{aten::Deg2Rad(45.90000F), 0.01213F},
                std::array{aten::Deg2Rad(46.80000F), 0.00893F},
                std::array{aten::Deg2Rad(47.70000F), 0.00630F},
                std::array{aten::Deg2Rad(48.60000F), 0.00445F},
                std::array{aten::Deg2Rad(49.50000F), 0.00273F},
                std::array{aten::Deg2Rad(50.40000F), 0.00157F},
                std::array{aten::Deg2Rad(51.30000F), 0.00081F},
                std::array{aten::Deg2Rad(52.20000F), 0.00036F},
                std::array{aten::Deg2Rad(53.10000F), 0.00012F},
                std::array{aten::Deg2Rad(54.00000F), 0.00001F},
                std::array{aten::Deg2Rad(54.90000F), 0.00000F},
                std::array{aten::Deg2Rad(55.80000F), 0.00000F},
                std::array{aten::Deg2Rad(56.70000F), 0.00000F},
                std::array{aten::Deg2Rad(57.60000F), 0.00000F},
                std::array{aten::Deg2Rad(58.50000F), 0.00000F},
                std::array{aten::Deg2Rad(59.40000F), 0.00000F},
                std::array{aten::Deg2Rad(60.30000F), 0.00000F},
                std::array{aten::Deg2Rad(61.20000F), 0.00000F},
                std::array{aten::Deg2Rad(62.10001F), 0.00000F},
                std::array{aten::Deg2Rad(63.00000F), 0.00000F},
                std::array{aten::Deg2Rad(63.90001F), 0.00000F},
                std::array{aten::Deg2Rad(64.80000F), 0.00000F},
                std::array{aten::Deg2Rad(65.70000F), 0.00000F},
                std::array{aten::Deg2Rad(66.60001F), 0.00000F},
                std::array{aten::Deg2Rad(67.50000F), 0.00000F},
                std::array{aten::Deg2Rad(68.39999F), 0.00000F},
                std::array{aten::Deg2Rad(69.30000F), 0.00000F},
                std::array{aten::Deg2Rad(70.20000F), 0.00000F},
                std::array{aten::Deg2Rad(71.10000F), 0.00000F},
                std::array{aten::Deg2Rad(72.00000F), 0.00000F},
                std::array{aten::Deg2Rad(72.90000F), 0.00000F},
                std::array{aten::Deg2Rad(73.80000F), 0.00000F},
                std::array{aten::Deg2Rad(74.70000F), 0.00000F},
                std::array{aten::Deg2Rad(75.60000F), 0.00000F},
                std::array{aten::Deg2Rad(76.50000F), 0.00000F},
                std::array{aten::Deg2Rad(77.40000F), 0.00000F},
                std::array{aten::Deg2Rad(78.30000F), 0.00000F},
                std::array{aten::Deg2Rad(79.20000F), 0.00000F},
                std::array{aten::Deg2Rad(80.10001F), 0.00000F},
                std::array{aten::Deg2Rad(81.00001F), 0.00000F},
                std::array{aten::Deg2Rad(81.90000F), 0.00000F},
                std::array{aten::Deg2Rad(82.80001F), 0.00000F},
                std::array{aten::Deg2Rad(83.70000F), 0.00000F},
                std::array{aten::Deg2Rad(84.60000F), 0.00000F},
                std::array{aten::Deg2Rad(85.50001F), 0.00000F},
                std::array{aten::Deg2Rad(86.40000F), 0.00000F},
                std::array{aten::Deg2Rad(87.30000F), 0.00000F},
                std::array{aten::Deg2Rad(88.20000F), 0.00000F},
                std::array{aten::Deg2Rad(89.10001F), 0.00000F},
                std::array{aten::Deg2Rad(90.00000F), 0.00000F},
            };
            constexpr auto Step = ERATable.back()[0] / (ERATable.size() - 1);

            // Inverse normal to align the vector into the prismatic sheet.
            const auto c = dot(into_prismatic_sheet_dir, -surface_normal);
            if (c < 0.0F) {
                return 0.0F;
            }

            const auto theta = aten::acos(c);

            const auto idx = static_cast<size_t>(theta / Step);

            float a = 0.0F;
            float b = 0.0F;
            float t = 0.0F;

            if (idx >= ERATable.size()) {
                return 0.0F;
            }
            else {
                // Compute interp factor.
                const auto d = ERATable.at(idx)[0];
                t = aten::cmpMin(1.0F, aten::abs(d - theta) / Step);

                // Obtain interp targets.
                a = ERATable[idx][1];
                if (idx < ERATable.size() - 1) {
                    // Not end of the table.
                    b = ERATable[idx + 1][1];
                }
            }

            const auto result = a * (1 - t) + b * t;
            return result;
        }

        static inline AT_DEVICE_API float ComputeRoughness(
            const float roughness,
            const float ni, const float nt,
            const aten::vec3& wi,
            const aten::vec3& wn)
        {
            const auto uo = -wi;

            // Compute mean of refract vector.
            const auto ut = material::ComputeRefractVector(ni, nt, wi, wn);

            const auto n = nt / ni;

            // Eq.2.
            const auto J1_denom = dot(-wi, wn) + n * dot(ut, wn);
            const auto J1 = J1_denom > 0 ? aten::abs(dot(uo, wn)) / aten::sqr(J1_denom) : 0.0F;

            // Eq.3.
            const auto J2_denom = -n * dot(ut, wn) + dot(uo, wn);
            const auto J2 = J2_denom > 0 ? aten::abs(dot(uo, wn)) / aten::sqr(J2_denom) : 0.0F;

            const auto a = roughness;
            const auto a2 = a * a;

            auto a0 = (J1 > 0 ? a2 / J1 : 0.0F) + (J2 > 0 ? a2 / J2 : 0.0F);
            a0 = aten::sqrt(a0);

            return a0;
        }

        static inline AT_DEVICE_API aten::vec3 EvalBRDF(
            const float roughness,
            const float ior,
            const aten::vec3& wn,
            const aten::vec3& wi,
            const aten::vec3& wo,
            float* used_E,
            float* used_F)
        {
            const auto ni = 1.0F;
            const auto nt = ior;

            const auto uo = -wi;

            // Compute mean of refract vector.
            const auto ut = material::ComputeRefractVector(ni, nt, wi, wn);

            const auto E = GetEffectiveRetroreflectiveArea(ut, wn);
            *used_E = E;

            // Eq.5.
            const auto a = ComputeRoughness(roughness, ni, nt, wi, wn);
            const auto D = MicrofacetBeckman::ComputeDistribution(wo, uo, a);

            // Eq.13.
            auto F = (1.0F - material::ComputeSchlickFresnel(ni, nt, -wi, wn));
            F *= (1.0F - material::ComputeSchlickFresnel(ni, nt, wo, wn));
            *used_F = F;

            // Eq.14.
            auto G = MicrofacetBeckman::ComputeG1(roughness, wi, ut);
            G *= MicrofacetBeckman::ComputeG1(roughness, ut, wo);

            // Eq.12.
            const auto c = aten::abs(dot(wo, wn));
            const auto brdf = c > 0 ? E * F * G * D / c : 0.0F;

            return aten::vec3(brdf);
        }

        static inline AT_DEVICE_API float EvalPDF(
            const float roughness,
            const float ni, const float nt,
            const aten::vec3& wn,
            const aten::vec3& wi,
            const aten::vec3& wo)
        {
            const auto uo = -wi;

            const auto a = ComputeRoughness(roughness, ni, nt, wi, wn);

            // Eq.5.
            const auto D = MicrofacetBeckman::ComputeDistribution(wo, uo, a);

            const auto pdf = D * aten::abs(dot(uo, wo));
            return pdf;
        }

        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2,
            const float roughness,
            const float ni, const float nt,
            const aten::vec3& wi,
            const aten::vec3& wn)
        {
            // According to the paper, NDF is directly applied to output vector.
            // So, apply usual backman imporntance sampling to output vector directly.
            // But, in this case, theta is computed from -wi (= uo) not the macro surface normal.
            const auto uo = -wi;

            const auto a = ComputeRoughness(roughness, ni, nt, wi, wn);
            const auto a2 = a * a;

            // NOTE:
            // log can't accept zero. If r1 is 1, (1 - r1) is zero.
            // To avoid it, if r1 is 1, to lessen r1 less than 1, multipley 0.99.
            const auto theta = aten::atan(aten::sqrt(-a2 * aten::log(1.0F - r1 * 0.99F)));
            const auto phi = AT_MATH_PI_2 * r2;

            const auto t = aten::getOrthoVector(uo);
            const auto b = normalize(aten::cross(uo, t));

            const auto costheta = aten::cos(theta);
            const auto sintheta = aten::sqrt(1.0F - costheta * costheta);

            const auto cosphi = aten::cos(phi);
            const auto sinphi = aten::sin(phi);

            auto wo = t * sintheta * cosphi + b * sintheta * sinphi + uo * costheta;
            wo = normalize(wo);

            return wo;
        }
    };

    class RetroreflectiveDiffuse {
    public:
        static inline AT_DEVICE_API aten::vec3 EvalBRDF(
            const float E,
            const float F,
            const float ni, const float nt,
            const aten::vec3& n,
            const aten::vec3& wi,
            const aten::vec3& wo)
        {
            // NOTE
            constexpr float kd = 1.0F;

            // Eq.15.
            // E and F should come from Retroreflection.
            const auto brdf_0 = F * (1.0F - E) * aten::sqr(ni / nt) * (kd / AT_MATH_PI);

            // Eq.16.
            // Precompute with schlick fresnel.
            // In this case, Fd = (1 - f0) * (-160 / 21).
            auto f0 = (ni - nt) / (ni + nt);
            f0 = f0 * f0;
            const auto Fd = (1.0F - f0) * (-160.0F / 21.0F);

            // Eq.18.
            const auto brdf = brdf_0 / (1.0F - kd * Fd);

            return aten::vec3(brdf);
        }

        static inline AT_DEVICE_API float EvalPDF(
            const aten::vec3& n,
            const aten::vec3& wo)
        {
            // Apply geometric infinite series.
            // https://en.wikipedia.org/wiki/Geometric_series
            const auto original_pdf = lambert::ComputePDF(n, wo);
            const auto pdf = 1.0F / (1.0F - original_pdf);
            return pdf;
        }

        static inline AT_DEVICE_API aten::vec3 SampleDirection(
            const float r1, const float r2,
            const aten::vec3& wn)
        {
            return lambert::SampleDirection(wn, r1, r2);
        }
    };

    AT_DEVICE_API void Retroreflective::ComputeWeights(
        std::array<float, Component::Num>& weights,
        const float ni, const float nt,
        const aten::vec3& wi,
        const aten::vec3& n)
    {
        // TODO:
        // In Beckman, shlick fresnel is computed with half vector and output vector.
        // This API is used to sample output direction.
        // We might not be able to know half vector and output vector.
        // Therefore, we use incident vector and macro surface normal alternatively.
        const auto F = material::ComputeSchlickFresnel(ni, nt, -wi, n);
        weights[Component::SurfaceReflection] = F;

        const auto ut = material::ComputeRefractVector(ni, nt, wi, n);
        const auto E = RetroreflectiveRetroreflection::GetEffectiveRetroreflectiveArea(ut, n);
        weights[Component::RetroReflection] = (1 - F) * E;

        weights[Component::Diffuse] = (1 - F) * (1 - E);

        float norm = 0.0F;
        for (auto w : weights) {
            norm += w;
        }

        for (auto& w : weights) {
            w /= norm;
        }
    }

    AT_DEVICE_API void Retroreflective::GetCDF(
        const std::array<float, Component::Num>& weights,
        std::array<float, Component::Num>& cdf)
    {
        cdf[Component::SurfaceReflection] = weights[Component::SurfaceReflection];
        cdf[Component::RetroReflection] = cdf[Component::SurfaceReflection] + weights[Component::RetroReflection];
        cdf[Component::Diffuse] = cdf[Component::RetroReflection] + weights[Component::Diffuse];
    }

    AT_DEVICE_API float Retroreflective::ComputePDF(
        const aten::MaterialParameter& param,
        const aten::vec3& wi,
        const aten::vec3& wo,
        const aten::vec3& n)
    {
        const auto roughness = param.standard.roughness;
        const auto ior = param.standard.ior;

        const auto ni = 1.0F;
        const auto nt = ior;

        std::array<float, Component::Num> weights;
        ComputeWeights(weights, ni, nt, wi, n);

        float pdf = 0.0F;

        if (weights[Component::SurfaceReflection] > 0.0F) {
            pdf += weights[Component::SurfaceReflection] * RetroreflectiveSurfaceReflection::EvalPDF(roughness, n, wi, wo);
        }
        if (weights[Component::RetroReflection] > 0.0F) {
            pdf += weights[Component::RetroReflection] * RetroreflectiveRetroreflection::EvalPDF(roughness, ni, nt, n, wi, wo);
        }
        if (weights[Component::Diffuse] > 0.0F) {
            pdf += weights[Component::Diffuse] * RetroreflectiveDiffuse::EvalPDF(n, wo);
        }

        return pdf;
    }

    AT_DEVICE_API aten::vec3 Retroreflective::SampleDirection(
        const float r1, const float r2, const float r3,
        const aten::MaterialParameter& param,
        const aten::vec3& wi,
        const aten::vec3& n)
    {
        const auto roughness = param.standard.roughness;
        const auto ior = param.standard.ior;

        const auto ni = 1.0F;
        const auto nt = ior;

        std::array<float, Component::Num> weights;
        ComputeWeights(weights, ni, nt, wi, n);

        std::array<float, Component::Num> cdf;
        GetCDF(weights, cdf);

        aten::vec3 wo;

        if (r3 < cdf[Component::SurfaceReflection]) {
            wo = RetroreflectiveSurfaceReflection::SampleDirection(r1, r2, roughness, wi, n);
        }
        else if (r3 < cdf[Component::RetroReflection]) {
            wo = RetroreflectiveRetroreflection::SampleDirection(r1, r2, roughness, ni, nt, wi, n);
        }
        else {
            wo = RetroreflectiveDiffuse::SampleDirection(r1, r2, n);
        }

        return wo;
    }

    AT_DEVICE_API aten::vec3 Retroreflective::ComputeBRDF(
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi,
        const aten::vec3& wo)
    {
        const auto roughness = param.standard.roughness;
        const auto ior = param.standard.ior;

        const auto ni = 1.0F;
        const auto nt = ior;

        const auto f_r = RetroreflectiveSurfaceReflection::EvalBRDF(
            roughness, ior,
            n, wi, wo);

        float used_E = 0.0F;
        float used_F = 0.0F;

        const auto f_rr = RetroreflectiveRetroreflection::EvalBRDF(
            roughness, ior,
            n, wi, wo,
            &used_E, &used_F);

        const auto f_d = RetroreflectiveDiffuse::EvalBRDF(
            used_E, used_F,
            ni, nt,
            n, wi, wo);

        const auto f = f_r + f_rr + f_d;

        return f;
    }
}
