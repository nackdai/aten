#include "material/refraction.h"

//#pragma optimize( "", off)

namespace AT_NAME
{
    AT_DEVICE_API real refraction::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        AT_ASSERT(false);
        return 1.0F;
    }

    AT_DEVICE_API aten::vec3 refraction::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler)
    {
        AT_ASSERT(false);
        return aten::vec3();
    }

    AT_DEVICE_API aten::vec3 refraction::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        AT_ASSERT(false);
        return aten::vec3();
    }

    AT_DEVICE_API void refraction::sample(
        AT_NAME::MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler)
    {
        SampleRefraction(*result, sampler, *param, normal, wi);
    }

    bool refraction::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, real(0.01), real(10));
        auto b1 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

        return b0 || b1;
    }

    AT_DEVICE_API void refraction::SampleRefraction(
        AT_NAME::MaterialSampling& result,
        aten::sampler* sampler,
        const aten::MaterialParameter& param,
        const aten::vec3& n,
        const aten::vec3& wi)
    {
        auto ni = 1.0F;
        auto nt = param.standard.ior;

        const auto V = -wi;
        auto N = n;

        const auto is_enter = dot(V, N) >= 0.0F;

        if (!is_enter) {
            N = -n;
            // NOTE
            // In cuda 11, std::swap isn't supported.
            auto swap = ni;
            ni = nt;
            nt = swap;
        }

        // NOTE:
        // cos_t^2 = 1 - sin_t^2
        // sin_i / sin_t = nt/ni <=> sin_t = (ni/nt) * sin_i
        // sin_t^2 = (ni/nt)^2 * sin_i^2
        //         = (ni/nt)^2 * (1 - cos_i^2)
        // cos_t^2 = 1 - sin_t^2 = 1 - (ni/nt)^2 * (1 - cos_i^2)
        const auto ni_nt = ni / nt;
        const auto cos_i = dot(V, N);
        const auto cos_t_2 = 1.0F - (ni_nt * ni_nt * (1.0F - cos_i * cos_i));

        if (cos_t_2 < 0.0F) {
            // Relection.
            result.pdf = specular::ComputeProbabilityToSampleOutputVector();
            result.dir = specular::SampleDirection(wi, N);
            result.bsdf = specular::ComputeBRDF(result.dir, N);
            return;
        }

        auto wo = ComputeRefractVector(ni, nt, wi, N);
        const auto F = material::ComputeSchlickFresnel(ni, nt, wo, N);

        const auto R = F;       // reflectance.
        const auto T = 1 - R;   // transmittance.

        if (param.isIdealRefraction) {
            // Regardless any reflectance, refraction happens.
            const auto c = aten::abs(dot(wo, N));

            // https://cgg.mff.cuni.cz/~jaroslav/teaching/2017-npgr010/slides/03%20-%20npgr010-2017%20-%20BRDF.pdf#page=48.00
            const auto nt_ni = nt / ni;
            const auto bsdf = c == 0.0F ? 0.0F : (nt_ni * nt_ni) * T / c;

            result.pdf = 1.0F;
            result.dir = wo;
            result.bsdf = aten::vec3(bsdf);
            return;
        }

        // Caribration to decrease probability for reflection.
        const auto prob = 0.25F + 0.5F * R;

        const auto u = sampler->nextSample();

        if (u < prob) {
            // Reflection.
            wo = material::ComputeReflectVector(wi, N);

            // NOTE
            // https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#SpecularReflection
            const auto c = aten::abs(dot(wo, N));

            const auto bsdf = c == 0.0F ? 0.0F : R / c;

            result.pdf = prob;
            result.dir = wo;
            result.bsdf = aten::vec3(bsdf);
        }
        else {
            // Refraction.
            const auto c = aten::abs(dot(wo, N));

            // https://cgg.mff.cuni.cz/~jaroslav/teaching/2017-npgr010/slides/03%20-%20npgr010-2017%20-%20BRDF.pdf#page=48.00
            const auto nt_ni = nt / ni;
            const auto bsdf = c == 0.0F ? 0.0F : (nt_ni * nt_ni) * T / c;

            result.pdf = 1.0F - prob;
            result.dir = wo;
            result.bsdf = aten::vec3(bsdf);
        }
    }
}
