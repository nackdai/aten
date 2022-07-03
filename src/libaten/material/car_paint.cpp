#include "material/car_paint.h"
#include "material/sample_texture.h"
#include "material/FlakesNormal.h"
#include "material/lambert.h"
#include "material/beckman.h"

namespace AT_NAME
{
    // MEMO
    // ApplyNormalでflakes_normalを計算して、外に渡しているので、内部的に計算不要？

    // TODO
    // Standardize API.
    inline AT_DEVICE_MTRL_API void applyTangentSpaceCoord(const aten::vec3& nml, const aten::vec3& src, aten::vec3& dst)
    {
        aten::vec3 n = normalize(nml);
        aten::vec3 t = aten::getOrthoVector(n);
        aten::vec3 b = cross(n, t);

        dst = src.z * n + src.x * t + src.y * b;
        dst = normalize(dst);
    }

    AT_DEVICE_MTRL_API real CarPaint::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        const auto& V = -wi;
        const auto& N = normal;

        auto fresnel = material::computeFresnel(
            real(1), param->carpaint.clearcoat_ior,
            V, N);

        auto beckman_pdf = MicrofacetBeckman::pdf(
            param->carpaint.clearcoat_roughness,
            N, wi, wo);

        auto flakes_beckman_pdf = MicrofacetBeckman::pdf(
            real(1),  // TODO
            N, wi, wo);

        // TODO
        // Enable to specify flake size.
        auto flakes_density = FlakesNormal::computeFlakeDensity(real(0.25), real(1));

        auto diffuse_pdf = lambert::pdf(N, wo);

        auto pdf = fresnel * beckman_pdf + (real(1) - fresnel) * (flakes_density * flakes_beckman_pdf + (1 - flakes_density) * diffuse_pdf);
        pdf = aten::clamp(pdf, real(0), real(1));

        return pdf;
    }

    AT_DEVICE_MTRL_API real CarPaint::pdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return pdf(&m_param, normal, wi, wo, u, v);
    }

    AT_DEVICE_MTRL_API aten::vec3 CarPaint::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler,
        real pre_sampled_r)
    {
        const aten::vec3 V = -wi;
        const aten::vec3 N = normal;

        auto r0 = pre_sampled_r;
        auto r1 = sampler->nextSample();

        auto fresnel = material::computeFresnel(
            real(1), param->carpaint.clearcoat_ior,
            V, N);

        // TODO
        // Enable to specify flake size.
        auto flakes_density = FlakesNormal::computeFlakeDensity(real(0.25), real(1));

        aten::vec3 dir;

        if (r0 < fresnel) {
            r0 /= fresnel;
            dir = MicrofacetBeckman::sampleDirection(
                param->carpaint.clearcoat_roughness,
                wi, N,
                r0, r1);
        }
        else {
            r0 -= fresnel;
            r0 /= (real(1) - fresnel);

            if (r1 < flakes_density) {
                // Flakes
                r1 /= flakes_density;
                dir = MicrofacetBeckman::sampleDirection(real(1), wi, N, r0, r1);
            }
            else {
                // Diffuse
                r1 -= flakes_density;
                r1 /= (real(1) - flakes_density);
                dir = lambert::sampleDirection(N, r0, r1);
            }
        }

        return dir;
    }

    AT_DEVICE_MTRL_API aten::vec3 CarPaint::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler,
        real pre_sampled_r) const
    {
        const aten::vec3& in = ray.dir;

        return sampleDirection(&m_param, normal, in, u, v, sampler, pre_sampled_r);
    }

    AT_DEVICE_MTRL_API aten::vec3 CarPaint::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        real pre_sampled_r)
    {
        const auto albedo = AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec4(real(1)));
        return bsdf(param, normal, wi, wo, u, v, albedo, pre_sampled_r);
    }

    AT_DEVICE_MTRL_API aten::vec3 CarPaint::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        const aten::vec3& externalAlbedo,
        real pre_sampled_r)
    {
        const aten::vec3 V = -wi;
        const aten::vec3 L = wo;
        const aten::vec3 N = normal;
        const aten::vec3 H = normalize(L + V);

        auto fresnel = material::computeFresnel(
            real(1), param->carpaint.clearcoat_ior,
            V, N);

        aten::vec3 bsdf;

        if (pre_sampled_r < fresnel) {
            bsdf = MicrofacetBeckman::bsdf(
                param->carpaint.clearcoat_color,
                param->carpaint.clearcoat_roughness,
                param->carpaint.clearcoat_ior,
                fresnel,
                N,
                wi, wo,
                u, v);
        }
        else {
            const bool is_on_flakes = FlakesNormal::gen(u, v).a > real(0);

            if (is_on_flakes) {
                // Flakes

                // TODO
                // flakes color
                const aten::vec3 flakes_color(real(3), real(3), real(0));

                real fresnel{ real(0) };

                bsdf = MicrofacetBeckman::bsdf(
                    flakes_color,
                    real(1),  // TODO
                    real(10),   // TODO
                    fresnel,
                    N,
                    wi, wo,
                    u, v);
            }
            else {
                // Diffuse
                bsdf = param->baseColor / AT_MATH_PI;
            }
        }

        bsdf = externalAlbedo * bsdf;
        return bsdf;
    }

    AT_DEVICE_MTRL_API aten::vec3 CarPaint::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        real pre_sampled_r) const
    {
        return bsdf(&m_param, normal, wi, wo, u, v, pre_sampled_r);
    }

    AT_DEVICE_MTRL_API MaterialSampling CarPaint::sample(
        const aten::ray& ray,
        const aten::vec3& normal,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real pre_sampled_r,
        real u, real v,
        bool isLightPath/*= false*/) const
    {
        MaterialSampling ret;

        sample(
            &ret,
            &m_param,
            normal,
            ray.dir,
            orgnormal,
            sampler, pre_sampled_r,
            u, v,
            isLightPath);

        return ret;
    }

    AT_DEVICE_MTRL_API void CarPaint::sample(
        MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real pre_sampled_r,
        real u, real v,
        bool isLightPath/*= false*/)
    {
        result->dir = sampleDirection(param, normal, wi, u, v, sampler, pre_sampled_r);
        result->pdf = pdf(param, normal, wi, result->dir, u, v);
        result->bsdf = bsdf(param, normal, wi, result->dir, u, v, pre_sampled_r);
    }

    AT_DEVICE_MTRL_API void CarPaint::sample(
        MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real pre_sampled_r,
        real u, real v,
        const aten::vec3& externalAlbedo,
        bool isLightPath/*= false*/)
    {
        result->dir = sampleDirection(param, normal, wi, u, v, sampler, pre_sampled_r);
        result->pdf = pdf(param, normal, wi, result->dir, u, v);
        result->bsdf = bsdf(param, normal, wi, result->dir, u, v, externalAlbedo, pre_sampled_r);
    }

    AT_DEVICE_MTRL_API real CarPaint::applyNormalMap(
        const aten::vec3& orgNml,
        aten::vec3& newNml,
        real u, real v,
        const aten::vec3& wi,
        aten::sampler* sampler) const
    {
        const aten::vec3 V = -wi;
        const aten::vec3 N = normalize(orgNml);

        if (sampler) {
            auto r0 = sampler->nextSample();

            auto fresnel = material::computeFresnel(real(1), m_param.carpaint.clearcoat_ior, V, N);

            if (r0 < fresnel) {
                newNml = N;
            }
            else {
                auto flakes_nml = FlakesNormal::gen(u, v);
                if (flakes_nml.a > real(0)) {
                    applyTangentSpaceCoord(orgNml, flakes_nml, newNml);
                }
                else {
                    newNml = N;
                }
            }
            return r0;
        }
        else {
            newNml = N;
            return real(0);
        }
    }

    bool CarPaint::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.standard, ior, real(0.01), real(10));
        auto b1 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

        return b0 || b1;
    }
}
