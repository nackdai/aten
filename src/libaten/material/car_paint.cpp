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

        auto flakes_density = FlakesNormal::computeFlakeDensity(
            param->carpaint.flake_size,
            real(1));

        auto diffuse_pdf = lambert::pdf(N, wo);

        auto pdf = fresnel * beckman_pdf + (real(1) - fresnel) * (flakes_density * flakes_beckman_pdf + (1 - flakes_density) * diffuse_pdf);
        pdf = aten::clamp(pdf, real(0), real(1));

        return pdf;
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

        auto flakes_density = FlakesNormal::computeFlakeDensity(
            param->carpaint.flake_size,
            real(1));

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
            const bool is_on_flakes = FlakesNormal::gen(
                u, v,
                param->carpaint.flake_scale,
                param->carpaint.flake_size,
                param->carpaint.flake_size_variance,
                param->carpaint.flake_normal_orientation
            ).a > real(0);

            if (is_on_flakes) {
                // Flakes
                real fresnel{ real(0) };

                bsdf = MicrofacetBeckman::bsdf(
                    param->carpaint.flakes_color * param->carpaint.flake_color_multiplier,
                    real(1),  // TODO
                    real(10),   // TODO
                    fresnel,
                    N,
                    wi, wo,
                    u, v);
            }
            else {
                // Diffuse
                bsdf = param->carpaint.diffuse_color / AT_MATH_PI;
            }
        }

        bsdf = externalAlbedo * bsdf;
        return bsdf;
    }

    AT_DEVICE_MTRL_API void CarPaint::sample(
        AT_NAME::MaterialSampling* result,
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
        AT_NAME::MaterialSampling* result,
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
        const aten::MaterialParameter* param,
        const aten::vec3& orgNml,
        aten::vec3& newNml,
        real u, real v,
        const aten::vec3& wi,
        aten::sampler* sampler)
    {
        const aten::vec3 V = -wi;
        const aten::vec3 N = normalize(orgNml);

        if (sampler) {
            auto r0 = sampler->nextSample();

            auto fresnel = material::computeFresnel(real(1), param->carpaint.clearcoat_ior, V, N);

            if (r0 < fresnel) {
                newNml = N;
            }
            else {
                auto flakes_nml = FlakesNormal::gen(
                    u, v,
                    param->carpaint.flake_scale,
                    param->carpaint.flake_size,
                    param->carpaint.flake_size_variance,
                    param->carpaint.flake_normal_orientation
                );
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
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, clearcoat_ior, real(0.01), real(10));
        auto b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, clearcoat_roughness, real(0.01), real(1.0));
        auto b2 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_scale, real(100), real(1000));
        auto b3 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_size, real(0.1), real(1.0));
        auto b4 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_size_variance, real(0.0), real(1.0));
        auto b5 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_normal_orientation, real(0.0), real(1.0));
        auto b6 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_color_multiplier, real(0.1), real(10.0));
        auto b7 = AT_EDIT_MATERIAL_PARAM(editor, m_param.carpaint, clearcoat_color);
        auto b8 = AT_EDIT_MATERIAL_PARAM(editor, m_param.carpaint, flakes_color);
        auto b9 = AT_EDIT_MATERIAL_PARAM(editor, m_param.carpaint, diffuse_color);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, albedoMap);
        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

        return b0 || b1 || b2 || b3 || b4 || b5 || b6 || b7 || b8 || b9;
    }
}
