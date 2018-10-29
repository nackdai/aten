#include "material/carpaint.h"
#include "material/ggx.h"
#include "material/FlakesNormal.h"
#include "material/sample_texture.h"

//#pragma optimize( "", off)

namespace AT_NAME
{
    AT_DEVICE_MTRL_API real CarPaintBRDF::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        const auto ni = real(1);
        const auto nt = param->ior;

        aten::vec3 V = -wi;
        aten::vec3 L = wo;
        aten::vec3 N = normal;
        aten::vec3 H = normalize(L + V);

        real F(1);
        {
            // http://d.hatena.ne.jp/hanecci/20130525/p3

            // NOTE
            // Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
            // R0 = ((n1 - n2) / (n1 + n2))^2

            auto r0 = (ni - nt) / (ni + nt);
            r0 = r0 * r0;

            auto LdotH = aten::abs(dot(L, H));

            F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
        }

        real pdf = real(0);

        {
            const auto m = param->carpaint.clearcoatRoughness;

            const auto n = (2 * AT_MATH_PI) / (4 * m * m) - 1;

            // half vector.
            auto wh = normalize(-wi + wo);

            auto costheta = dot(normal, wh);

            auto c = dot(wo, wh);

            pdf += F * (n + 1) / (2 * AT_MATH_PI) * aten::pow(costheta, n) / (4 * c);
        }

        auto density = FlakesNormal::computeFlakeDensity(param->carpaint.flake_size, real(1280) / 720);

#if 0
        {
            // half vector.
            auto wh = normalize(-wi + wo);

            auto costheta = dot(normal, wh);
            auto sintheta = 1 - costheta * costheta;
            auto tantheta = sintheta / costheta;

            auto c = dot(wo, wh);

            auto cos4 = costheta * costheta * costheta * costheta;
            auto tan2 = tantheta * tantheta;

            const auto m = param->flakeLayerRoughness;

            pdf += (1 - F) * density * 1 / (m * m * cos4) * aten::exp(-tan2 / (m * m)) / (4 * c);
        }
#else
        {
            auto costheta = dot(normal, -wi);
            auto sintheta = 1 - costheta * costheta;
            auto inRefractTheta = aten::asin(ni / nt * sintheta);

            costheta = aten::cos(inRefractTheta);
            sintheta = aten::sin(inRefractTheta);
            auto tantheta = sintheta / costheta;

            // half vector.
            auto wh = normalize(-wi + wo);

            auto c = dot(wo, wh);

            auto cos4 = costheta * costheta * costheta * costheta;

            auto tan2 = tantheta * tantheta;

            const auto m = param->carpaint.flakeLayerRoughness;

            pdf += (1 - F) * density * 1 / (m * m * cos4) * aten::exp(-tan2 / (m * m)) / (4 * c);
        }
#endif

        {
            auto c = dot(normal, wo);
            c = aten::abs(c);

            pdf += (1 - F) * (1 - density) * c / AT_MATH_PI;
        }

        return pdf;
    }

    AT_DEVICE_MTRL_API real CarPaintBRDF::pdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return pdf(&m_param, normal, wi, wo, u, v);
    }

    static inline AT_DEVICE_MTRL_API aten::vec3 computeRefract(
        real nc,
        real nt,
        const aten::vec3& wi,
        const aten::vec3& normal)
    {
        aten::vec3 in = -wi;
        aten::vec3 nml = normal;

        bool into = (dot(in, normal) > real(0));

        if (!into) {
            nml = -nml;
        }

        real nnt = into ? nc / nt : nt / nc;
        real ddn = dot(wi, nml);

        // NOTE
        // https://qiita.com/mebiusbox2/items/315e10031d15173f0aa5

        auto d = dot(in, nml);
        auto refract = -nnt * (in - d * nml) - aten::sqrt(real(1) - nnt * nnt * (1 - ddn * ddn)) * nml;

        return std::move(refract);
    }

    AT_DEVICE_MTRL_API aten::vec3 CarPaintBRDF::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler)
    {
        const auto ni = real(1);
        const auto nt = param->ior;

        const auto m = param->carpaint.clearcoatRoughness;

        const auto n = (2 * AT_MATH_PI) / (4 * m * m) - 1;

        auto r1 = sampler->nextSample();
        auto r2 = sampler->nextSample();

        const auto theta = aten::acos(aten::pow(r1, 1 / (n + 1)));
        const auto phi = 2 * AT_MATH_PI * r2;

        auto N = normal;
        auto T = aten::getOrthoVector(N);
        auto B = cross(N, T);

        auto costheta = aten::cos(theta);
        auto sintheta = aten::sqrt(1 - costheta * costheta);

        auto cosphi = aten::cos(phi);
        auto sinphi = aten::sqrt(1 - cosphi * cosphi);

        auto w = T * sintheta * cosphi + B * sintheta * sinphi + N * costheta;
        w = normalize(w);

        auto dir = wi - 2 * dot(wi, w) * w;

        real F(1);
        {
            aten::vec3 V = -wi;
            aten::vec3 L = dir;
            aten::vec3 N = normal;
            aten::vec3 H = normalize(L + V);

            // http://d.hatena.ne.jp/hanecci/20130525/p3

            // NOTE
            // Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
            // R0 = ((n1 - n2) / (n1 + n2))^2

            auto r0 = (ni - nt) / (ni + nt);
            r0 = r0 * r0;

            auto LdotH = aten::abs(dot(L, H));

            F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
        }

#if 1
        if (r1 < F) {
            // Nothing...
        }
        else {
            r1 -= F;
            r1 /= (1 - F);

            auto density = FlakesNormal::computeFlakeDensity(param->carpaint.flake_size, real(1280) / 720);

            if (r1 < density) {
                r1 /= density;

                auto theta = aten::atan(m * aten::sqrt(-aten::log(1 - r1)));
                auto phi = 2 * AT_MATH_PI * r2;

                auto costheta = aten::cos(theta);
                auto sintheta = aten::sqrt(1 - costheta * costheta);

                auto cosphi = aten::cos(phi);
                auto sinphi = aten::sqrt(1 - cosphi * cosphi);

#if 1
                auto nf = aten::vec3(1, 0, 0) * sintheta * cosphi + aten::vec3(0, 1, 0) * sintheta * sinphi + aten::vec3(0, 0, 1) * costheta;
                nf = normalize(nf);

                auto R = computeRefract(ni, nt, wi, normal);

                dir = R - 2 * dot(R, nf) * nf;

                dir = computeRefract(ni, nt, dir, normal);
#else
                dir = T * sintheta * cosphi + B * sintheta * sinphi + N * costheta;
                dir = normalize(dir);
                dir = computeRefract(ni, nt, dir, normal);
#endif
            }
            else {
                r1 -= density;
                r1 /= (1 - density);

                auto N = normal;
                auto T = aten::getOrthoVector(N);
                auto B = cross(N, T);

                // コサイン項を使った重点的サンプリング.
                r1 = 2 * AT_MATH_PI * r1;
                auto r2s = aten::sqrt(r2);

                const real x = aten::cos(r1) * r2s;
                const real y = aten::sin(r1) * r2s;
                const real z = aten::sqrt(real(1) - r2);

                dir = normalize((T * x + B * y + N * z));

                dir = computeRefract(ni, nt, dir, normal);
            }
        }
#else
        {
            auto theta = aten::atan(m * aten::sqrt(-aten::log(1 - r1)));
            auto phi = 2 * AT_MATH_PI * r2;

            auto costheta = aten::cos(theta);
            auto sintheta = aten::sqrt(1 - costheta * costheta);

            auto cosphi = aten::cos(phi);
            auto sinphi = aten::sqrt(1 - cosphi * cosphi);

            auto nf = aten::vec3(1, 0, 0) * sintheta * cosphi + aten::vec3(0, 1, 0) * sintheta * sinphi + aten::vec3(0, 0, 1) * costheta;
            nf = normalize(nf);

            auto R = computeRefract(ni, nt, wi, normal);

            dir = R - 2 * dot(R, nf) * nf;

            dir = computeRefract(ni, nt, dir, normal);
        }
#endif

        return std::move(dir);
    }

    AT_DEVICE_MTRL_API aten::vec3 CarPaintBRDF::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler) const
    {
        return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
    }

    AT_DEVICE_MTRL_API aten::vec3 CarPaintBRDF::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        auto albedo = param->baseColor;
        albedo *= AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec3(real(1)));

        aten::vec3 ret = bsdf(param, normal, wi, wo, u, v, albedo);
        return std::move(ret);
    }

    AT_DEVICE_MTRL_API aten::vec3 CarPaintBRDF::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        const aten::vec3& externalAlbedo)
    {
        const auto ni = real(1);
        const auto nt = param->ior;

        aten::vec3 V = -wi;
        aten::vec3 L = wo;
        aten::vec3 N = normal;
        aten::vec3 H = normalize(L + V);

        auto NdotH = aten::abs(dot(N, H));
        auto VdotH = aten::abs(dot(V, H));
        auto NdotL = aten::abs(dot(N, L));
        auto NdotV = aten::abs(dot(N, V));

        real F(1);
        {
            // http://d.hatena.ne.jp/hanecci/20130525/p3

            // NOTE
            // Fschlick(v,h) ≒ R0 + (1 - R0)(1 - cosΘ)^5
            // R0 = ((n1 - n2) / (n1 + n2))^2

            auto r0 = (ni - nt) / (ni + nt);
            r0 = r0 * r0;

            auto LdotH = aten::abs(dot(L, H));

            F = r0 + (1 - r0) * aten::pow((1 - LdotH), 5);
        }

        aten::vec3 bsdf(0);

        // Gross.
        {
            const auto m = param->carpaint.clearcoatRoughness;
            const auto n = (2 * AT_MATH_PI) / (4 * m * m) - 1;

            auto G1_lh = MicrofacetGGX::computeGGXSmithG1(m, L, N);
            auto G1_vh = MicrofacetGGX::computeGGXSmithG1(m, V, N);

            auto G = G1_lh * G1_vh;

            real D = (n + 1) / (2 * AT_MATH_PI) * aten::pow(NdotH, n);

            auto denom = 4 * NdotL * NdotV;

            bsdf += denom > AT_MATH_EPSILON ? F * G * D / denom : 0;
        }

        const auto density = FlakesNormal::computeFlakeDensity(param->carpaint.flake_size, real(1280) / 720);

#if 1
        // Glitter.
        {
            auto costheta = dot(normal, -wi);
            auto sintheta = 1 - costheta * costheta;
            auto inRefractTheta = aten::asin(ni / nt * sintheta);

            auto inCostheta = aten::cos(inRefractTheta);
            sintheta = aten::sin(inRefractTheta);
            auto tantheta = sintheta / inCostheta;

            costheta = dot(normal, wo);
            sintheta = 1 - costheta * costheta;
            auto outRefractTheta = aten::asin(ni / nt * sintheta);

            auto outCosTheta = aten::cos(outRefractTheta);

            // half vector.
            auto wh = normalize(-wi + wo);

            auto c = dot(wo, wh);

            auto cos4 = inCostheta * inCostheta * inCostheta * inCostheta;

            auto tan2 = tantheta * tantheta;

            const auto m = param->carpaint.flakeLayerRoughness;

            auto G1_lh = MicrofacetGGX::computeGGXSmithG1(m, L, N);
            auto G1_vh = MicrofacetGGX::computeGGXSmithG1(m, V, N);

            auto G = G1_lh * G1_vh;

            auto D = 1 / (m * m * cos4) * aten::exp(-tan2 / (m * m)) / (4 * c);
            
            auto denom = 4 * inCostheta * outCosTheta;

            auto r = param->carpaint.flake_reflection;
            auto t = aten::cmpMin(param->carpaint.flake_transmittance, 1 - AT_MATH_EPSILON);

            bsdf += denom > AT_MATH_EPSILON ? param->carpaint.glitterColor * density * (1 - F) * (r / (1 - t)) * G * D / denom : aten::vec3(0);
        }
#endif

        // Shade.
        {
            bsdf += (1 - F) * (1 - density) * externalAlbedo / AT_MATH_PI;
        }

        // Sparkle.
        {
            auto flakeNml = FlakesNormal::gen(
                u, v,
                param->carpaint.flake_scale,
                param->carpaint.flake_size,
                param->carpaint.flake_size_variance,
                param->carpaint.flake_normal_orientation);

            if (flakeNml.a > 0) {
                auto T = aten::getOrthoVector(N);
                auto B = cross(N, T);
                flakeNml = normalize(T * flakeNml.x + B * flakeNml.y + N * flakeNml.z);

#if 0
                auto p = aten::clamp(dot(normal, -wi), real(0), real(1));
                auto c = aten::abs(dot((aten::vec3)flakeNml, -wi));
                bsdf += param->flakeColor * (1 - F) * density * c * aten::pow(p, 16);
#else
                aten::vec3 f = flakeNml;

                aten::vec3 n = normalize(normal + f);
                auto c = aten::abs(dot(n, -wi));
            
                bsdf += param->carpaint.flake_intensity * param->carpaint.flakeColor * (1 - F) * density * aten::pow(c, 16);
#endif
            }
        }

        return std::move(bsdf);
    }

    AT_DEVICE_MTRL_API aten::vec3 CarPaintBRDF::bsdf(
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return std::move(bsdf(&m_param, normal, wi, wo, u, v));
    }

    AT_DEVICE_MTRL_API MaterialSampling CarPaintBRDF::sample(
        const aten::ray& ray,
        const aten::vec3& normal,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
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
            sampler,
            u, v,
            isLightPath);

        return std::move(ret);
    }

    AT_DEVICE_MTRL_API void CarPaintBRDF::sample(
        MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real u, real v,
        bool isLightPath/*= false*/)
    {
        auto albedo = param->baseColor;
        albedo *= AT_NAME::sampleTexture(param->albedoMap, u, v, aten::vec3(real(1)));

        sample(
            result,
            param,
            normal,
            wi,
            orgnormal,
            sampler,
            u, v,
            albedo,
            isLightPath);
    }

    AT_DEVICE_MTRL_API void CarPaintBRDF::sample(
        MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real u, real v,
        const aten::vec3& externalAlbedo,
        bool isLightPath/*= false*/)
    {
        result->dir = sampleDirection(param, normal, wi, u, v, sampler);
        
        const auto& wo = result->dir;

        result->pdf = pdf(param, normal, wi, wo, u, v);
        result->bsdf = bsdf(param, normal, wi, wo, u, v, externalAlbedo);
    }

    bool CarPaintBRDF::edit(aten::IMaterialParamEditor* editor)
    {
        bool b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, clearcoatRoughness, 0, 1);
        bool b1 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flakeLayerRoughness, 0, 1);

        bool b2 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_scale, 1, 5000);
        bool b3 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_size, 0.01, 1);
        bool b4 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_size_variance, 0, 1);
        bool b5 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_normal_orientation, 0, 1);
        
        bool b6 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_reflection, 0, 1);
        bool b7 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_transmittance, 0, 1);

        bool b8 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param.carpaint, flake_intensity, 0, 10);

        bool b9 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, ior, real(0.01), real(10));

        bool b10 = AT_EDIT_MATERIAL_PARAM(editor, m_param.carpaint, glitterColor);
        bool b11 = AT_EDIT_MATERIAL_PARAM(editor, m_param.carpaint, flakeColor);
        bool b12 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        return b0 || b1 || b2 || b3 || b4 || b5 || b6 || b7 || b8 || b9 || b10 || b11 || b12;
    }
}
