#include "material/refraction.h"
#include "scene/hitable.h"
#include "material/sample_texture.h"

//#pragma optimize( "", off)

namespace AT_NAME
{
    AT_DEVICE_MTRL_API real refraction::pdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        AT_ASSERT(false);

        auto ret = real(1);
        return ret;
    }

    AT_DEVICE_MTRL_API real refraction::pdf(
        const aten::vec3& normal, 
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return pdf(&m_param, normal, wi, wo, u, v);
    }

    AT_DEVICE_MTRL_API aten::vec3 refraction::sampleDirection(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        real u, real v,
        aten::sampler* sampler)
    {
        AT_ASSERT(false);

        aten::vec3 in = -wi;
        aten::vec3 nml = normal;

        bool into = (dot(in, normal) > real(0));

        if (!into) {
            nml = -nml;
        }

        auto reflect = wi - 2 * dot(nml, wi) * nml;
        reflect = normalize(reflect);

        return std::move(reflect);
    }

    AT_DEVICE_MTRL_API aten::vec3 refraction::sampleDirection(
        const aten::ray& ray,
        const aten::vec3& normal,
        real u, real v,
        aten::sampler* sampler) const
    {
        return std::move(sampleDirection(&m_param, normal, ray.dir, u, v, sampler));
    }

    AT_DEVICE_MTRL_API aten::vec3 refraction::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v)
    {
        AT_ASSERT(false);

        aten::vec3 albedo = param->baseColor;
        albedo *= sampleTexture(
            param->albedoMap,
            u, v,
            aten::vec3(real(1)));

        return std::move(albedo);
    }

    AT_DEVICE_MTRL_API aten::vec3 refraction::bsdf(
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v,
        const aten::vec3& externalAlbedo)
    {
        AT_ASSERT(false);

        aten::vec3 albedo = param->baseColor;
        albedo *= externalAlbedo;

        return std::move(albedo);
    }

    AT_DEVICE_MTRL_API aten::vec3 refraction::bsdf(
        const aten::vec3& normal, 
        const aten::vec3& wi,
        const aten::vec3& wo,
        real u, real v) const
    {
        return std::move(bsdf(&m_param, normal, wi, wo, u, v));
    }

    MaterialSampling refraction::sample(
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

    AT_DEVICE_MTRL_API void refraction::sample(
        MaterialSampling* result,
        const aten::MaterialParameter* param,
        const aten::vec3& normal,
        const aten::vec3& wi,
        const aten::vec3& orgnormal,
        aten::sampler* sampler,
        real u, real v,
        bool isLightPath/*= false*/)
    {
        aten::vec3 in = -wi;
        aten::vec3 nml = normal;

        bool into = (dot(in, normal) > real(0));

        if (!into) {
            nml = -nml;
        }

        auto reflect = wi - 2 * dot(nml, wi) * nml;
        reflect = normalize(reflect);

        real nc = real(1);        // ^‹ó‚Ì‹üÜ—¦.
        real nt = param->ior;    // •¨‘Ì“à•”‚Ì‹üÜ—¦.
        real nnt = into ? nc / nt : nt / nc;
        real ddn = dot(wi, nml);

        // NOTE
        // cos_t^2 = 1 - sin_t^2
        // sin_t^2 = (nc/nt)^2 * sin_i^2
        //         = (nc/nt)^2 * (1 - cos_i^2)
        // sin_i / sin_t = nt/nc
        //   -> sin_t = (nc/nt) * sin_i
        //            = (nc/nt) * sqrt(1 - cos_i)
        real cos2t = real(1) - nnt * nnt * (real(1) - ddn * ddn);

        aten::vec3 albedo = param->baseColor;

        if (cos2t < real(0)) {
            //AT_PRINTF("Reflection in refraction...\n");

            // ‘S”½ŽË.
            result->pdf = real(1);
            result->bsdf = albedo;
            result->dir = reflect;
            result->fresnel = real(1);

            return;
        }
#if 0
        aten::vec3 refract = wi * nnt - normal * (into ? real(1) : real(-1)) * (ddn * nnt + sqrt(cos2t));
#elif 0
        // NOTE
        // https://www.vcl.jp/~kanazawa/raytracing/?page_id=478

        auto invnnt = 1 / nnt;
        aten::vec3 refract = nnt * (wi - (aten::sqrt(invnnt * invnnt - (1 - ddn * ddn)) - ddn) * nml);
#else
        // NOTE
        // https://qiita.com/mebiusbox2/items/315e10031d15173f0aa5

        auto d = dot(in, nml);
        auto refract = -nnt * (in - d * nml) - aten::sqrt(real(1) - nnt * nnt * (1 - d * d)) * nml;
#endif
        refract = normalize(refract);

        // Schlick‚É‚æ‚éFresnel‚Ì”½ŽËŒW”‚Ì‹ßŽ—‚ðŽg‚¤.
        const real a = nt - nc;
        const real b = nt + nc;
        const real r0 = (a * a) / (b * b);

        const real c = 1 - (into ? -ddn : dot(refract, -nml));

        // ”½ŽË•ûŒü‚ÌŒõ‚ª”½ŽË‚µ‚Äray.dir‚Ì•ûŒü‚É‰^‚ÔŠ„‡B“¯Žž‚É‹üÜ•ûŒü‚ÌŒõ‚ª”½ŽË‚·‚é•ûŒü‚É‰^‚ÔŠ„‡.
        const real fresnel = r0 + (1 - r0) * aten::pow(c, 5);

        auto Re = fresnel;
        auto Tr = (1 - Re);

        real r = real(0.5);
        if (sampler) {
            r = sampler->nextSample();
        }

        if (param->isIdealRefraction) {
            result->dir = refract;

            // ƒŒƒC‚Ì‰^‚Ô•úŽË‹P“x‚Í‹üÜ—¦‚ÌˆÙ‚È‚é•¨‘ÌŠÔ‚ðˆÚ“®‚·‚é‚Æ‚«A‹üÜ—¦‚Ì”ä‚Ì“ñæ‚Ì•ª‚¾‚¯•Ï‰»‚·‚é.
            if (isLightPath) {
                // TODO
                // —vŠm”F...
                nnt = into ? nt / nc : nc / nt;
            }
            else {
                nnt = into ? nc / nt : nt / nc;
            }

            real nnt2 = nnt * nnt;

            result->bsdf = nnt2 * Tr * albedo;

            result->fresnel = 0;

            result->subpdf = real(1);
        }
        else {
            auto prob = real(0.25) + real(0.5) * Re;

            if (r < prob) {
                // ”½ŽË.
                result->dir = reflect;
                result->bsdf = Re * albedo;
                result->bsdf /= prob;

                result->subpdf = prob;

                result->fresnel = Re;
            }
            else {
                // ‹üÜ.
                result->dir = refract;

                // ƒŒƒC‚Ì‰^‚Ô•úŽË‹P“x‚Í‹üÜ—¦‚ÌˆÙ‚È‚é•¨‘ÌŠÔ‚ðˆÚ“®‚·‚é‚Æ‚«A‹üÜ—¦‚Ì”ä‚Ì“ñæ‚Ì•ª‚¾‚¯•Ï‰»‚·‚é.
                if (isLightPath) {
                    // TODO
                    // —vŠm”F...
                    nnt = into ? nt / nc : nc / nt;
                }
                else {
                    nnt = into ? nc / nt : nt / nc;
                }

                real nnt2 = nnt * nnt;

                result->bsdf = nnt2 * Tr * albedo;
                result->bsdf /= (1 - prob);

                result->subpdf = (1 - prob);

                result->fresnel = Tr * nnt2;;
            }
        }

        result->pdf = 1;
    }

    refraction::RefractionSampling refraction::check(
        const material* mtrl,
        const aten::vec3& in,
        const aten::vec3& normal,
        const aten::vec3& orienting_normal)
    {
        if (!mtrl->isSingular() || !mtrl->isTranslucent()) {
            return std::move(RefractionSampling(false, real(0), real(0)));
        }

        // ƒŒƒC‚ª“üŽË‚µ‚Ä‚­‚é‘¤‚Ì•¨‘Ì‚Ì‹üÜ—¦.
        real ni = real(1);    // ^‹ó

        // •¨‘Ì“à•”‚Ì‹üÜ—¦.
        real nt = mtrl->ior();

        bool into = (dot(normal, orienting_normal) > real(0));

        auto reflect = in - 2 * dot(normal, in) * normal;
        reflect = normalize(reflect);

        real cos_i = dot(in, normal);
        real nnt = into ? ni / nt : nt / ni;

        // NOTE
        // cos_t^2 = 1 - sin_t^2
        // sin_t^2 = (nc/nt)^2 * sin_i^2 = (nc/nt)^2 * (1 - cos_i^2)
        // sin_i / sin_t = nt/nc -> sin_t = (nc/nt) * sin_i = (nc/nt) * sqrt(1 - cos_i)
        real cos_t_2 = real(1) - (nnt * nnt) * (real(1) - cos_i * cos_i);

        aten::vec3 albedo = mtrl->color();

        if (cos_t_2 < real(0)) {
            return std::move(RefractionSampling(false, real(1), real(0)));
        }

        aten::vec3 n = into ? normal : -normal;
#if 0
        aten::vec3 refract = in * nnt - hitrec.normal * (into ? 1.0 : -1.0) * (cos_i * nnt + sqrt(cos_t_2));
#else
        // NOTE
        // https://www.vcl.jp/~kanazawa/raytracing/?page_id=478

        auto invnnt = 1 / nnt;
        aten::vec3 refract = nnt * (in - (aten::sqrt(invnnt * invnnt - (1 - cos_i * cos_i)) - (-cos_i)) * normal);
#endif
        refract = normalize(refract);

        const auto r0 = ((nt - ni) * (nt - ni)) / ((nt + ni) * (nt + ni));

        const auto c = 1 - (into ? -cos_i : dot(refract, -normal));

        // ”½ŽË•ûŒü‚ÌŒõ‚ª”½ŽË‚µ‚Äray.dir‚Ì•ûŒü‚É‰^‚ÔŠ„‡B“¯Žž‚É‹üÜ•ûŒü‚ÌŒõ‚ª”½ŽË‚·‚é•ûŒü‚É‰^‚ÔŠ„‡.
        auto fresnel = r0 + (1 - r0) * aten::pow(c, 5);

        // ƒŒƒC‚Ì‰^‚Ô•úŽË‹P“x‚Í‹üÜ—¦‚ÌˆÙ‚È‚é•¨‘ÌŠÔ‚ðˆÚ“®‚·‚é‚Æ‚«A‹üÜ—¦‚Ì”ä‚Ì“ñæ‚Ì•ª‚¾‚¯•Ï‰»‚·‚é.
        real nn = nnt * nnt;

        auto Re = fresnel;
        auto Tr = (1 - Re) * nn;

        const refraction* refr = reinterpret_cast<const refraction*>(mtrl);

        if (refr->isIdealRefraction()) {
            return std::move(RefractionSampling(true, real(0), real(1), true));
        }
        else {
            auto prob = 0.25 + 0.5 * Re;
            return std::move(RefractionSampling(true, real(prob), real(1 - prob)));
        }
    }

    bool refraction::edit(aten::IMaterialParamEditor* editor)
    {
        auto b0 = AT_EDIT_MATERIAL_PARAM_RANGE(editor, m_param, ior, real(0.01), real(10));
        auto b1 = AT_EDIT_MATERIAL_PARAM(editor, m_param, baseColor);

        AT_EDIT_MATERIAL_PARAM_TEXTURE(editor, m_param, normalMap);

        return b0 || b1;
    }
}
