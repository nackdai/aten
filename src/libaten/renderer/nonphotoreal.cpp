#include "renderer/nonphotoreal.h"

namespace aten
{
    vec3 shadeNPR(
        const context& ctxt,
        const material* mtrl,
        const vec3& p,
        const vec3& normal,
        real u, real v,
        scene* scene,
        sampler* sampler)
    {
        AT_ASSERT(mtrl->isNPR());
        const NPRMaterial* nprMtrl = reinterpret_cast<const NPRMaterial*>(mtrl);

        auto light = nprMtrl->getTargetLight();
        AT_ASSERT(light);

        real cosShadow = 0;

        if (light) {
            auto sampleres = light->sample(ctxt, p, sampler);

            vec3 posLight = sampleres.pos;
            vec3 nmlLight = sampleres.nml;
            real pdfLight = sampleres.pdf;

            auto lightobj = sampleres.obj;

            vec3 dirToLight = normalize(sampleres.dir);
            aten::ray shadowRay(p, dirToLight, normal);

            hitrecord tmpRec;

            if (scene->hitLight(ctxt, light.get(), posLight, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
                cosShadow = dot(normal, dirToLight);
            }
        }

        auto ret = nprMtrl->bsdf(cosShadow, u, v);

        return ret;
    }
}
