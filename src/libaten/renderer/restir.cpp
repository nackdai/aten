#include "renderer/restir.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/nonphotoreal.h"
#include "renderer/renderer_utility.h"
#include "sampler/cmj.h"

#include "material/lambert.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (28)
#define BREAK_Y    (182)
#pragma optimize( "", off)
#endif

namespace aten
{
    ReSTIR::Path ReSTIR::radiance(
        const context& ctxt,
        sampler* sampler,
        uint32_t maxDepth,
        const ray& inRay,
        camera* cam,
        CameraSampleResult& camsample,
        scene* scene)
    {
        uint32_t depth = 0;
        uint32_t rrDepth = m_rrDepth;

        Path path;
        path.ray = inRay;

        while (depth < maxDepth) {
            path.rec = hitrecord();

            bool willContinue = true;
            Intersection isect;

            if (scene->hit(ctxt, path.ray, AT_MATH_EPSILON, AT_MATH_INF, path.rec, isect)) {
                willContinue = shade(ctxt, sampler, scene, cam, camsample, depth, path);
            }
            else {
                shadeMiss(scene, depth, path);
                willContinue = false;
            }

            if (depth < m_startDepth && !path.isTerminate) {
                path.contrib = vec3(0);
            }

            if (!willContinue) {
                break;
            }

            depth++;
        }

        return path;
    }

    bool ReSTIR::shade(
        const context& ctxt,
        sampler* sampler,
        scene* scene,
        camera* cam,
        CameraSampleResult& camsample,
        int depth,
        Path& path)
    {
        uint32_t rrDepth = m_rrDepth;

        auto mtrl = ctxt.getMaterial(path.rec.mtrlid);

        bool isBackfacing = dot(path.rec.normal, -path.ray.dir) < real(0);

        // 交差位置の法線.
        // 物体からのレイの入出を考慮.
        vec3 orienting_normal = path.rec.normal;

        // Implicit conection to light.
        if (mtrl->isEmissive()) {
            if (!isBackfacing) {
                real weight = 1.0f;

                if (depth > 0
                    && !(path.prevMtrl && path.prevMtrl->isSingularOrTranslucent()))
                {
                    auto cosLight = dot(orienting_normal, -path.ray.dir);
                    auto dist2 = aten::squared_length(path.rec.p - path.ray.org);

                    if (cosLight >= 0) {
                        auto pdfLight = 1 / path.rec.area;

                        // Convert pdf area to sradian.
                        // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
                        // p31 - p35
                        pdfLight = pdfLight * dist2 / cosLight;

                        weight = path.pdfb / (pdfLight + path.pdfb);
                    }
                }

                auto emit = static_cast<aten::vec3>(mtrl->color());
                path.contrib += path.throughput * weight * emit;
            }

            path.isTerminate = true;
            return false;
        }

        if (!mtrl->isTranslucent() && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        // Apply normal map.
        mtrl->applyNormalMap(orienting_normal, orienting_normal, path.rec.u, path.rec.v);

        // Non-Photo-Real.
        if (mtrl->isNPR()) {
            path.contrib = shadeNPR(ctxt, mtrl.get(), path.rec.p, orienting_normal, path.rec.u, path.rec.v, scene, sampler);
            path.isTerminate = true;
            return false;
        }

        // Explicit conection to light.
        if (!mtrl->isSingularOrTranslucent())
        {
            real lightSelectPdf = 1;
            LightSampleResult sampleres;

            auto light = scene->sampleLightWithReservoir(
                ctxt,
                path.rec.p,
                orienting_normal,
                [&](const aten::vec3& dir_to_light) -> aten::vec3 {
                    auto pdf = mtrl->pdf(orienting_normal, path.ray.dir, dir_to_light, path.rec.u, path.rec.v);
                    auto bsdf = mtrl->bsdf(orienting_normal, path.ray.dir, dir_to_light, path.rec.u, path.rec.v);
                    return bsdf / pdf;
                },
                sampler,
                lightSelectPdf, sampleres);

            if (light) {
                const vec3& posLight = sampleres.pos;
                const vec3& nmlLight = sampleres.nml;
                real pdfLight = sampleres.pdf;

                auto lightobj = sampleres.obj;

                vec3 dirToLight = normalize(sampleres.dir);

                // TODO
                // Do we need to consider offset for shadow ray?
#if 0
                auto shadowRayOrg = path.rec.p + AT_MATH_EPSILON * orienting_normal;
                auto tmp = path.rec.p + dirToLight - shadowRayOrg;
                auto shadowRayDir = normalize(tmp);
#else
                auto shadowRayOrg = path.rec.p;
                auto shadowRayDir = dirToLight;
#endif

                if (dot(shadowRayDir, orienting_normal) > real(0)) {
                    aten::ray shadowRay(shadowRayOrg, shadowRayDir, orienting_normal);

                    hitrecord tmpRec;

                    if (scene->hitLight(ctxt, light.get(), posLight, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
                        // Shadow ray hits the light.
                        auto cosShadow = dot(orienting_normal, dirToLight);

                        auto bsdf = mtrl->bsdf(orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);
                        auto pdfb = mtrl->pdf(orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);

                        bsdf *= path.throughput;

                        // Get light color.
                        auto emit = sampleres.finalColor;

                        cosShadow = aten::abs(cosShadow);

                        if (light->isInfinite() || light->isSingular()) {
                            if (cosShadow >= 0) {
                                path.contrib += (bsdf * emit * cosShadow) * lightSelectPdf;
                            }
                        }
                        else {
                            auto cosLight = dot(nmlLight, -dirToLight);

                            if (cosShadow >= 0 && cosLight >= 0) {
                                auto dist2 = squared_length(sampleres.dir);
                                auto G = cosShadow * cosLight / dist2;
                                path.contrib += (bsdf * emit * G) * lightSelectPdf;
                            }
                        }
                    }
                }
            }
        }

        real russianProb = real(1);

        if (depth > rrDepth) {
            auto t = normalize(path.throughput);
            auto p = std::max(t.r, std::max(t.g, t.b));

            russianProb = sampler->nextSample();

            if (russianProb >= p) {
                path.contrib = vec3();
                return false;
            }
            else {
                russianProb = p;
            }
        }

        auto sampling = mtrl->sample(path.ray, orienting_normal, path.rec.normal, sampler, path.rec.u, path.rec.v);

        auto nextDir = normalize(sampling.dir);
        auto pdfb = sampling.pdf;
        auto bsdf = sampling.bsdf;

        // Get normal to add ray offset.
        // In refraction material case, new ray direction might be computed with inverted normal.
        // For example, when a ray go into the refraction surface, inverted normal is used to compute new ray direction.
        auto rayBasedNormal = (!isBackfacing && mtrl->isTranslucent())
            ? -orienting_normal
            : orienting_normal;

        real c = 1;
        if (!mtrl->isSingular()) {
            // TODO
            // AMDのはabsしているが....
            //c = aten::abs(dot(orienting_normal, nextDir));
            c = dot(rayBasedNormal, nextDir);
        }

        if (pdfb > 0 && c > 0) {
            path.throughput *= bsdf * c / pdfb;
            path.throughput /= russianProb;
        }
        else {
            return false;
        }

        path.prevMtrl = mtrl;

        path.pdfb = pdfb;

        // Make next ray.
        path.ray = aten::ray(path.rec.p, nextDir, rayBasedNormal);

        return true;
    }

    void ReSTIR::shadeMiss(
        scene* scene,
        int depth,
        Path& path)
    {
        auto ibl = scene->getIBL();
        aten::vec3 emit(real(0));
        real misW = real(1);

        if (ibl) {
            if (depth == 0) {
                emit = ibl->getEnvMap()->sample(path.ray);
                misW = real(1);
                path.isTerminate = true;
            }
            else {
                emit = ibl->getEnvMap()->sample(path.ray);
                auto pdfLight = ibl->samplePdf(path.ray);
                misW = path.pdfb / (pdfLight + path.pdfb);
            }
        }
        else {
            emit = sampleBG(path.ray);
            misW = real(1);
        }

        path.contrib += path.throughput * misW * emit;
    }

    static uint32_t frame = 0;

    void ReSTIR::onRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
    {
        frame++;

        int width = dst.width;
        int height = dst.height;
        uint32_t samples = dst.sample;

        m_maxDepth = dst.maxDepth;
        m_rrDepth = dst.russianRouletteDepth;
        m_startDepth = dst.startDepth;

        if (m_rrDepth > m_maxDepth) {
            m_rrDepth = m_maxDepth - 1;
        }

        auto time = timer::getSystemTime();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
            auto idx = OMPUtil::getThreadIdx();

            auto t = timer::getSystemTime();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pos = y * width + x;

                    vec3 col = vec3(0);
                    vec3 col2 = vec3(0);
                    uint32_t cnt = 0;

#ifdef RELEASE_DEBUG
                    if (x == BREAK_X && y == BREAK_Y) {
                        DEBUG_BREAK();
                    }
#endif

                    for (uint32_t i = 0; i < samples; i++) {
                        auto scramble = aten::getRandom(pos) * 0x1fe3434f;

                        CMJ rnd;
                        rnd.init(frame, i, scramble);

                        real u = real(x + rnd.nextSample()) / real(width);
                        real v = real(y + rnd.nextSample()) / real(height);

                        auto camsample = camera->sample(u, v, &rnd);

                        auto ray = camsample.r;

                        auto path = radiance(
                            ctxt,
                            &rnd,
                            ray,
                            camera,
                            camsample,
                            scene);

                        if (isInvalidColor(path.contrib)) {
                            AT_PRINTF("Invalid(%d/%d[%d])\n", x, y, i);
                            continue;
                        }

                        auto pdfOnImageSensor = camsample.pdfOnImageSensor;
                        auto pdfOnLens = camsample.pdfOnLens;

                        auto s = camera->getSensitivity(
                            camsample.posOnImageSensor,
                            camsample.posOnLens);

                        auto c = path.contrib * s / (pdfOnImageSensor * pdfOnLens);

                        col += c;
                        col2 += c * c;
                        cnt++;

                        if (path.isTerminate) {
                            break;
                        }
                    }

                    col /= (real)cnt;

                    dst.buffer->put(x, y, vec4(col, 1));

                    if (dst.variance) {
                        col2 /= (real)cnt;
                        dst.variance->put(x, y, vec4(col2 - col * col, real(1)));
                    }
                }
            }
        }
    }
}
