#include "renderer/pathtracing.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/nonphotoreal.h"
#include "renderer/renderer_utility.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/wanghash.h"
#include "sampler/cmj.h"
#include "sampler/bluenoiseSampler.h"

#include "material/lambert.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (28)
#define BREAK_Y    (182)
#pragma optimize( "", off)
#endif

namespace aten
{
    PathTracing::Path PathTracing::radiance(
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

    bool PathTracing::shade(
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

        auto multipliedAlbedo = mtrl->sampleMultipliedAlbedo(path.rec.u, path.rec.v);
        AlphaBlendedMaterialSampling smplAlphaBlend;
        auto isAlphaBlended = material::sampleAlphaBlend(
            smplAlphaBlend,
            path.accumulatedAlpha,
            multipliedAlbedo,
            path.ray,
            path.rec.p,
            orienting_normal,
            sampler,
            path.rec.u, path.rec.v);

        if (isAlphaBlended) {
            path.prevMtrl = mtrl;
            path.pdfb = smplAlphaBlend.pdf;

            path.ray = smplAlphaBlend.ray;

            path.throughput += smplAlphaBlend.bsdf;
            path.accumulatedAlpha *= real(1) - smplAlphaBlend.alpha;

            return true;
        }

        if (!mtrl->isTranslucent() && isBackfacing) {
            orienting_normal = -orienting_normal;
        }

        // Apply normal map.
        mtrl->applyNormalMap(orienting_normal, orienting_normal, path.rec.u, path.rec.v);

#if 0
        if (depth == 0) {
            auto Wdash = cam->getWdash(
                path.rec.p,
                camsample.posOnImageSensor,
                camsample.posOnLens,
                camsample.posOnObjectplane);
            auto areaPdf = cam->getPdfImageSensorArea(
                path.rec.p, orienting_normal,
                camsample.posOnImageSensor,
                camsample.posOnLens,
                camsample.posOnObjectplane);

            path.throughput *= Wdash;
            path.throughput /= areaPdf;
        }
#endif

        // Non-Photo-Real.
        if (mtrl->isNPR()) {
            path.contrib = shadeNPR(ctxt, mtrl.get(), path.rec.p, orienting_normal, path.rec.u, path.rec.v, scene, sampler);
            path.isTerminate = true;
            return false;
        }

        if (m_virtualLight) {
            if (mtrl->isGlossy()
                && (path.prevMtrl && !path.prevMtrl->isGlossy()))
            {
                return false;
            }
        }

        // Explicit conection to light.
        if (!mtrl->isSingularOrTranslucent())
        {
            real lightSelectPdf = 1;
            LightSampleResult sampleres;

            auto light = scene->sampleLight(
                ctxt,
                path.rec.p,
                orienting_normal,
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

                        if (light->isInfinite() || light->isSingular()) {
                            if (pdfLight > real(0) && cosShadow >= 0) {
                                auto misW = light->isSingular()
                                    ? real(1)
                                    : aten::computeBalanceHeuristic(pdfLight * lightSelectPdf, pdfb);
                                path.contrib += (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;
                            }
                        }
                        else {
                            auto cosLight = dot(nmlLight, -dirToLight);

                            if (cosShadow >= 0 && cosLight >= 0) {
                                auto dist2 = squared_length(sampleres.dir);
                                auto G = cosShadow * cosLight / dist2;

                                if (pdfb > real(0) && pdfLight > real(0)) {
                                    // Convert pdf from steradian to area.
                                    // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
                                    // p31 - p35
                                    pdfb = pdfb * cosLight / dist2;
                                    auto misW = aten::computeBalanceHeuristic(pdfLight * lightSelectPdf, pdfb);
                                    path.contrib += (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;
                                }
                            }
                        }
                    }
                }
            }

#if 1
            if (m_virtualLight)
            {
                auto sampleres = m_virtualLight->sample(ctxt, path.rec.p, nullptr);

                const vec3& posLight = sampleres.pos;
                const vec3& nmlLight = sampleres.nml;
                real pdfLight = sampleres.pdf;

                auto lightobj = sampleres.obj;

                vec3 dirToLight = normalize(sampleres.dir);
                aten::ray shadowRay(path.rec.p, dirToLight, orienting_normal);

                hitrecord tmpRec;

                if (scene->hitLight(ctxt, m_virtualLight, posLight, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
                    auto cosShadow = dot(orienting_normal, dirToLight);
                    auto dist2 = squared_length(sampleres.dir);
                    auto dist = aten::sqrt(dist2);

                    auto bsdf = mtrl->bsdf(orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);
                    auto pdfb = mtrl->pdf(orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);

                    // Get light color.
                    auto emit = sampleres.finalColor;

                    auto c = dot(m_lightDir, -dirToLight);
                    real visible = (real)(c > real(0) ? 1 : 0);

                    auto misW = pdfLight / (pdfb + pdfLight);
                    path.contrib += visible * misW * bsdf * emit * cosShadow / pdfLight;
                }

                if (!mtrl->isGlossy()) {
                    return false;
                }
            }
#endif
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
            path.throughput *= path.accumulatedAlpha * bsdf * c / pdfb;
            path.throughput /= russianProb;
        }
        else {
            return false;
        }

        if (!isAlphaBlended) {
            // Reset alpha blend.
            path.accumulatedAlpha = real(1);
        }

        path.prevMtrl = mtrl;

        path.pdfb = pdfb;

        // Make next ray.
        path.ray = aten::ray(path.rec.p, nextDir, rayBasedNormal);

        return true;
    }

    void PathTracing::shadeMiss(
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

        path.contrib += path.throughput * misW * emit * path.accumulatedAlpha;
    }

    static uint32_t frame = 0;

    void PathTracing::onRender(
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

                        //XorShift rnd(scramble + t.milliSeconds);
                        //Halton rnd(scramble + t.milliSeconds);
                        //Sobol rnd;
                        //WangHash rnd(scramble + t.milliSeconds);
#if 1
                        CMJ rnd;
                        rnd.init(frame, i, scramble);
#else
                        // Experimental
                        BlueNoiseSampler rnd;
                        for (auto tex : m_noisetex) {
                            rnd.registerNoiseTexture(tex);
                        }
                        rnd.init(x, y, frame, m_maxDepth, 1);
#endif

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
