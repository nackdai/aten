#include "renderer/directlight.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/wanghash.h"

namespace aten
{
    static inline bool isInvalidColor(const vec3& v)
    {
        bool b = isInvalid(v);
        if (!b) {
            if (v.x < 0 || v.y < 0 || v.z < 0) {
                b = true;
            }
        }

        return b;
    }

    DirectLightRenderer::Path DirectLightRenderer::radiance(
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

            if (!willContinue) {
                break;
            }

            depth++;
        }

        return std::move(path);
    }

    bool DirectLightRenderer::shade(
        const context& ctxt,
        sampler* sampler,
        scene* scene,
        camera* cam,
        CameraSampleResult& camsample,
        int depth,
        Path& path)
    {
        uint32_t rrDepth = m_rrDepth;

        // 交差位置の法線.
        // 物体からのレイの入出を考慮.
        vec3 orienting_normal = dot(path.rec.normal, path.ray.dir) < 0.0 ? path.rec.normal : -path.rec.normal;

        auto mtrl = ctxt.getMaterial(path.rec.mtrlid);

        // Apply normal map.
        mtrl->applyNormalMap(orienting_normal, orienting_normal, path.rec.u, path.rec.v);

        // Implicit conection to light.
        if (mtrl->isEmissive()) {
            if (depth == 0) {
                // Ray hits the light directly.
                path.contrib = mtrl->color();
                path.isTerminate = true;
                return false;
            }
            else if (path.prevMtrl && path.prevMtrl->isSingular()) {
                auto emit = mtrl->color();
                path.contrib += path.throughput * emit;
                return false;
            }
            else {
                auto cosLight = dot(orienting_normal, -path.ray.dir);
                auto dist2 = squared_length(path.rec.p - path.ray.org);

                if (cosLight >= 0) {
                    auto pdfLight = 1 / path.rec.area;

                    // Convert pdf area to sradian.
                    // http://www.slideshare.net/h013/edubpt-v100
                    // p31 - p35
                    pdfLight = pdfLight * dist2 / cosLight;

                    auto misW = path.pdfb / (pdfLight + path.pdfb);

                    auto emit = mtrl->color();

                    path.contrib += path.throughput * misW * emit;

                    // When ray hit the light, tracing will finish.
                    return false;
                }
            }
        }

        // Explicit conection to light.
        if (mtrl->isSingular()) {
            auto sampling = mtrl->sample(path.ray, orienting_normal, path.rec.normal, sampler, path.rec.u, path.rec.v);

            auto nextDir = normalize(sampling.dir);
            auto pdfb = sampling.pdf;
            auto bsdf = sampling.bsdf;

            if (pdfb > 0) {
                path.throughput *= bsdf / pdfb;
            }
            else {
                return false;
            }

            path.prevMtrl = mtrl;

            path.pdfb = pdfb;

            // Make next ray.
            path.ray = aten::ray(path.rec.p, nextDir);

            return true;
        }
        else
        {
            int lightNum = scene->lightNum();

            for (int i = 0; i < lightNum; i++) {
                auto light = scene->getLight(i);

                auto sampleres = light->sample(ctxt, path.rec.p, sampler);

                const vec3& posLight = sampleres.pos;
                const vec3& nmlLight = sampleres.nml;
                real pdfLight = sampleres.pdf;

                auto lightobj = sampleres.obj;

                vec3 dirToLight = normalize(sampleres.dir);
                aten::ray shadowRay(path.rec.p, dirToLight);

                auto bsdf = mtrl->bsdf(orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);
                auto pdfb = mtrl->pdf(orienting_normal, path.ray.dir, dirToLight, path.rec.u, path.rec.v);

                hitrecord tmpRec;

                if (scene->hitLight(ctxt, light, posLight, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
                    // Shadow ray hits the light.
                    auto cosShadow = dot(orienting_normal, dirToLight);

                    // Get light color.
                    auto emit = sampleres.finalColor;

                    if (light->isSingular() || light->isInfinite()) {
                        if (pdfLight > real(0) && cosShadow >= 0) {
                            auto misW = pdfLight / (pdfb + pdfLight);
                            path.contrib += (misW * bsdf * path.throughput * emit * cosShadow / pdfLight);
                        }
                    }
                    else {
                        auto cosLight = dot(nmlLight, -dirToLight);

                        if (cosShadow >= 0 && cosLight >= 0) {
                            auto dist2 = squared_length(sampleres.dir);
                            auto G = cosShadow * cosLight / dist2;

                            if (pdfb > real(0) && pdfLight > real(0)) {
                                pdfb = pdfb * cosLight / dist2;

                                auto misW = pdfLight / (pdfb + pdfLight);

                                path.contrib += (misW * (bsdf * path.throughput * emit * G) / pdfLight);
                            }
                        }
                    }

                    if (!light->isSingular())
                    {
                        auto sampling = mtrl->sample(path.ray, orienting_normal, path.rec.normal, sampler, path.rec.u, path.rec.v);

                        auto nextDir = normalize(sampling.dir);
                        pdfb = sampling.pdf;
                        bsdf = sampling.bsdf;

                        auto c = dot(orienting_normal, nextDir);
                        vec3 throughput(1, 1, 1);

                        if (pdfb > 0 && c > 0) {
                            throughput *= bsdf * c / pdfb;
                        }

                        ray nextRay = aten::ray(path.rec.p, nextDir);

                        aten::Intersection tmpIsect;

                        if (scene->hit(ctxt, nextRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec, tmpIsect)) {
                            auto tmpmtrl = ctxt.getMaterial(tmpRec.mtrlid);

                            // Implicit conection to light.
                            if (tmpmtrl->isEmissive()) {
                                auto cosLight = dot(orienting_normal, -nextRay.dir);
                                auto dist2 = squared_length(tmpRec.p - nextRay.org);

                                if (cosLight >= 0) {
                                    auto pdfLight = 1 / tmpRec.area;

                                    pdfLight = pdfLight * dist2 / cosLight;

                                    auto misW = pdfb / (pdfLight + pdfb);

                                    auto emit = mtrl->color();

                                    path.contrib += throughput * misW * emit;
                                }
                            }
                        }
                        else {
                            auto ibl = scene->getIBL();
                            if (ibl) {
                                auto pdfLight = ibl->samplePdf(nextRay);
                                auto misW = pdfb / (pdfLight + pdfb);
                                auto emit = ibl->getEnvMap()->sample(nextRay);
                                path.contrib += throughput * misW * emit;
                            }
                        }
                    }
                }
            }

            return false;
        }
    }

    void DirectLightRenderer::shadeMiss(
        scene* scene,
        int depth,
        Path& path)
    {
        auto ibl = scene->getIBL();
        if (ibl) {
            if (depth == 0) {
                auto bg = ibl->getEnvMap()->sample(path.ray);
                path.contrib += path.throughput * bg;
                path.isTerminate = true;
            }
            else {
                auto pdfLight = ibl->samplePdf(path.ray);
                auto misW = path.pdfb / (pdfLight + path.pdfb);
                auto emit = ibl->getEnvMap()->sample(path.ray);
                path.contrib += path.throughput * misW * emit;
            }
        }
        else {
            auto bg = sampleBG(path.ray);
            path.contrib += path.throughput * bg;
        }
    }

    void DirectLightRenderer::onRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
    {
        int width = dst.width;
        int height = dst.height;
        uint32_t samples = dst.sample;

        m_maxDepth = dst.maxDepth;
        m_rrDepth = dst.russianRouletteDepth;
        m_startDepth = dst.startDepth;

        if (m_rrDepth > m_maxDepth) {
            m_rrDepth = m_maxDepth - 1;
        }


#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp parallel
#endif
        {
            auto idx = OMPUtil::getThreadIdx();

            auto time = timer::getSystemTime();

#if defined(ENABLE_OMP) && !defined(RELEASE_DEBUG)
#pragma omp for
#endif
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pos = y * width + x;

                    vec3 col = vec3(0);
                    vec3 col2 = vec3(0);
                    uint32_t cnt = 0;

                    for (uint32_t i = 0; i < samples; i++) {
                        auto scramble = aten::getRandom(pos) * 0x1fe3434f;

                        //XorShift rnd(scramble + time.milliSeconds);
                        //Halton rnd(scramble + time.milliSeconds);
                        Sobol rnd(scramble + time.milliSeconds);
                        //WangHash rnd(scramble + time.milliSeconds);

                        real u = real(x + rnd.nextSample()) / real(width);
                        real v = real(y + rnd.nextSample()) / real(height);

                        auto camsample = camera->sample(u, v, &rnd);

                        auto ray = camsample.r;

                        auto path = radiance(
                            ctxt,
                            &rnd,
                            m_maxDepth,
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
