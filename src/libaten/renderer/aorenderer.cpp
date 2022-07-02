#include "renderer/aorenderer.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/nonphotoreal.h"
#include "sampler/xorshift.h"
#include "sampler/halton.h"
#include "sampler/sobolproxy.h"
#include "sampler/wanghash.h"
#include "sampler/cmj.h"
#include "sampler/bluenoiseSampler.h"

#include "material/lambert.h"

//#define Deterministic_Path_Termination

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (6)
#define BREAK_Y    (0)
#pragma optimize( "", off)
#endif

namespace aten
{
    AORenderer::Path AORenderer::radiance(
        const context& ctxt,
        sampler* sampler,
        const ray& inRay,
        scene* scene)
    {
        Path path;
        path.ray = inRay;

        path.rec = hitrecord();

        Intersection isect;

        if (scene->hit(ctxt, path.ray, AT_MATH_EPSILON, AT_MATH_INF, path.rec, isect)) {
            shade(ctxt, sampler, scene, path);
        }
        else {
            shadeMiss(path);
        }

        return path;
    }

    bool AORenderer::shade(
        const context& ctxt,
        sampler* sampler,
        scene* scene,
        Path& path)
    {
        // 交差位置の法線.
        // 物体からのレイの入出を考慮.
        vec3 orienting_normal = path.rec.normal;

        auto mtrl = ctxt.getMaterial(path.rec.mtrlid);

        // Apply normal map.
        (void)mtrl->applyNormalMap(orienting_normal, orienting_normal, path.rec.u, path.rec.v, path.ray.dir, nullptr);

        aten::vec3 ao_color(0.0f);

        aten::hitrecord rec;
        aten::Intersection isect;

        for (uint32_t i = 0; i < m_numAORays; i++) {
            auto nextDir = aten::lambert::sampleDirection(orienting_normal, sampler);
            auto pdfb = aten::lambert::pdf(orienting_normal, nextDir);

            real c = dot(orienting_normal, nextDir);

            auto ao_ray = aten::ray(path.rec.p, nextDir, orienting_normal);

            auto isHit = scene->hit(ctxt, ao_ray, AT_MATH_EPSILON, m_AORadius, rec, isect);

            if (isHit) {
                if (c > 0.0f) {
                    ao_color += aten::vec3(isect.t / m_AORadius * c / pdfb);
                }
            }
            else {
                ao_color += aten::vec3(real(1));
            }
        }

        path.contrib = ao_color / (real)m_numAORays;

        return true;
    }

    void AORenderer::shadeMiss(Path& path)
    {
        path.contrib = aten::vec3(real(1));
    }

    static uint32_t frame = 0;

    void AORenderer::onRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
    {
        frame++;

        int width = dst.width;
        int height = dst.height;

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

#ifdef RELEASE_DEBUG
                    if (x == BREAK_X && y == BREAK_Y) {
                        DEBUG_BREAK();
                    }
#endif

                    auto scramble = aten::getRandom(pos) * 0x1fe3434f;

                    CMJ rnd;
                    rnd.init(frame, 0, scramble);

                    real u = real(x + rnd.nextSample()) / real(width);
                    real v = real(y + rnd.nextSample()) / real(height);

                    auto camsample = camera->sample(u, v, &rnd);

                    auto ray = camsample.r;

                    auto path = radiance(
                        ctxt,
                        &rnd,
                        ray,
                        scene);

                    if (isInvalidColor(path.contrib)) {
                        AT_PRINTF("Invalid(%d/%d)\n", x, y);
                        continue;
                    }

                    dst.buffer->put(x, y, vec4(path.contrib, 1));
                }
            }
        }
    }
}
