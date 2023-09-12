#include "renderer/ao/aorenderer.h"

#include "material/lambert.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/ao/aorenderer_impl.h"
#include "renderer/pathtracing/pathtracing_impl.h"

//#define RELEASE_DEBUG

#ifdef RELEASE_DEBUG
#define BREAK_X    (6)
#define BREAK_Y    (0)
#pragma optimize( "", off)
#endif

namespace aten
{
    void AORenderer::radiance(
        int32_t idx,
        uint32_t rnd,
        const context& ctxt,
        const ray& inRay,
        scene* scene)
    {
        Intersection isect;

        uint32_t depth = 0;

        while (depth < max_depth_) {
            if (scene->hit(ctxt, inRay, AT_MATH_EPSILON, AT_MATH_INF, isect)) {
                path_host_.paths.attrib[idx].isHit = true;

                AT_NAME::ShandeAO(
                    idx,
                    get_frame_count(), rnd,
                    m_numAORays, m_AORadius,
                    path_host_.paths, ctxt, inRay, isect, scene);
            }
            else {
                bool is_first_bounce = (depth == 0);
                AT_NAME::ShadeMissAO(
                    idx,
                    is_first_bounce,
                    path_host_.paths);
                return;
            }

            depth++;
        }
    }

    void AORenderer::onRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
    {
        int32_t width = dst.width;
        int32_t height = dst.height;
        uint32_t samples = dst.sample;

        max_depth_ = dst.maxDepth;

        path_host_.init(width, height);

        for (auto& attrib : path_host_.attrib) {
            attrib.isKill = false;
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
            for (int32_t y = 0; y < height; y++) {
                for (int32_t x = 0; x < width; x++) {
                    vec3 col = vec3(0);
                    uint32_t cnt = 0;

#ifdef RELEASE_DEBUG
                    if (x == BREAK_X && y == BREAK_Y) {
                        DEBUG_BREAK();
                    }
#endif

                    if (path_host_.paths.attrib[idx].isKill) {
                        continue;
                    }

                    int32_t idx = y * width + x;

                    for (uint32_t i = 0; i < samples; i++) {
                        const auto rnd = aten::getRandom(idx);
                        const auto& camsample = camera->param();

                        aten::ray ray;

                        GeneratePath(
                            ray,
                            idx,
                            x, y,
                            i, get_frame_count(),
                            path_host_.paths,
                            camsample,
                            rnd);

                        path_host_.paths.contrib[idx].contrib = aten::vec3(0);

                        radiance(
                            idx, rnd,
                            ctxt, ray, scene);

                        if (isInvalidColor(path_host_.paths.contrib[idx].contrib)) {
                            AT_PRINTF("Invalid(%d/%d)\n", x, y);
                            continue;
                        }

                        auto c = path_host_.paths.contrib[idx].contrib;

                        col += c;
                        cnt++;

                        if (path_host_.paths.attrib[idx].isTerminate) {
                            break;
                        }
                    }

                    col /= (real)cnt;
                    dst.buffer->put(x, y, vec4(col, 1));
                }
            }
        }
    }
}
