#include "raytracing.h"
#include "misc/omputil.h"
#include "renderer/nonphotoreal.h"

namespace aten
{
    vec3 RayTracing::radiance(
        const context& ctxt,
        const ray& inRay,
        scene* scene)
    {
        uint32_t depth = 0;

        aten::ray ray = inRay;

        vec3 contribution = vec3(0, 0, 0);
        vec3 throughput = vec3(1, 1, 1);

        while (depth < m_maxDepth) {
            hitrecord rec;
            Intersection isect;

            if (scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, rec, isect)) {
                auto mtrl = ctxt.getMaterial(rec.mtrlid);

                if (mtrl->isEmissive()) {
                    auto emit = static_cast<aten::vec3>(mtrl->color());
                    contribution += throughput * emit;
                    return contribution;
                }

                auto isBackfacing = dot(rec.normal, -ray.dir) < real(0.0);

                // 交差位置の法線.
                // 物体からのレイの入出を考慮.
                vec3 orienting_normal = rec.normal;
                if (!mtrl->isTranslucent() && isBackfacing) {
                    orienting_normal = -orienting_normal;
                }

                // Get normal to add ray offset.
                // In refraction material case, new ray direction might be computed with inverted normal.
                // For example, when a ray go into the refraction surface, inverted normal is used to compute new ray direction.
                auto rayBasedNormal = (!isBackfacing && mtrl->isTranslucent())
                    ? -orienting_normal
                    : orienting_normal;

                if (mtrl->isSingular() || mtrl->isTranslucent()) {
                    auto sampling = mtrl->sample(ray, orienting_normal, rec.normal, nullptr, real(0), rec.u, rec.v);

                    auto nextDir = normalize(sampling.dir);
                    auto bsdf = sampling.bsdf;

                    throughput *= bsdf;

                    // Make next ray.
                    ray = aten::ray(rec.p, nextDir, rayBasedNormal);
                }
                else if (mtrl->isNPR()) {
                    // Non-Photo-Real.
                    contribution = shadeNPR(ctxt, mtrl.get(), rec.p, orienting_normal, rec.u, rec.v, scene, nullptr);
                    return contribution;
                }
                else {
                    auto lightNum = scene->lightNum();

                    for (int i = 0; i < lightNum; i++) {
                        auto light = scene->getLight(i);

                        if (light->isIBL()) {
                            continue;
                        }

                        // TODO
                        // In area light case, we don't specify sampler.
                        // So, in area light sampler, center of light is choosed
                        // as the position which ray try to reach to the light.
                        // But, if the light is square angle and it is composed of 2 triangles,
                        // center of the light might be edge of border of 2 triangles.
                        // Therefore, the ray might not hit any tirangles of the light.
                        // エリアライトの場合に、samplerがnullptrのときはライトのAABBの中心が
                        // レイの向かう先として選択される。
                        // その場合に、もしエリアライトが正方形で二つの三角形から構成されていたときに
                        // 中心位置は三角形の境目に位置していることになり、誤差等によりヒットしない可能性がある.
                        auto sampleres = light->sample(ctxt, rec.p, nullptr);

                        if (!light->isSingular() && !sampleres.obj) {
                            continue;
                        }

                        vec3 dirToLight = sampleres.dir;
                        auto len = length(dirToLight);

                        dirToLight = normalize(dirToLight);

                        if (dot(orienting_normal, dirToLight) < real(0)) {
                            continue;
                        }

                        auto albedo = static_cast<aten::vec3>(mtrl->color());

                        aten::ray shadowRay(rec.p, dirToLight, orienting_normal);

                        hitrecord tmpRec;

                        if (scene->hitLight(ctxt, light.get(), sampleres.pos, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
                            auto lightColor = sampleres.finalColor;

                            if (light->isInfinite()) {
                                len = real(1);
                            }

                            const auto c0 = std::max(real(0.0), dot(orienting_normal, dirToLight));
                            real c1 = real(1);

                            if (!light->isSingular()) {
                                c1 = std::max(real(0.0), dot(sampleres.nml, -dirToLight));
                            }

                            auto G = c0 * c1 / (len * len);

                            contribution += throughput * (albedo * lightColor) * G;
                        }
                    }

                    break;
                }
            }
            else {
                auto ibl = scene->getIBL();

                if (ibl) {
                    auto bg = ibl->getEnvMap()->sample(ray);
                    contribution += throughput * bg;
                }
                else {
                    auto bg = sampleBG(ray);
                    contribution += throughput * bg;
                }

                return contribution;
            }

            depth++;
        }

        return contribution;
    }

    void RayTracing::onRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
    {
        int width = dst.width;
        int height = dst.height;

        m_maxDepth = dst.maxDepth;

        uint32_t sample = 1;

#ifdef ENABLE_OMP
#pragma omp parallel
#endif
        {
#ifdef ENABLE_OMP
#pragma omp for
#endif
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    //if (x == 419 && y == 107) {
                    //if (x == 408 && y == 112) {
                    if (x == 259 && y == 0) {
                        int xxx = 0;
                    }
                    int pos = y * width + x;

                    real u = (real(x) + real(0.5)) / real(width - 1);
                    real v = (real(y) + real(0.5)) / real(height - 1);

                    auto camsample = camera->sample(u, v, nullptr);

                    auto col = radiance(ctxt, camsample.r, scene);

                    dst.buffer->put(x, y, vec4(col, 1));
                }
            }
        }
    }
}
