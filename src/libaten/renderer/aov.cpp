#include "renderer/aov.h"
#include "sampler/xorshift.h"

namespace aten
{
    AOVRenderer::Path AOVRenderer::radiance(
        const context& ctxt,
        const ray& inRay,
        scene* scene,
        sampler* sampler)
    {
        ray ray = inRay;

        const auto maxDepth = m_maxDepth;
        uint32_t depth = 0;

        vec3 throughput = vec3(1);

        Path path;

        while (depth < maxDepth) {
            hitrecord rec;
            Intersection isect;

            if (scene->hit(ctxt, ray, AT_MATH_EPSILON, AT_MATH_INF, false, rec, isect)) {
                // 交差位置の法線.
                // 物体からのレイの入出を考慮.
                vec3 orienting_normal = dot(rec.normal, ray.dir) < 0.0 ? rec.normal : -rec.normal;

                auto mtrl = ctxt.getMaterial(rec.mtrlid);

                // Apply normal map.
                mtrl->applyNormalMap(orienting_normal, orienting_normal, rec.u, rec.v);

                if (depth == 0) {
                    path.normal = orienting_normal;
                    path.depth = isect.t;

                    path.shapeid = isect.objid;
                    path.mtrlid = mtrl->id();
                }

                if (mtrl->isEmissive()) {
                    path.albedo = mtrl->color();
                    path.albedo *= throughput;
                    break;
                }
                else if (mtrl->isSingular()) {
                    auto sample = mtrl->sample(ray, orienting_normal, rec.normal, nullptr, rec.u, rec.v);

                    const auto& nextDir = sample.dir;
                    throughput *= sample.bsdf;

                    // Make next ray.
                    ray = aten::ray(rec.p, nextDir);
                }
                else {
                    {
                        real lightSelectPdf = 1;
                        LightSampleResult sampleres;

                        auto light = scene->sampleLight(
                            ctxt,
                            rec.p,
                            orienting_normal,
                            sampler,
                            lightSelectPdf, sampleres);

                        if(light) {
                            vec3 posLight = sampleres.pos;
                            vec3 nmlLight = sampleres.nml;
                            real pdfLight = sampleres.pdf;

                            auto lightobj = sampleres.obj;

                            vec3 dirToLight = normalize(sampleres.dir);
                            aten::ray shadowRay(rec.p, dirToLight);

                            hitrecord tmpRec;

                            if (scene->hitLight(ctxt, light, posLight, shadowRay, AT_MATH_EPSILON, AT_MATH_INF, tmpRec)) {
                                path.visibility = 1;
                            }
                        }
                    }

                    path.albedo = mtrl->color();
                    path.albedo *= mtrl->sampleAlbedoMap(rec.u, rec.v);
                    path.albedo *= throughput;
                    break;
                }
            }
            else {
                auto ibl = scene->getIBL();
                if (ibl) {
                    auto bg = ibl->getEnvMap()->sample(ray);
                    path.albedo = throughput * bg;
                }
                else {
                    auto bg = sampleBG(ray);
                    path.albedo = throughput * bg;
                }

                if (depth == 0) {
                    // Far far away...
                    path.depth = AT_MATH_INF;
                }

                path.visibility = 1;

                break;
            }

            depth++;
        }

        return std::move(path);
    }

    void AOVRenderer::onRender(
        const context& ctxt,
        Destination& dst,
        scene* scene,
        camera* camera)
    {
        int width = dst.width;
        int height = dst.height;

        m_maxDepth = dst.maxDepth;

        real depthNorm = 1 / dst.geominfo.depthMax;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pos = y * width + x;

                XorShift rnd((y * height * 4 + x * 4) + 1);

                real u = real(x + 0.5) / real(width);
                real v = real(y + 0.5) / real(height);

                auto camsample = camera->sample(u, v, &rnd);

                auto path = radiance(ctxt, camsample.r, scene, &rnd);

                if (dst.geominfo.nml_depth) {
                    if (dst.geominfo.needNormalize) {
                        // [-1, 1] -> [0, 1]
                        auto normal = (path.normal + real(1)) * real(0.5);

                        // [-∞, ∞] -> [-d, d]
                        real depth = std::min(aten::abs(path.depth), dst.geominfo.depthMax);
                        depth *= path.depth < 0 ? -1 : 1;

                        if (dst.geominfo.needNormalize) {
                            // [-d, d] -> [-1, 1]
                            depth *= depthNorm;

                            // [-1, 1] -> [0, 1]
                            depth = (depth + 1) * real(0.5);
                        }

                        dst.geominfo.nml_depth->put(x, y, vec4(normal, depth));
                    }
                    else {
                        dst.geominfo.nml_depth->put(x, y, vec4(path.normal, path.depth));
                    }
                }
                if (dst.geominfo.albedo_vis) {
                    auto albedo = path.albedo;

                    if (dst.geominfo.needNormalize) {
                        // TODO
                        albedo.x = std::min<real>(albedo.x, 1);
                        albedo.y = std::min<real>(albedo.y, 1);
                        albedo.z = std::min<real>(albedo.z, 1);
                    }

                    dst.geominfo.albedo_vis->put(x, y, vec4(albedo, path.visibility));
                }
                if (dst.geominfo.ids) {
                    dst.geominfo.ids->put(x, y, vec4(path.shapeid, path.mtrlid, 0, 0));
                }
            }
        }
    }
}
