#pragma once

#include "kernel/idatendefs.cuh"
#include "kernel/pt_params.h"
#include "kernel/context.cuh"
#include "kernel/accelerator.cuh"

namespace kernel {
    inline __device__ bool hitShadowRay(
        bool enableLod,
        const idaten::Context& ctxt,
        const idaten::ShadowRay& shadowRay)
    {
        auto targetLightId = shadowRay.targetLightId;
        auto distToLight = shadowRay.distToLight;

        auto light = ctxt.lights[targetLightId];
        auto lightobj = (light.objid >= 0 ? &ctxt.shapes[light.objid] : nullptr);

        real distHitObjToRayOrg = AT_MATH_INF;

        // Ray aim to the area light.
        // So, if ray doesn't hit anything in intersectCloserBVH, ray hit the area light.
        const aten::GeomParameter* hitobj = lightobj;

        aten::Intersection isectTmp;

        bool isHit = false;

        aten::ray r(shadowRay.rayorg, shadowRay.raydir);

        isHit = intersectCloser(&ctxt, r, &isectTmp, distToLight - AT_MATH_EPSILON, enableLod);

        if (isHit) {
            hitobj = &ctxt.shapes[isectTmp.objid];
        }

        isHit = AT_NAME::scene::hitLight(
            isHit,
            light.attrib,
            lightobj,
            distToLight,
            distHitObjToRayOrg,
            isectTmp.t,
            hitobj);

        return isHit;
    }

    inline __device__ void hitImplicitLight(
        bool is_back_facing,
        int bounce,
        idaten::PathContrib& path_contrib,
        idaten::PathAttribute& path_attrib,
        idaten::PathThroughput& path_throughput,
        const aten::ray& ray,
        const aten::vec3& hit_pos,
        const aten::vec3& hit_nml,
        float hit_area,
        const aten::MaterialParameter& mtrl)
    {
        if (!is_back_facing) {
            float weight = 1.0f;

            if (bounce > 0 && !path_attrib.isSingular) {
                auto cosLight = dot(hit_nml, -ray.dir);
                auto dist2 = aten::squared_length(hit_pos - ray.org);

                if (cosLight >= 0) {
                    auto pdfLight = 1 / hit_area;

                    // Convert pdf area to sradian.
                    // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
                    // p31 - p35
                    pdfLight = pdfLight * dist2 / cosLight;

                    weight = path_throughput.pdfb / (pdfLight + path_throughput.pdfb);
                }
            }

            auto contrib = path_throughput.throughput * weight * static_cast<aten::vec3>(mtrl.baseColor);
            path_contrib.contrib += make_float3(contrib.x, contrib.y, contrib.z);
        }

        // When ray hit the light, tracing will finish.
        path_attrib.isTerminate = true;
    }

    inline __device__ float executeRussianProbability(
        int bounce,
        int rrBounce,
        idaten::PathAttribute& path_attrib,
        idaten::PathThroughput& path_throughput,
        aten::sampler& sampler)
    {
        float russianProb = 1.0f;

        if (bounce > rrBounce) {
            auto t = normalize(path_throughput.throughput);
            auto p = aten::cmpMax(t.r, aten::cmpMax(t.g, t.b));

            russianProb = sampler.nextSample();

            if (russianProb >= p) {
                //shPaths[threadIdx.x].contrib = aten::vec3(0);
                path_attrib.isTerminate = true;
            }
            else {
                russianProb = max(p, 0.01f);
            }
        }

        return russianProb;
    }

    inline __device__ bool fillShadowRay(
        idaten::ShadowRay& shadow_ray,
        idaten::Context& ctxt,
        int bounce,
        aten::sampler& sampler,
        const idaten::PathThroughput& throughtput,
        int target_light_idx,
        const aten::LightParameter& light,
        const aten::MaterialParameter& mtrl,
        const aten::ray& ray,
        const aten::vec3& hit_pos,
        const aten::vec3& hit_nml,
        real hit_u, real hit_v,
        const aten::vec4& external_albedo,
        real lightSelectPdf)
    {
        bool isShadowRayActive = false;

        aten::LightSampleResult sampleres;
        sampleLight(&sampleres, &ctxt, &light, hit_pos, hit_nml, &sampler, bounce);

        const auto& posLight = sampleres.pos;
        const auto& nmlLight = sampleres.nml;
        real pdfLight = sampleres.pdf;

        auto dirToLight = normalize(sampleres.dir);
        auto distToLight = length(posLight - hit_pos);

        auto shadowRayOrg = hit_pos + AT_MATH_EPSILON * hit_nml;
        auto tmp = hit_pos + dirToLight - shadowRayOrg;
        auto shadowRayDir = normalize(tmp);

        shadow_ray.rayorg = shadowRayOrg;
        shadow_ray.raydir = shadowRayDir;
        shadow_ray.targetLightId = target_light_idx;
        shadow_ray.distToLight = distToLight;
        shadow_ray.lightcontrib = aten::vec3(0);
        {
            auto cosShadow = dot(hit_nml, dirToLight);

            real pdfb = samplePDF(&ctxt, &mtrl, hit_nml, ray.dir, dirToLight, hit_u, hit_v);
            auto bsdf = sampleBSDF(&ctxt, &mtrl, hit_nml, ray.dir, dirToLight, hit_u, hit_v, external_albedo);

            bsdf *= throughtput.throughput;

            // Get light color.
            auto emit = sampleres.finalColor;

            if (light.attrib.isInfinite || light.attrib.isSingular) {
                if (pdfLight > real(0) && cosShadow >= 0) {
                    auto misW = light.attrib.isSingular
                        ? 1.0f
                        : AT_NAME::computeBalanceHeuristic(pdfLight * lightSelectPdf, pdfb);
                    shadow_ray.lightcontrib =
                        (misW * bsdf * emit * cosShadow / pdfLight) / lightSelectPdf;

                    isShadowRayActive = true;
                }
            }
            else {
                auto cosLight = dot(nmlLight, -dirToLight);

                if (cosShadow >= 0 && cosLight >= 0) {
                    auto dist2 = aten::squared_length(sampleres.dir);
                    auto G = cosShadow * cosLight / dist2;

                    if (pdfb > real(0) && pdfLight > real(0)) {
                        // Convert pdf from steradian to area.
                        // http://kagamin.net/hole/edubpt/edubpt_v100.pdf
                        // p31 - p35
                        pdfb = pdfb * cosLight / dist2;

                        auto misW = AT_NAME::computeBalanceHeuristic(pdfLight * lightSelectPdf, pdfb);

                        shadow_ray.lightcontrib =
                            (misW * (bsdf * emit * G) / pdfLight) / lightSelectPdf;

                        isShadowRayActive = true;
                    }
                }
            }
        }

        return isShadowRayActive;
    }
}
