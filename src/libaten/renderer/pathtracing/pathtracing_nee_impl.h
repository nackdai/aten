#pragma once

#include <optional>

#include "defs.h"
#include "light/light.h"
#include "material/material.h"
#include "material/material_impl.h"
#include "renderer/pathtracing/pt_params.h"

// TODO:
// Unify 2 almost same APIs to compute NEE.

namespace AT_NAME
{
    namespace _detail {
        inline AT_DEVICE_API float ComputeBalanceHeuristic(float f, float g)
        {
            return f / (f + g);
        }
    }

    inline AT_DEVICE_API std::optional<aten::vec3> ComputeRadianceNEE(
        const AT_NAME::context& ctxt,
        const aten::vec3& throughput,
        const aten::vec3& wi,
        const aten::vec3& surface_nml,
        const aten::MaterialParameter& surface_mtrl,
        const float pre_sampled_random,
        float hit_u, float hit_v,
        const float light_select_prob,
        const aten::LightSampleResult& light_sample)
    {
        auto cosShadow = dot(surface_nml, light_sample.dir);

        float path_pdf{ AT_NAME::material::samplePDF(ctxt, &surface_mtrl, surface_nml, wi, light_sample.dir, hit_u, hit_v) };
        auto mtrl_eval_result{ AT_NAME::material::sampleBSDF(ctxt, throughput, &surface_mtrl, surface_nml, wi, light_sample.dir, hit_u, hit_v, pre_sampled_random) };

        if (mtrl_eval_result.pdf > 0) {
            path_pdf = mtrl_eval_result.pdf;
        }
        const auto& bsdf = mtrl_eval_result.bsdf;

        // Get light color.
        const auto& emit{ light_sample.light_color };

        auto cosLight = dot(light_sample.nml, -light_sample.dir);

        auto dist2 = aten::sqr(light_sample.dist_to_light);
        dist2 = (light_sample.attrib.isInfinite || light_sample.attrib.is_singular) ? float{ 1 } : dist2;

        if (cosShadow >= 0 && cosLight >= 0
            && dist2 > 0
            && path_pdf > float(0) && light_sample.pdf > float(0))
        {
            // NOTE:
            // Regarding punctual light, nothing to sample.
            // It means there is nothing to convert pdf.
            // TODO: IBL...
            if (!light_sample.attrib.isInfinite) {
                // Convert path PDF to NEE PDF.
                // i.e. Convert solid angle PDF to area PDF.
                // NEE samples the point on the light, and the point sampling means the PDF is area.
                // Sampling the direction towards the light point means the PDF is solid angle.
                // To align the path PDF to NEE PDF, converting solid angl PDF to area PDF is necessary.
                path_pdf = path_pdf * cosLight / dist2;
            }

            auto misW = light_sample.attrib.is_singular
                ? 1.0f
                : _detail::ComputeBalanceHeuristic(light_sample.pdf * light_select_prob, path_pdf);

            const auto G = light_sample.attrib.isInfinite
                ? cosShadow * cosLight
                : cosShadow * cosLight / dist2;

            // NOTE:
            // 3point rendering equation.
            // Compute as area PDF.
            const auto contrib = (misW * bsdf * emit * G / light_sample.pdf) / light_select_prob;
            return contrib;
        }

        return std::nullopt;
    }

    inline AT_DEVICE_API std::optional<aten::vec3> ComputeRadianceNEEWithAlphaBlending(
        const int32_t idx,
        const AT_NAME::context& ctxt,
        const AT_NAME::Path& paths,
        const aten::vec3& wi,
        const aten::vec3& surface_nml,
        const aten::MaterialParameter& surface_mtrl,
        const float pre_sampled_random,
        float hit_u, float hit_v,
        const float light_select_prob,
        const aten::LightSampleResult& light_sample)
    {
        auto cosShadow = dot(surface_nml, light_sample.dir);

        float path_pdf{ AT_NAME::material::samplePDF(ctxt, &surface_mtrl, surface_nml, wi, light_sample.dir, hit_u, hit_v) };
        auto mtrl_eval_result{ AT_NAME::material::sampleBSDF(
            ctxt,
            paths.throughput[idx].throughput,
            &surface_mtrl, surface_nml,
            wi, light_sample.dir,
            hit_u, hit_v, pre_sampled_random) };

        if (mtrl_eval_result.pdf > 0) {
            path_pdf = mtrl_eval_result.pdf;
        }
        auto bsdf = mtrl_eval_result.bsdf;

        // Get light color.
        const auto& emit{ light_sample.light_color };

        auto cosLight = dot(light_sample.nml, -light_sample.dir);

        auto dist2 = aten::sqr(light_sample.dist_to_light);
        dist2 = (light_sample.attrib.isInfinite || light_sample.attrib.is_singular) ? float{ 1 } : dist2;

        if (cosShadow >= 0 && cosLight >= 0
            && dist2 > 0
            && path_pdf > float(0) && light_sample.pdf > float(0))
        {
            // NOTE:
            // Regarding punctual light, nothing to sample.
            // It means there is nothing to convert pdf.
            // TODO: IBL...
            if (!light_sample.attrib.isInfinite) {
                // Convert path PDF to NEE PDF.
                // i.e. Convert solid angle PDF to area PDF.
                // NEE samples the point on the light, and the point sampling means the PDF is area.
                // Sampling the direction towards the light point means the PDF is solid angle.
                // To align the path PDF to NEE PDF, converting solid angl PDF to area PDF is necessary.
                path_pdf = path_pdf * cosLight / dist2;
            }

            auto misW = light_sample.attrib.is_singular
                ? 1.0f
                : _detail::ComputeBalanceHeuristic(light_sample.pdf * light_select_prob, path_pdf);

            const auto G = light_sample.attrib.isInfinite
                ? cosShadow * cosLight
                : cosShadow * cosLight / dist2;

            if (!paths.attrib[idx].has_applied_alpha_blending_in_nee) {
                bsdf = paths.throughput[idx].transmission * bsdf + paths.throughput[idx].alpha_blend_radiance_on_the_way;
            }

            // NOTE:
            // 3point rendering equation.
            // Compute as area PDF.
            const auto contrib = (misW * bsdf * emit * G / light_sample.pdf) / light_select_prob;
            return contrib;
        }

        return std::nullopt;
    }
}
