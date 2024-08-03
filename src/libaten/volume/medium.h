#pragma once

#include "defs.h"
#include "material/material.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "misc/tuple.h"
#include "renderer/pathtracing/pt_params.h"
#include "sampler/sampler.h"
#include "volume/phase_function.h"

#ifndef NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED
namespace nanovdb {
    class FloatGrid;
}
#endif

namespace AT_NAME {
    class HomogeniousMedium {
    public:
        HomogeniousMedium() = delete;
        ~HomogeniousMedium() = delete;

        HomogeniousMedium(const HomogeniousMedium&) = delete;
        HomogeniousMedium(HomogeniousMedium&&) = delete;
        HomogeniousMedium& operator=(const HomogeniousMedium&) = delete;
        HomogeniousMedium& operator=(HomogeniousMedium&&) = delete;

        static AT_DEVICE_API aten::tuple<bool, aten::ray> Sample(
            AT_NAME::PathThroughput& throughput,
            AT_NAME::sampler& sampler,
            const aten::ray& curr_ray,
            const AT_NAME::MediumParameter& medium,
            const float distance_to_surface)
        {
            const auto sigma_a = medium.sigma_a;
            const auto sigma_s = medium.sigma_s;
            const auto sigma_t = HomogeniousMedium::sigma_t(medium);

            const auto r1 = sampler.nextSample();

            // Sample distance.
            const float s = -aten::log(aten::cmpMax(1.0F - r1, 0.0F)) / sigma_t;

            // Hit volume boundary.
            if (s >= distance_to_surface) {
                // Same direction, but next ray origin is the bound of the volume.
                aten::ray next_ray{ curr_ray };
                next_ray.org += next_ray.dir * distance_to_surface;
                return aten::make_tuple(false, next_ray);
            }

            const auto r2 = sampler.nextSample();

            const auto Pa = sigma_a / sigma_t;
            const auto Ps = 1.0F - Pa;
            const auto tr = Transmittance(sigma_t, s);
            const auto pdf = sigma_t * tr;

            aten::ray next_ray{ curr_ray };
            next_ray.org += next_ray.dir * s;

            if (r2 < Pa) {
                // Absorb internal emission.
#if 0
                const auto medium_throughput = tr * (sigma_a * medium.le / Pa) / pdf;
                throughput.throughput *= medium_throughput;
#else
                // As the optimization, just multiplying internal emission Le is enough.
                throughput.throughput *= medium.le;
#endif
            }
            else {
                // Sample next direction by phase function.
                const auto r3 = sampler.nextSample();
                const auto r4 = sampler.nextSample();
                next_ray.dir = HenyeyGreensteinPhaseFunction::SampleDirection(r3, r4, medium.phase_function_g, -curr_ray.dir);
                next_ray.dir = normalize(next_ray.dir);
#if 0
                const auto medium_throughput = tr * (sigma_s / Ps) / pdf;
                throughput.throughput *= medium_throughput;
#else
                // As the optimization, nothing to multiply to throughput.
#endif
            }

            return aten::make_tuple(true, next_ray);
        }

        static AT_DEVICE_API float sigma_t(const AT_NAME::MediumParameter& medium)
        {
            return medium.sigma_a + medium.sigma_s;
        }

        static AT_DEVICE_API float Transmittance(const float sigma_t, const float distance)
        {
            return aten::exp(-sigma_t * distance);
        }

        static AT_DEVICE_API float Transmittance(
            const float sigma_t,
            const aten::vec3& p1, const aten::vec3& p2)
        {
            const auto distance = length(p1 - p2);
            return aten::exp(-sigma_t * distance);
        }

        static AT_DEVICE_API float TransmittanceFromMediumParam(
            const aten::MediumParameter& medium,
            const aten::vec3& p1, const aten::vec3& p2)
        {
            const auto sigma_t = HomogeniousMedium::sigma_t(medium);
            return Transmittance(sigma_t, p1, p2);
        }

        static AT_DEVICE_API float TransmittanceFromMediumParam(
            const aten::MediumParameter& medium,
            const float distance)
        {
            const auto sigma_t = HomogeniousMedium::sigma_t(medium);
            return Transmittance(sigma_t, distance);
        }

        static AT_DEVICE_API AT_NAME::MaterialParameter CreateMaterialParameter(
            const float g,
            const float sigma_a,
            const float sigma_s,
            const aten::vec3& le)
        {
            AT_NAME::MaterialParameter mtrl;

            // TODO
            mtrl.type = AT_NAME::MaterialType::MaterialTypeMax;

            mtrl.is_medium = true;
            mtrl.medium.phase_function_g = g;
            mtrl.medium.sigma_a = sigma_a;
            mtrl.medium.sigma_s = sigma_s;
            mtrl.medium.le = le;

            mtrl.attrib.isEmissive = false;
            mtrl.attrib.isSingular = false;
            mtrl.attrib.isGlossy = false;
            mtrl.attrib.isTranslucent = false;

            return mtrl;
        }
    };

    class HeterogeneousMedium {
    public:
        HeterogeneousMedium() = delete;
        ~HeterogeneousMedium() = delete;

        HeterogeneousMedium(const HeterogeneousMedium&) = delete;
        HeterogeneousMedium(HeterogeneousMedium&&) = delete;
        HeterogeneousMedium& operator=(const HeterogeneousMedium&) = delete;
        HeterogeneousMedium& operator=(HeterogeneousMedium&&) = delete;

        template <template<class> class NanoVdbBuildTraits, class NanoVdbHandle>
        static auto EvalMajorant(
            NanoVdbHandle& handle,
            nanovdb::FloatGrid* grid,
            const float sigma_a, const float sigma_s)
            -> std::enable_if_t<NanoVdbBuildTraits<typename NanoVdbHandle::BufferType>::hasDeviceDual, float>
        {
            return EvalMajorant(grid, sigma_a, sigma_s);
        }

        static AT_DEVICE_API aten::tuple<bool, aten::ray> Sample(
            AT_NAME::PathThroughput& throughput,
            AT_NAME::sampler& sampler,
            const aten::ray& curr_ray,
            const aten::MediumParameter& param,
            nanovdb::FloatGrid* grid,
            const float min_s, const float max_s);

    private:
        static float EvalMajorant(
            nanovdb::FloatGrid* grid,
            const float sigma_a, const float sigma_s);
    };
}
