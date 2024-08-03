#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/Ray.h>

#include "volume/medium.h"

#include "volume/grid.h"

namespace AT_NAME {
    float HeterogeneousMedium::EvalMajorant(
        nanovdb::FloatGrid* grid,
        const float sigma_a, const float sigma_s)
    {
        if (!grid->hasMinMax()) {
            // Re-compute grid min max.
            nanovdb::gridStats(*grid, nanovdb::StatsMode::MinMax);
        }

        const auto& coord_bbox = grid->indexBBox();
        const auto extrema = nanovdb::getExtrema(*grid, coord_bbox);
        const auto max_density = extrema.max();

        const auto majorant = (sigma_a + sigma_s) * max_density;
        return majorant;
    }

    namespace _nanovdb_detail {
        inline AT_DEVICE_API float GetValueFromGrid(nanovdb::FloatGrid* grid, const aten::ray& ray)
        {
            using Vec3F = nanovdb::Vec3<float>;
            using RayF = nanovdb::Ray<float>;

            RayF world_ray(
                Vec3F(ray.org.x, ray.org.y, ray.org.z),
                Vec3F(ray.dir.x, ray.dir.y, ray.dir.z));

            RayF index_ray = world_ray.worldToIndexF(*grid);

            auto accessor = grid->tree().getAccessor();

            // NOTE:
            // It is assumed that "ray" is already advanced to the position to get the value.
            // It means no need to advance anymore. So, specify 0 to get the current position from "index_ray".
            const auto value = accessor.getValue(nanovdb::Coord::Floor(index_ray(0)));
            return value;
        }

        enum class TrackMode {
            Absorption,
            Scattering,
            Null,
        };
    }

    AT_DEVICE_API aten::tuple<bool, aten::ray> HeterogeneousMedium::Sample(
        AT_NAME::PathThroughput& throughput,
        AT_NAME::sampler& sampler,
        const aten::ray& ray,
        const aten::MediumParameter& param,
        nanovdb::FloatGrid* grid,
        const float min_s, const float max_s)
    {
        AT_ASSERT(param.majorant > 0.0F);

        auto curr_ray = ray;
        auto sample_s = min_s;

        while (true) {
            const auto u = sampler.nextSample();
            sample_s += -aten::log(1.0F - u) / param.majorant;

            // Hit volume boundary.
            if (sample_s >= max_s) {
                // Same direction, but next ray origin is the bound of the volume.
                aten::ray next_ray{ curr_ray };
                next_ray.org += next_ray.dir * max_s;
                return aten::make_tuple(false, next_ray);
            }

            aten::ray next_ray{ curr_ray };
            next_ray.org += next_ray.dir * sample_s;

            // Compute sigma_a, sigma_s, sigma_n at this position.
            const auto density = _nanovdb_detail::GetValueFromGrid(grid, next_ray);
            const auto sigma_a_at = param.sigma_a * density;
            const auto sigma_s_at = param.sigma_s * density;
            const auto sigma_n = param.majorant - sigma_a_at - sigma_s_at;

            const auto prob_sigma_a = sigma_a_at / param.majorant;
            const auto prob_sigma_s = sigma_s_at / param.majorant;

            _nanovdb_detail::TrackMode track_mode = _nanovdb_detail::TrackMode::Null;

            // Judge which should be tracked.
            const auto r = sampler.nextSample();
            if (prob_sigma_s < prob_sigma_a) {
                if (r < prob_sigma_s) {
                    track_mode = _nanovdb_detail::TrackMode::Scattering;
                }
                else if (r < prob_sigma_a) {
                    track_mode = _nanovdb_detail::TrackMode::Absorption;
                }
            }
            else {
                if (r < prob_sigma_a) {
                    track_mode = _nanovdb_detail::TrackMode::Absorption;
                }
                else if (r < prob_sigma_s) {
                    track_mode = _nanovdb_detail::TrackMode::Scattering;
                }
            }

            const auto tr = HomogeniousMedium::Transmittance(param.majorant, sample_s);
            const auto pdf = param.majorant * tr;

            if (track_mode == _nanovdb_detail::TrackMode::Absorption) {
                // TODO
                const auto medium_throughput = tr * (sigma_a_at * param.le / prob_sigma_a) / pdf;
                throughput.throughput *= medium_throughput;
            }
            else if (track_mode == _nanovdb_detail::TrackMode::Scattering) {
                const auto r_0 = sampler.nextSample();
                const auto r_1 = sampler.nextSample();
                next_ray.dir = HenyeyGreensteinPhaseFunction::SampleDirection(r_0, r_1, param.phase_function_g, -curr_ray.dir);
                next_ray.dir = normalize(next_ray.dir);

                // Basic formula for pdf is the same as homogeneous.
                const auto medium_throughput = tr * (sigma_s_at / prob_sigma_s) / pdf;
                throughput.throughput *= medium_throughput;

                return aten::make_tuple(true, next_ray);
            }
        }
    }
}
