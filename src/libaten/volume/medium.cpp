#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/Ray.h>

#include "volume/medium.h"

#include "volume/grid.h"

namespace AT_NAME {
    AT_DEVICE_API float HeterogeneousMedium::EvalMajorant(
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
        inline AT_DEVICE_API float GetValueInGrid(nanovdb::FloatGrid* grid, const aten::ray& ray, float t)
        {
            // TODO
            // If the advanced ray can be passed, no need to advance the ray with t.

            nanovdb::Ray<float> world_ray(
                nanovdb::Vec3f(ray.org.x, ray.org.y, ray.org.z),
                nanovdb::Vec3f(ray.dir.x, ray.dir.y, ray.dir.z));

            auto world_pos = world_ray(t);

            auto index_pos = grid->worldToIndexF(world_pos);

            auto accessor = grid->getAccessor();

            // TODO
            // Tri linear sampling.
            //nanovdb::TrilinearSampler<decltype(accessor)> sampler(accessor);

            const auto value = accessor.getValue(nanovdb::Coord::Floor(index_pos));
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
        aten::sampler& sampler,
        const aten::ray& ray,
        const aten::MediumParameter& param,
        nanovdb::FloatGrid* grid)
    {
        auto clip_info = AT_NAME::Grid::ClipRayByGridBoundingBox(ray, grid);
        if (!clip_info.has_value()) {
            return aten::make_tuple(false, ray);
        }

        float min_s, max_s;
        aten::tie(min_s, max_s) = clip_info.value();

        AT_ASSERT(param.majorant > 0.0F);

        // TODO
        // Not support absorption yet.
        AT_ASSERT(param.sigma_a == 0.0F);

        auto curr_ray = ray;
        auto sample_s = min_s;

        while (true) {
            const auto u = sampler.nextSample();
            const auto ds = -aten::log(1.0F - u) / param.majorant;
            sample_s += ds;

            // Hit volume boundary.
            if (sample_s >= max_s) {
                // Same direction, but next ray origin is the bound of the volume.
                aten::ray next_ray{ curr_ray };
                next_ray.org += next_ray.dir * max_s;
                return aten::make_tuple(false, next_ray);
            }

            // Compute based on the current ray in every loop.
            // So, it's ok to adovance the ray origin with sampled s not delta s.
            aten::ray next_ray{ curr_ray };
            next_ray.org += next_ray.dir * sample_s;

            // Compute sigma_a, sigma_s, sigma_n at this position.
            const auto density = _nanovdb_detail::GetValueInGrid(grid, curr_ray, sample_s);
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

            const auto tr = HomogeniousMedium::EvaluateTransmittance(param.majorant, ds);
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

    AT_DEVICE_API float HeterogeneousMedium::EvaluateTransmittance(
        nanovdb::FloatGrid* grid,
        aten::sampler& sampler,
        const aten::MediumParameter& medium,
        const aten::vec3& p1, const aten::vec3& p2)
    {
        aten::ray ray(p1, p2 - p1);

        auto clip_info = AT_NAME::Grid::ClipRayByGridBoundingBox(ray, grid);
        if (!clip_info.has_value()) {
            return 1.0F;
        }

        float min_s, max_s;
        aten::tie(min_s, max_s) = clip_info.value();

        float sample_s = min_s;
        float transmittance = 1.0F;

        while (true) {
            const auto u = sampler.nextSample();
            const auto ds = -aten::log(1.0F - u) / medium.majorant;
            sample_s += ds;

            if (sample_s > max_s) {
                break;
            }

            // Compute based on the current ray in every loop.
            // So, it's ok to adovance the ray origin with sampled s not delta s.
            aten::ray next_ray{ ray };
            next_ray.org += next_ray.dir * sample_s;

            // Compute sigma_a, sigma_s, sigma_n at this position.
            const auto density = _nanovdb_detail::GetValueInGrid(grid, ray, sample_s);
            const auto sigma_a_at = medium.sigma_a * density;
            const auto sigma_s_at = medium.sigma_s * density;
            const auto sigma_n = medium.majorant - sigma_a_at - sigma_s_at;

            const auto r = sampler.nextSample();

            if (r >= sigma_n / medium.majorant) {
                return 0.0F;
            }

            transmittance *= (sigma_n / medium.majorant);
        }

        return transmittance;
    }
}
