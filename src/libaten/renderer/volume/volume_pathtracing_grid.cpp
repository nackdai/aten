#pragma warning(push)
#pragma warning(disable:4146)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#pragma warning(pop)

#include "renderer/volume/volume_pathtracing.h"

#include "material/material_impl.h"
#include "misc/omputil.h"
#include "misc/timer.h"
#include "renderer/pathtracing/pathtracing_impl.h"
#include "renderer/volume/volume_pathtracing_impl.h"
#include "sampler/cmj.h"
#include "volume/medium.h"
#include "volume/grid.h"

namespace aten
{
    bool VolumePathTracing::ShadeWithGrid(
        int32_t idx,
        aten::Path& paths,
        const context& ctxt,
        ray* rays,
        const aten::Intersection& isect,
        scene* scene,
        int32_t rrDepth,
        int32_t bounce)
    {
        const auto russianProb = AT_NAME::ComputeRussianProbability(
            bounce, rrDepth,
            paths.attrib[idx], paths.throughput[idx],
            paths.sampler[idx]);
        if (paths.attrib[idx].isTerminate) {
            return false;
        }
        paths.throughput[idx].throughput /= russianProb;

        auto* sampler = &paths.sampler[idx];

        auto ray = rays[idx];
        const auto& obj = ctxt.GetObject(static_cast<uint32_t>(isect.objid));

        aten::hitrecord rec;
        AT_NAME::evaluate_hit_result(rec, obj, ctxt, ray, isect);

        bool isBackfacing = dot(rec.normal, -ray.dir) < float(0);

        vec3 orienting_normal = rec.normal;

        aten::MaterialParameter mtrl;
        AT_NAME::FillMaterial(
            mtrl,
            ctxt,
            rec.mtrlid, rec.isVoxel);

        bool is_scattered = false;
        aten::ray next_ray(rec.p, ray.dir);

        auto* grid = (mtrl.is_medium && mtrl.medium.grid_idx >= 0)
            ? ctxt.GetGrid(mtrl.medium.grid_idx)
            : nullptr;

        if (grid) {
            auto clip_info = aten::Grid::ClipRayByGridBoundingBox(ray, grid);
            if (clip_info.has_value()) {
                float t0, t1;
                aten::tie(t0, t1) = clip_info.value();

                aten::tie(is_scattered, next_ray) = AT_NAME::HeterogeneousMedium::Sample(
                    paths.throughput[idx],
                    paths.sampler[idx],
                    ray,
                    mtrl.medium,
                    grid,
                    t0, t1);

                ray = next_ray;
            }
        }

        bool is_reflected_or_refracted = false;

        if (is_scattered) {
            rays[idx] = ray;
        }
        else {
            // Implicit conection to light.
            auto is_hit_implicit_light = AT_NAME::HitImplicitLight(
                ctxt, isect.objid,
                isBackfacing,
                bounce,
                paths.contrib[idx], paths.attrib[idx], paths.throughput[idx],
                ray,
                rec.p, orienting_normal,
                rec.area,
                mtrl);
            if (is_hit_implicit_light) {
                return false;
            }

            const auto curr_ray = ray;

            if (mtrl.is_medium && !AT_NAME::IsSubsurface(mtrl)) {
                auto ray_base_nml = dot(ray.dir, orienting_normal) > 0
                    ? orienting_normal
                    : -orienting_normal;
                rays[idx] = aten::ray(rec.p, ray.dir, ray_base_nml);
            }
            else {
                auto albedo = AT_NAME::sampleTexture(mtrl.albedoMap, rec.u, rec.v, mtrl.baseColor, bounce);

                if (!mtrl.attrib.isTranslucent && isBackfacing) {
                    orienting_normal = -orienting_normal;
                }

                // Apply normal map.
                auto pre_sampled_r = material::applyNormal(
                    &mtrl,
                    mtrl.normalMap,
                    orienting_normal, orienting_normal,
                    rec.u, rec.v,
                    ray.dir, sampler);

                aten::MaterialSampling sampling;
                material::sampleMaterial(
                    &sampling,
                    &mtrl,
                    orienting_normal,
                    ray.dir,
                    rec.normal,
                    sampler, pre_sampled_r,
                    rec.u, rec.v);

                AT_NAME::PrepareForNextBounce(
                    idx,
                    rec, isBackfacing, russianProb,
                    orienting_normal,
                    mtrl, sampling,
                    albedo,
                    paths, rays);

                is_reflected_or_refracted = true;
            }

            AT_NAME::UpdateMedium(curr_ray, rays[idx].dir, orienting_normal, mtrl, paths.throughput[idx].mediums);
        }

        bool will_update_depth = is_scattered || is_reflected_or_refracted;

        return will_update_depth;
    }
}
