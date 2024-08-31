#pragma warning(push)
#pragma warning(disable:4146)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Ray.h>
#pragma warning(pop)

#include "volume/grid.h"

namespace AT_NAME {
    AT_DEVICE_API std::optional<aten::tuple<float, float>> Grid::ClipRayByGridBoundingBox(
        const aten::ray& ray,
        const nanovdb::FloatGrid* grid)
    {
        nanovdb::Ray<float> world_ray(
            nanovdb::Vec3f(ray.org.x, ray.org.y, ray.org.z),
            nanovdb::Vec3f(ray.dir.x, ray.dir.y, ray.dir.z));

        nanovdb::BBoxR test_bbox(nanovdb::Vec3d(-10, -10, -10), nanovdb::Vec3d(10, 10, 10));
        nanovdb::Ray<float> test_ray(
            nanovdb::Vec3f(0, 0, 9),
            nanovdb::Vec3f(0, 0, -1));
        test_ray.clip(test_bbox);

        const auto bbox = grid->worldBBox();
        // Clip to bounds.
        if (world_ray.clip(bbox)) {
            return aten::make_tuple(world_ray.t0(), world_ray.t1());
        }

        return std::nullopt;
    }

    AT_DEVICE_API float Grid::GetValueInGrid(const aten::vec3& p, const nanovdb::FloatGrid* grid)
    {
        // TODO:
        // tri linear sampling etc...
        const auto index = grid->worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
        auto accessor = grid->getAccessor();
        const auto value = accessor.getValue(nanovdb::Coord::Floor(index));
        return value;
    }
}
