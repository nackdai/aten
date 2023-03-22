#include "accelerator/accelerator.h"
#include "scene/hitable.h"

namespace aten
{
    accelerator* hitable::getInternalAccelerator()
    {
        return nullptr;
    }

    void hitable::evalHitResult(
        const context& ctxt,
        const hitable* obj,
        const ray& r,
        hitrecord& rec,
        const Intersection& isect)
    {
        if (isect.isVoxel) {
            // For voxel.

            // Compute hit point.
            rec.p = r.org + isect.t * r.dir;
            rec.p = rec.p + AT_MATH_EPSILON * rec.normal;

            rec.normal.x = isect.nml_x;
            rec.normal.y = isect.nml_y;
            rec.normal.z = isect.nml_z;

            rec.mtrlid = isect.mtrlid;

            rec.isVoxel = true;
        }
        else {
            obj->evalHitResult(ctxt, r, rec, isect);
            rec.mtrlid = isect.mtrlid;
            rec.meshid = isect.meshid;

            rec.isVoxel = false;
        }
    }

    void hitable::evalHitResultForAreaLight(
        const context& ctxt,
        const hitable* obj,
        const ray& r,
        hitrecord& rec,
        const Intersection& isect)
    {
        obj->evalHitResult(ctxt, r, rec, isect);
        rec.mtrlid = isect.mtrlid;
        rec.meshid = isect.meshid;
    }
}
