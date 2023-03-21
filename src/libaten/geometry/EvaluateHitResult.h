#pragma once

#include "geometry/geomparam.h"
#include "geometry/sphere.h"
#include "geometry/PolygonObject.h"
#include "scene/hit_parameter.h"

namespace AT_NAME
{
    AT_DEVICE_API inline void evaluate_hit_result(
        const context& ctxt,
        const aten::ObjectParameter& obj,
        const aten::ray& r,
        aten::hitrecord& rec,
        const aten::Intersection& isect)
    {
        if (isect.isVoxel) {
            // For voxel.

            // Repair normal.
            rec.normal = normalize(aten::vec3(isect.nml_x, isect.nml_y, isect.nml_z));

            // Compute hit point.
            rec.p = r.org + isect.t * r.dir;
            rec.p = rec.p + AT_MATH_EPSILON * rec.normal;

            rec.isVoxel = true;
        }
        else {
            // Get real object. If the object is instance, we need to get real one.
            const auto& real_obj = obj.object_id >= 0 ? ctxt.get_object(obj.object_id) : obj;

            // TODO
            aten::mat4 mtxL2W;

            if (real_obj.type == aten::ObjectType::Polygon) {
                AT_NAME::PolygonObject::evaluate_hit_result(
                    real_obj,
                    ctxt,
                    r,
                    mtxL2W,
                    rec,
                    isect);
            }
            else if (real_obj.type == aten::ObjectType::Sphere) {
                AT_NAME::sphere::evalHitResult(&real_obj, r, &rec, &isect);
            }
            else {
                // TODO
            }

            rec.isVoxel = false;
        }

        rec.mtrlid = isect.mtrlid;
        rec.meshid = isect.meshid;
    }
}
