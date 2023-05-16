#pragma once

#include "geometry/geomparam.h"
#include "geometry/sphere.h"
#include "geometry/PolygonObject.h"
#include "scene/hit_parameter.h"

namespace AT_NAME
{
    inline AT_DEVICE_API void evaluate_hit_result(
        aten::hitrecord& rec,
        const aten::ObjectParameter& obj,
        const AT_NAME::context& ctxt,
        const aten::ray& r,
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
            const auto& real_obj = obj.object_id >= 0 ? ctxt.GetObject(obj.object_id) : obj;
            const auto mtx_id = obj.object_id >= 0 ? obj.mtx_id : real_obj.mtx_id;

            aten::mat4 mtxL2W;
            if (mtx_id >= 0) {
                mtxL2W = ctxt.GetMatrix(mtx_id);
            }

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

            // Transform local to world.
            rec.p = mtxL2W.apply(rec.p);
            rec.normal = normalize(mtxL2W.applyXYZ(rec.normal));

            rec.isVoxel = false;
        }

        rec.mtrlid = isect.mtrlid;
        rec.meshid = isect.meshid;
    }

    inline AT_DEVICE_API void sample_pos_and_normal(
        aten::SamplePosNormalPdfResult* result,
        const aten::ObjectParameter& obj,
        const AT_NAME::context& ctxt,
        aten::sampler* sampler)
    {
        // Get real object. If the object is instance, we need to get real one.
        const auto& real_obj = obj.object_id >= 0 ? ctxt.GetObject(obj.object_id) : obj;
        const auto mtx_id = obj.object_id >= 0 ? obj.mtx_id : real_obj.mtx_id;

        aten::mat4 mtxL2W;
        if (mtx_id >= 0) {
            mtxL2W = ctxt.GetMatrix(mtx_id);
        }

        if (real_obj.type == aten::ObjectType::Polygon) {
            PolygonObject::sample_pos_and_normal(
                result,
                real_obj,
                ctxt,
                mtxL2W,
                sampler);
        }
        else if (real_obj.type == aten::ObjectType::Sphere) {
            sphere::sample_pos_and_normal(result, real_obj, mtxL2W, sampler);
        }
        else {
            // TODO
        }
    }
}
