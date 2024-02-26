#pragma once

#include "geometry/geomparam.h"
#include "geometry/sphere.h"
#include "geometry/PolygonObject.h"
#include "scene/hit_parameter.h"

namespace AT_NAME
{
    template <class CONTEXT>
    inline AT_HOST_DEVICE_API void evaluate_hit_result(
        aten::hitrecord& rec,
        const aten::ObjectParameter& obj,
        const CONTEXT& ctxt,
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

            aten::mat4 mtx_L2W;
            if (mtx_id >= 0) {
                mtx_L2W = ctxt.GetMatrix(mtx_id);
            }

            if (real_obj.type == aten::ObjectType::Polygons) {
                AT_NAME::PolygonObject::evaluate_hit_result(
                    real_obj,
                    ctxt,
                    r,
                    mtx_L2W,
                    rec,
                    isect);
            }
            else if (real_obj.type == aten::ObjectType::Sphere) {
                AT_NAME::sphere::EvaluateHitResult(&real_obj, r, &rec, &isect);
            }
            else {
                // TODO
            }

            // Transform local to world.
            rec.p = mtx_L2W.apply(rec.p);
            rec.normal = normalize(mtx_L2W.applyXYZ(rec.normal));

            rec.isVoxel = false;
        }

        rec.mtrlid = isect.mtrlid;
        rec.meshid = isect.meshid;
    }

    template <class CONTEXT>
    inline AT_HOST_DEVICE_API void SamplePosAndNormal(
        aten::SamplePosNormalPdfResult* result,
        const aten::ObjectParameter& obj,
        const CONTEXT& ctxt,
        aten::sampler* sampler)
    {
        // Get real object. If the object is instance, we need to get real one.
        const auto& real_obj = obj.object_id >= 0 ? ctxt.GetObject(obj.object_id) : obj;
        const auto mtx_id = obj.object_id >= 0 ? obj.mtx_id : real_obj.mtx_id;

        aten::mat4 mtx_L2W;
        if (mtx_id >= 0) {
            mtx_L2W = ctxt.GetMatrix(mtx_id);
        }

        if (real_obj.type == aten::ObjectType::Polygons) {
            AT_NAME::PolygonObject::SamplePosAndNormal(
                result,
                real_obj,
                ctxt,
                mtx_L2W,
                sampler);
        }
        else if (real_obj.type == aten::ObjectType::Sphere) {
            AT_NAME::sphere::SamplePosAndNormal(result, real_obj, mtx_L2W, sampler);
        }
        else {
            // TODO
        }
    }
}
