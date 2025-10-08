#pragma once

#include "types.h"
#include "math/mat4.h"
#include "geometry/transformable.h"
#include "geometry/NoHitableMesh.h"
#include "geometry/geomparam.h"
#include "scene/host_scene_context.h"

namespace AT_NAME
{
    class sphere : public virtual aten::transformable, public aten::NoHitableMesh {
        friend class TransformableFactory;

    public:
        sphere(const aten::vec3& center, float radius, std::shared_ptr<AT_NAME::material> mtrl)
            : aten::transformable(aten::ObjectType::Sphere)
        {
            mtrl_ = mtrl;

            param_.sphere.center = center;
            param_.sphere.radius = radius;

            auto _min = center - radius;
            auto _max = center + radius;

            setBoundingBox(aten::aabb(_min, _max));
        }

        virtual ~sphere() {}

    public:
        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            float t_min, float t_max,
            aten::Intersection& isect) const override final;

        static AT_HOST_DEVICE_API bool hit(
            const aten::ObjectParameter* param,
            const aten::ray& r,
            float t_min, float t_max,
            aten::Intersection* isect);

        static AT_HOST_DEVICE_API void EvaluateHitResult(
            const aten::ObjectParameter* param,
            const aten::ray& r,
            aten::hitrecord* rec,
            const aten::Intersection* isect);

        const aten::vec3& center() const
        {
            return param_.sphere.center;
        }

        float radius() const
        {
            return param_.sphere.radius;
        }

        static AT_HOST_DEVICE_API void SamplePosAndNormal(
            aten::SamplePosNormalPdfResult* result,
            const aten::ObjectParameter& param,
            const aten::mat4& mtx_L2W,
            aten::sampler* sampler);

        static AT_HOST_DEVICE_API void SamplePosAndNormal(
            aten::SamplePosNormalPdfResult* result,
            const aten::ObjectParameter& param,
            aten::sampler* sampler)
        {
            SamplePosAndNormal(result, param, aten::mat4(), sampler);
        }

    private:
        std::shared_ptr<AT_NAME::material> mtrl_;
    };
}
