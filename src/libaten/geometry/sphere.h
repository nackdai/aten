#pragma once

#include "types.h"
#include "math/mat4.h"
#include "geometry/transformable.h"
#include "geometry/geombase.h"
#include "geometry/geomparam.h"
#include "scene/context.h"

namespace AT_NAME
{
    class sphere : public aten::geom<aten::transformable> {
        friend class TransformableFactory;

    public:
        template <typename MTRL>
        sphere(const aten::vec3& center, real radius, const MTRL& mtrl)
            : aten::transformable(aten::GeometryType::Sphere, mtrl)
        {
            m_param.sphere.center = center;
            m_param.sphere.radius = radius;

            auto _min = center - radius;
            auto _max = center + radius;

            setBoundingBox(aten::aabb(_min, _max));
        }

        virtual ~sphere() {}

    public:
        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection& isect) const override final;

        static AT_DEVICE_API bool hit(
            const aten::GeometryParameter* param,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection* isect);

        virtual void evalHitResult(
            const aten::context& ctxt,
            const aten::ray& r,
            aten::hitrecord& rec,
            const aten::Intersection& isect) const override final;

        static AT_DEVICE_API void evalHitResult(
            const aten::GeometryParameter* param,
            const aten::ray& r,
            aten::hitrecord* rec,
            const aten::Intersection* isect);

        static AT_DEVICE_API void evalHitResult(
            const aten::GeometryParameter* param,
            const aten::ray& r,
            const aten::mat4& mtxL2W,
            aten::hitrecord* rec,
            const aten::Intersection* isect);

        const aten::vec3& center() const
        {
            return m_param.sphere.center;
        }

        real radius() const
        {
            return m_param.sphere.radius;
        }

        virtual void getSamplePosNormalArea(
            const aten::context& ctxt,
            aten::SamplePosNormalPdfResult* result,
            aten::sampler* sampler) const override final;

        static AT_DEVICE_API void getSamplePosNormalArea(
            aten::SamplePosNormalPdfResult* result,
            const aten::GeometryParameter* param,
            aten::sampler* sampler);

        virtual void evalHitResult(
            const aten::context& ctxt,
            const aten::ray& r,
            const aten::mat4& mtxL2W,
            aten::hitrecord& rec,
            const aten::Intersection& isect) const override final;

        virtual void getSamplePosNormalArea(
            const aten::context& ctxt,
            aten::SamplePosNormalPdfResult* result,
            const aten::mat4& mtxL2W,
            aten::sampler* sampler) const override final;

    private:
        static AT_DEVICE_API void getSamplePosNormalArea(
            aten::SamplePosNormalPdfResult* result,
            const aten::GeometryParameter* param,
            const aten::mat4& mtxL2W,
            aten::sampler* sampler);

    private:
        aten::GeometryParameter m_param;
    };
}
