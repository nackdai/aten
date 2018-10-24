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

    private:
        sphere(const aten::vec3& center, real radius, material* mtrl);

    public:
        virtual ~sphere() {}

    public:
        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection& isect) const override final;

        static AT_DEVICE_API bool hit(
            const aten::GeomParameter* param,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection* isect);

        virtual void evalHitResult(
            const aten::context& ctxt,
            const aten::ray& r, 
            aten::hitrecord& rec,
            const aten::Intersection& isect) const override final;

        static AT_DEVICE_API void evalHitResult(
            const aten::GeomParameter* param,
            const aten::ray& r,
            aten::hitrecord* rec,
            const aten::Intersection* isect);

        static AT_DEVICE_API void evalHitResult(
            const aten::GeomParameter* param,
            const aten::ray& r,
            const aten::mat4& mtxL2W, 
            aten::hitrecord* rec,
            const aten::Intersection* isect);

        const aten::vec3& center() const
        {
            return m_param.center;
        }

        real radius() const
        {
            return m_param.radius;
        }

        virtual void getSamplePosNormalArea(
            const aten::context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            aten::sampler* sampler) const override final;

        static AT_DEVICE_API void getSamplePosNormalArea(
            aten::hitable::SamplePosNormalPdfResult* result,
            const aten::GeomParameter* param,
            aten::sampler* sampler);

        virtual void evalHitResult(
            const aten::context& ctxt,
            const aten::ray& r,
            const aten::mat4& mtxL2W,
            aten::hitrecord& rec,
            const aten::Intersection& isect) const override final;

        virtual void getSamplePosNormalArea(
            const aten::context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            const aten::mat4& mtxL2W,
            aten::sampler* sampler) const override final;

    private:
        static AT_DEVICE_API void getSamplePosNormalArea(
            aten::hitable::SamplePosNormalPdfResult* result,
            const aten::GeomParameter* param,
            const aten::mat4& mtxL2W,
            aten::sampler* sampler);

    private:
        aten::GeomParameter m_param;
    };
}
