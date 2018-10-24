#pragma once

#include "types.h"
#include "math/mat4.h"
#include "geometry/transformable.h"
#include "geometry/geombase.h"
#include "geometry/geomparam.h"

namespace AT_NAME
{
    class cube : public aten::geom<aten::transformable> {
        friend class TransformableFactory;

    private:
        cube(const aten::vec3& center, real w, real h, real d, material* mtrl);

    public:
        virtual ~cube() {}

    public:
        virtual bool hit(
            const context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection& isect) const override final;

        const aten::vec3& center() const
        {
            return m_param.center;
        }

        const aten::vec3& size() const
        {
            return m_param.size;
        }

        virtual void getSamplePosNormalArea(
            const context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            aten::sampler* sampler) const override final;

        virtual void evalHitResult(
            const context& ctxt,
            const aten::ray& r,
            const aten::mat4& mtxL2W,
            aten::hitrecord& rec,
            const aten::Intersection& isect) const override final;

        virtual void getSamplePosNormalArea(
            const context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            const aten::mat4& mtxL2W,
            aten::sampler* sampler) const override final;

    private:
        virtual void evalHitResult(
            const context& ctxt,
            const aten::ray& r,
            aten::hitrecord& rec, 
            const aten::Intersection& isect) const override final;

    private:
        enum Face {
            POS_X,
            NEG_X,
            POS_Y,
            NEG_Y,
            POS_Z,
            NEG_Z,
        };

        Face getRandomPosOn(aten::vec3& pos, aten::sampler* sampler) const;

        static Face findFace(const aten::vec3& d);
    };
}
