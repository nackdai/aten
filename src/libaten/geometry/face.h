#pragma once

#include <atomic>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/geomparam.h"
#include "geometry/transformable.h"
#include "geometry/geombase.h"
#include "geometry/vertex.h"

namespace AT_NAME
{
    class objshape;

    class face : public aten::hitable {
        static std::atomic<int> s_id;
        static std::vector<face*> s_faces;

    public:
        face();
        virtual ~face();

    public:
        virtual bool hit(
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection& isect) const override;

        static bool hit(
            const aten::PrimitiveParamter* param,
            const aten::vec3& v0,
            const aten::vec3& v1,
            const aten::vec3& v2,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection* isect);

        virtual void evalHitResult(
            const aten::ray& r, 
            aten::hitrecord& rec,
            const aten::Intersection& isect) const;

        static void evalHitResult(
            const aten::vertex& v0,
            const aten::vertex& v1,
            const aten::vertex& v2,
            aten::hitrecord* rec,
            const aten::Intersection* isect);

        virtual void getSamplePosNormalArea(
            aten::hitable::SamplePosNormalPdfResult* result,
            aten::sampler* sampler) const override;

        virtual int geomid() const override;

        void build(objshape* _parent);

        aten::aabb computeAABB() const;

        static const std::vector<face*>& faces()
        {
            return s_faces;
        }

        static int findIdx(hitable* h);
    
        aten::PrimitiveParamter param;
        objshape* parent{ nullptr };
        int id{ -1 };
    };
}
