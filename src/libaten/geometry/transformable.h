#pragma once

#include "types.h"
#include "scene/hitable.h"
#include "math/mat4.h"
#include "geometry/geomparam.h"

#include <vector>

namespace aten
{
    class transformable : public hitable {
        static std::vector<transformable*> g_transformables;
        static std::vector<transformable*> g_transformablesPolygonObjList;

    public:
        transformable();
        virtual ~transformable();

    protected:
        transformable(GeometryType type);

    public:
        virtual void getSamplePosNormalArea(
            aten::hitable::SamplePosNormalPdfResult* result,
            const mat4& mtxL2W,
            sampler* sampler) const = 0;

        virtual void evalHitResult(
            const ray& r,
            const mat4& mtxL2W,
            hitrecord& rec,
            const Intersection& isect) const = 0;

        virtual const GeomParameter& getParam() const
        {
            AT_ASSERT(false);
            return std::move(GeomParameter(GeometryType::GeometryTypeMax));
        }

        virtual void getMatrices(
            aten::mat4& mtxL2W,
            aten::mat4& mtxW2L) const
        {
            mtxL2W.identity();
            mtxW2L.identity();
        }

        int id() const
        {
            return m_id;
        }

        virtual void collectTriangles(std::vector<aten::PrimitiveParamter>& triangles) const
        {
            // Nothing is done...
        }

        static uint32_t getShapeNum();
        
        static transformable* getShape(uint32_t idx);
        static transformable* getShapeAsHitable(const hitable* shape);

        static int findIdx(const transformable* shape);
        static int findIdxAsHitable(const hitable* shape);
        static int findIdxFromPolygonObjList(const hitable* shape);
        
        static const std::vector<transformable*>& getShapes();
        static const std::vector<transformable*>& getShapesPolygonObjList();

        static void gatherAllTransformMatrixAndSetMtxIdx(std::vector<aten::mat4>& mtxs);

    private:
        int m_id{ -1 };
    };
}
