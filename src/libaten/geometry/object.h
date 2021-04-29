#pragma once

#include <memory>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/face.h"
#include "geometry/objshape.h"
#include "geometry/transformable.h"
#include "scene/context.h"

namespace AT_NAME
{
    class object : public aten::transformable {
        friend class TransformableFactory;

    public:
        object()
            : transformable(aten::GeometryType::Polygon)
        {}

        virtual ~object();

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection& isect) const override final;

        virtual void evalHitResult(
            const aten::context& ctxt,
            const aten::ray& r,
            const aten::mat4& mtxL2W,
            aten::hitrecord& rec,
            const aten::Intersection& isect) const override final;

        virtual aten::accelerator* getInternalAccelerator() override final
        {
            return m_accel.get();
        }

        virtual void drawForGBuffer(
            aten::hitable::FuncPreDraw func,
            const aten::context& ctxt,
            const aten::mat4& mtxL2W,
            const aten::mat4& mtxPrevL2W,
            int parentId,
            uint32_t triOffset) override final;

        void draw(
            AT_NAME::FuncObjectMeshDraw func,
            const aten::context& ctxt) const;

        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtxL2W) override final;

        bool exportInternalAccelTree(
            const aten::context& ctxt,
            const char* path);

        bool importInternalAccelTree(const char* path, const aten::context& ctxt, int offsetTriIdx);

        void buildForRasterizeRendering(const aten::context& ctxt);

        void gatherTrianglesAndMaterials(
            std::vector<std::vector<AT_NAME::face*>>& tris,
            std::vector<AT_NAME::material*>& mtrls);

        virtual void collectTriangles(std::vector<aten::PrimitiveParamter>& triangles) const override final;

        virtual uint32_t getTriangleCount() const override final
        {
            return m_triangles;
        }

        void build(const aten::context& ctxt);

        virtual void getSamplePosNormalArea(
            const aten::context& ctxt,
            aten::hitable::SamplePosNormalPdfResult* result,
            const aten::mat4& mtxL2W,
            aten::sampler* sampler) const override final;

        void appendShape(const std::shared_ptr<objshape>& shape)
        {
            AT_ASSERT(shape);
            m_shapes.push_back(shape);
        }

        uint32_t getShapeNum() const
        {
            return static_cast<uint32_t>(m_shapes.size());
        }

        objshape* getShape(uint32_t idx)
        {
            AT_ASSERT(idx < getShapeNum());
            return m_shapes[idx].get();
        }

    private:
        std::vector<std::shared_ptr<objshape>> m_shapes;

        std::shared_ptr<aten::accelerator> m_accel;
        uint32_t m_triangles{ 0 };
    };
}
