#pragma once

#include <memory>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/triangle.h"
#include "geometry/TriangleGroupMesh.h"
#include "geometry/transformable.h"
#include "scene/context.h"

namespace AT_NAME
{
    /**
    * @brief Object is polygons to have the multiple materials.
    **/
    class PolygonObject : public aten::transformable {
        friend class TransformableFactory;

    public:
        PolygonObject()
            : transformable(aten::ObjectType::Polygon)
        {}

        virtual ~PolygonObject() = default;

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection& isect) const override final;

        static void PolygonObject::evaluate_hit_result(
            const aten::ObjectParameter& obj,
            const context& ctxt,
            const aten::ray& r,
            const aten::mat4& mtxL2W,
            aten::hitrecord& rec,
            const aten::Intersection& isect);

        virtual aten::accelerator* getInternalAccelerator() override final
        {
            return m_accel.get();
        }

        virtual void render(
            aten::hitable::FuncPreDraw func,
            const aten::context& ctxt,
            const aten::mat4& mtxL2W,
            const aten::mat4& mtxPrevL2W,
            int32_t parentId,
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

        bool importInternalAccelTree(const char* path, const aten::context& ctxt, int32_t offsetTriIdx);

        void buildForRasterizeRendering(const aten::context& ctxt);

        virtual void collectTriangles(std::vector<aten::TriangleParameter>& triangles) const override final;

        virtual uint32_t getTriangleCount() const override final
        {
            return m_param.triangle_num;
        }

        void build(const aten::context& ctxt);

        static void sample_pos_and_normal(
            aten::SamplePosNormalPdfResult* result,
            const aten::ObjectParameter& param,
            const aten::context& ctxt,
            const aten::mat4& mtxL2W,
            aten::sampler* sampler);

        void appendShape(const std::shared_ptr<TriangleGroupMesh>& shape)
        {
            AT_ASSERT(shape);
            m_shapes.push_back(shape);
        }

        const std::vector<std::shared_ptr<TriangleGroupMesh>>& getShapes() const
        {
            return m_shapes;
        }

        void setName(const char* name)
        {
            name_.assign(name);
        }

        const char* getName()
        {
            return name_.empty() ? nullptr : name_.c_str();
        }

    private:
        std::vector<std::shared_ptr<TriangleGroupMesh>> m_shapes;

        std::shared_ptr<aten::accelerator> m_accel;

        std::string name_;
    };
}
