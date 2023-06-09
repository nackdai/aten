#pragma once

#include <memory>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/triangle.h"
#include "geometry/TriangleGroupMesh.h"
#include "geometry/transformable.h"
#include "scene/host_scene_context.h"

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

        template <typename CONTEXT>
        static AT_DEVICE_API void evaluate_hit_result(
            const aten::ObjectParameter& obj,
            const CONTEXT& ctxt,
            const aten::ray& r,
            const aten::mat4& mtxL2W,
            aten::hitrecord& rec,
            const aten::Intersection& isect)
        {
            const auto& faceParam = ctxt.GetTriangle(isect.triangle_id);

            AT_NAME::triangle::evalHitResult(ctxt, faceParam, &rec, faceParam, &isect);

            auto p0 = ctxt.GetPositionAsVec4(faceParam.idx[0]);
            auto p1 = ctxt.GetPositionAsVec4(faceParam.idx[1]);
            auto p2 = ctxt.GetPositionAsVec4(faceParam.idx[2]);

            p0.w = p1.w = p2.w = real(1);

            real orignalLen = length(p1.v - p0.v);

            real scaledLen = 0;
            {
                auto _p0 = mtxL2W.apply(p0);
                auto _p1 = mtxL2W.apply(p1);

                scaledLen = length(_p1.v - _p0.v);
            }

            real ratio = scaledLen / orignalLen;
            ratio = ratio * ratio;

            rec.area = obj.area * ratio;

            rec.mtrlid = isect.mtrlid;
        }

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
            std::string_view path);

        bool importInternalAccelTree(
            std::string_view path, const aten::context& ctxt, int32_t offsetTriIdx);

        void buildForRasterizeRendering(const aten::context& ctxt);

        virtual void collectTriangles(std::vector<aten::TriangleParameter>& triangles) const override final;

        virtual uint32_t getTriangleCount() const override final
        {
            return m_param.triangle_num;
        }

        void build(const aten::context& ctxt);

        template <typename CONTEXT>
        static AT_DEVICE_API void sample_pos_and_normal(
            aten::SamplePosNormalPdfResult* result,
            const aten::ObjectParameter& param,
            const CONTEXT& ctxt,
            const aten::mat4& mtxL2W,
            aten::sampler* sampler)
        {
            auto r = sampler->nextSample();
            uint32_t tri_idx = static_cast<uint32_t>(param.triangle_num * r);
            tri_idx += param.triangle_id;

            const auto& tri_param = ctxt.GetTriangle(tri_idx);

            auto v0 = ctxt.GetPositionAsVec4(tri_param.idx[0]);
            auto v1 = ctxt.GetPositionAsVec4(tri_param.idx[1]);
            auto v2 = ctxt.GetPositionAsVec4(tri_param.idx[2]);

            v0.w = v1.w = v2.w = real(1);

            real orignalLen = (v1 - v0).length();

            real scaledLen = 0;
            {
                auto p0 = mtxL2W.apply(v0);
                auto p1 = mtxL2W.apply(v1);

                scaledLen = length(p1.v - p0.v);
            }

            real ratio = scaledLen / orignalLen;
            ratio = ratio * ratio;

            auto area = param.area * ratio;

            AT_NAME::triangle::sample_pos_and_normal(
                ctxt,
                tri_param,
                result, sampler);

            result->triangle_id = tri_idx;

            result->area = area;
        }

        void appendShape(const std::shared_ptr<TriangleGroupMesh>& shape)
        {
            AT_ASSERT(shape);
            m_shapes.push_back(shape);
        }

        const std::vector<std::shared_ptr<TriangleGroupMesh>>& getShapes() const
        {
            return m_shapes;
        }

        void setName(std::string_view name)
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
