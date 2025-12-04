#pragma once

#include <memory>

#include "types.h"
#include "math/mat4.h"
#include "geometry/triangle.h"
#include "geometry/transformable.h"

namespace aten {
    class texture;
}

namespace AT_NAME
{
    class TriangleGroupMesh;

    /**
     * @brief Object is polygons to have the multiple materials.
     */
    class PolygonObject : public aten::transformable {
        friend class TransformableFactory;

    public:
        PolygonObject()
            : transformable(aten::ObjectType::Polygons)
        {}

        virtual ~PolygonObject() = default;

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            float t_min, float t_max,
            aten::Intersection& isect) const override final;

        template <class CONTEXT>
        static AT_HOST_DEVICE_API void evaluate_hit_result(
            const aten::ObjectParameter& obj,
            const CONTEXT& ctxt,
            const aten::ray& r,
            const aten::mat4& mtx_L2W,
            aten::hitrecord& rec,
            const aten::Intersection& isect)
        {
            const auto& faceParam = ctxt.GetTriangle(isect.triangle_id);

            AT_NAME::triangle::EvaluateHitResult(ctxt, faceParam, &rec, &isect);

            auto p0{ ctxt.GetPositionAsVec4(faceParam.v0.idx[0]) };
            auto p1{ ctxt.GetPositionAsVec4(faceParam.v0.idx[1]) };
            auto p2{ ctxt.GetPositionAsVec4(faceParam.v0.idx[2]) };

            p0.w = p1.w = p2.w = float(1);

            float orignalLen = length(p1.v - p0.v);

            float scaledLen = 0;
            {
                auto _p0 = mtx_L2W.apply(p0);
                auto _p1 = mtx_L2W.apply(p1);

                scaledLen = length(_p1.v - _p0.v);
            }

            float ratio = scaledLen / orignalLen;
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
            const aten::mat4& mtx_L2W,
            const aten::mat4& mtx_prev_L2W,
            int32_t parentId,
            uint32_t triOffset) override final;

        void draw(
            std::function<void(const aten::vec3&, const aten::texture*, int32_t)> func,
            const aten::context& ctxt) const;

        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtx_L2W) override final;

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

        void build(aten::context& ctxt, const std::optional<aten::vec3>& scale);

        template <class CONTEXT>
        static AT_HOST_DEVICE_API void SamplePosAndNormal(
            aten::SamplePosNormalPdfResult* result,
            const aten::ObjectParameter& param,
            const CONTEXT& ctxt,
            const aten::mat4& mtx_L2W,
            aten::sampler* sampler)
        {
            auto r = sampler->nextSample();
            uint32_t tri_idx = static_cast<uint32_t>(param.triangle_num * r);
            tri_idx += param.triangle_id;

            const auto& tri_param = ctxt.GetTriangle(tri_idx);

            auto v0{ ctxt.GetPositionAsVec4(tri_param.v0.idx[0]) };
            auto v1{ ctxt.GetPositionAsVec4(tri_param.v0.idx[1]) };
            auto v2{ ctxt.GetPositionAsVec4(tri_param.v0.idx[2]) };

            v0.w = v1.w = v2.w = float(1);

            float orignalLen = (v1 - v0).length();

            float scaledLen = 0;
            {
                auto p0 = mtx_L2W.apply(v0);
                auto p1 = mtx_L2W.apply(v1);

                scaledLen = length(p1.v - p0.v);
            }

            float ratio = scaledLen / orignalLen;
            ratio = ratio * ratio;

            auto area = param.area * ratio;

            AT_NAME::triangle::SamplePosAndNormal(
                ctxt,
                tri_param,
                result, sampler);

            result->triangle_id = tri_idx;

            result->area = area;
        }

        void appendShape(const std::shared_ptr<TriangleGroupMesh>& shape);

        const std::vector<std::shared_ptr<TriangleGroupMesh>>& getShapes() const;

    private:
        std::vector<std::shared_ptr<TriangleGroupMesh>> m_shapes;

        std::shared_ptr<aten::accelerator> m_accel;
    };
}
