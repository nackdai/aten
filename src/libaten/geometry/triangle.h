#pragma once

#include <optional>

#include "types.h"
#include "math/intersect.h"
#include "math/mat4.h"
#include "geometry/geomparam.h"
#include "geometry/transformable.h"
#include "geometry/NoHitableMesh.h"
#include "geometry/vertex.h"

namespace aten {
    class context;
}

namespace AT_NAME
{
    /**
     * @brief Triangle.
     */
    class triangle : public aten::hitable {
        friend class context;

    public:
        triangle() = default;
        virtual ~triangle() = default;

        triangle(const triangle&) = delete;
        triangle(triangle&&) = delete;
        triangle& operator=(const triangle&) = delete;
        triangle& operator=(triangle&&) = delete;

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            float t_min, float t_max,
            aten::Intersection& isect) const override;

        template <class CONTEXT>
        static AT_HOST_DEVICE_API bool hit(
            const aten::TriangleParameter& param,
            const CONTEXT& ctxt,
            const aten::ray& r,
            aten::Intersection* isect)
        {
            bool isHit = false;

            const auto v0{ ctxt.GetPositionAsVec3(param.idx[0]) };
            const auto v1{ ctxt.GetPositionAsVec3(param.idx[1]) };
            const auto v2{ ctxt.GetPositionAsVec3(param.idx[2]) };

            const auto res = intersectTriangle(r, v0, v1, v2);

            if (res.isIntersect) {
                if (res.t < isect->t) {
                    isect->t = res.t;

                    isect->a = res.a;
                    isect->b = res.b;

                    isHit = true;
                }
            }

            return isHit;
        }

        template <class CONTEXT>
        static AT_HOST_DEVICE_API void EvaluateHitResult(
            const CONTEXT& ctxt,
            const aten::TriangleParameter& tri,
            aten::hitrecord* rec,
            const aten::Intersection* isect)
        {
            const auto p0{ ctxt.GetPositionAsVec4(tri.idx[0]) };
            const auto p1{ ctxt.GetPositionAsVec4(tri.idx[1]) };
            const auto p2{ ctxt.GetPositionAsVec4(tri.idx[2]) };

            const auto n0{ ctxt.GetNormalAsVec4(tri.idx[0]) };
            const auto n1{ ctxt.GetNormalAsVec4(tri.idx[1]) };
            const auto n2{ ctxt.GetNormalAsVec4(tri.idx[2]) };

            // Extract uv.
            const auto u0 = p0.w;
            const auto v0 = n0.w;

            const auto u1 = p1.w;
            const auto v1 = n1.w;

            const auto u2 = p2.w;
            const auto v2 = n2.w;

            // NOTE
            // http://d.hatena.ne.jp/Zellij/20131207/p1

            float a = isect->a;
            float b = isect->b;
            float c = 1 - a - b;

            // 重心座標系(barycentric coordinates).
            // v0基準.
            // p = (1 - a - b)*v0 + a*v1 + b*v2
            rec->p = c * p0 + a * p1 + b * p2;
            rec->normal = c * n0 + a * n1 + b * n2;

            rec->u = c * u0 + a * u1 + b * u2;
            rec->v = c * v0 + a * v1 + b * v2;

            if (tri.needNormal > 0) {
                auto e01 = p1 - p0;
                auto e02 = p2 - p0;

                e01.w = e02.w = float(0);

                rec->normal = normalize(cross(e01, e02));
            }

            rec->area = tri.area;
        }

        template <class CONTEXT>
        static AT_HOST_DEVICE_API void SamplePosAndNormal(
            const CONTEXT& ctxt,
            const aten::TriangleParameter& tri,
            aten::SamplePosNormalPdfResult* result,
            aten::sampler* sampler)
        {
            const auto p0{ ctxt.GetPositionAsVec4(tri.idx[0]) };
            const auto p1{ ctxt.GetPositionAsVec4(tri.idx[1]) };
            const auto p2{ ctxt.GetPositionAsVec4(tri.idx[2]) };

            const auto n0{ ctxt.GetNormalAsVec4(tri.idx[0]) };
            const auto n1{ ctxt.GetNormalAsVec4(tri.idx[1]) };
            const auto n2{ ctxt.GetNormalAsVec4(tri.idx[2]) };

            float r0 = sampler->nextSample();
            float r1 = sampler->nextSample();

            float a = aten::sqrt(r0) * (float(1) - r1);
            float b = aten::sqrt(r0) * r1;

            // 重心座標系(barycentric coordinates).
            // v0基準.
            // p = (1 - a - b)*v0 + a*v1 + b*v2
            aten::vec3 p = (1 - a - b) * p0 + a * p1 + b * p2;

            aten::vec3 n = (1 - a - b) * n0 + a * n1 + b * n2;
            n = normalize(n);

            // 三角形の面積 = ２辺の外積の長さ / 2;
            auto e0 = p1 - p0;
            auto e1 = p2 - p0;
            auto area = float(0.5) * cross(e0, e1).length();

            result->pos = p;
            result->nml = n;
            result->area = area;

            result->a = a;
            result->b = b;
        }

        virtual int32_t GetMeshId() const override;

        void build(
            aten::context& ctxt,
            int32_t mtrlid,
            int32_t geomid,
            std::optional<aten::vec3> scale);

        aten::aabb ComputeAABB(const aten::context& ctxt) const;

        const aten::TriangleParameter& GetParam() const
        {
            return param_;
        }

        int32_t GetId() const
        {
            return m_id;
        }

    private:
        static std::shared_ptr<triangle> create(
            aten::context& ctxt,
            const aten::TriangleParameter& param,
            std::optional<aten::vec3> scale);

        template <class T>
        auto updateIndex(T id)
            -> std::enable_if_t<(std::is_signed<T>::value && !std::is_floating_point<T>::value) || std::is_same<T, std::size_t>::value, void>
        {
            m_id = static_cast<decltype(m_id)>(id);
        }

        void BuildTriangle(
            const aten::context& ctxt,
            const aten::vertex& v0,
            const aten::vertex& v1,
            const aten::vertex& v2,
            int32_t mtrlid,
            int32_t geomid);

    private:
        aten::TriangleParameter param_;
        int32_t m_id{ -1 };
    };
}
