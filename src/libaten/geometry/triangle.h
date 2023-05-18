#pragma once

#include <atomic>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/geomparam.h"
#include "geometry/transformable.h"
#include "geometry/NoHitableMesh.h"
#include "geometry/vertex.h"
#include "scene/host_scene_context.h"

namespace AT_NAME
{
    /**
    * @brief Triangle.
    **/
    class triangle : public aten::hitable {
        friend class context;

    public:
        triangle() = default;
        virtual ~triangle() = default;

        virtual bool hit(
            const aten::context& ctxt,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection& isect) const override;

        static bool hit(
            const aten::TriangleParameter* param,
            const aten::vec3& v0,
            const aten::vec3& v1,
            const aten::vec3& v2,
            const aten::ray& r,
            real t_min, real t_max,
            aten::Intersection* isect);

        static void evalHitResult(
            const aten::context& ctxt,
            const aten::TriangleParameter& tri,
            aten::hitrecord* rec,
            const aten::TriangleParameter& param,
            const aten::Intersection* isect);

        static void sample_pos_and_normal(
            const aten::context& ctxt,
            const aten::TriangleParameter& tri,
            aten::SamplePosNormalPdfResult* result,
            aten::sampler* sampler)
        {
            const auto p0 = ctxt.GetPositionAsVec4(tri.idx[0]);
            const auto p1 = ctxt.GetPositionAsVec4(tri.idx[1]);
            const auto p2 = ctxt.GetPositionAsVec4(tri.idx[2]);

            const auto n0 = ctxt.GetNormalAsVec4(tri.idx[0]);
            const auto n1 = ctxt.GetNormalAsVec4(tri.idx[1]);
            const auto n2 = ctxt.GetNormalAsVec4(tri.idx[2]);

            real r0 = sampler->nextSample();
            real r1 = sampler->nextSample();

            real a = aten::sqrt(r0) * (real(1) - r1);
            real b = aten::sqrt(r0) * r1;

            // dSÀ•WŒn(barycentric coordinates).
            // v0Šî€.
            // p = (1 - a - b)*v0 + a*v1 + b*v2
            aten::vec3 p = (1 - a - b) * p0 + a * p1 + b * p2;

            aten::vec3 n = (1 - a - b) * n0 + a * n1 + b * n2;
            n = normalize(n);

            // ŽOŠpŒ`‚Ì–ÊÏ = ‚Q•Ó‚ÌŠOÏ‚Ì’·‚³ / 2;
            auto e0 = p1 - p0;
            auto e1 = p2 - p0;
            auto area = real(0.5) * cross(e0, e1).length();

            result->pos = p;
            result->nml = n;
            result->area = area;

            result->a = a;
            result->b = b;
        }

        virtual int32_t mesh_id() const override;

        void build(
            const aten::context& ctxt,
            int32_t mtrlid,
            int32_t geomid);

        aten::aabb computeAABB(const aten::context& ctxt) const;

        const aten::TriangleParameter& getParam() const
        {
            return param_;
        }

        int32_t getId() const
        {
            return m_id;
        }

    private:
        static std::shared_ptr<triangle> create(
            const aten::context& ctxt,
            const aten::TriangleParameter& param);

        template <typename T>
        auto updateIndex(T id)
            -> std::enable_if_t<(std::is_signed<T>::value && !std::is_floating_point<T>::value) || std::is_same<T, std::size_t>::value, void>
        {
            m_id = static_cast<decltype(m_id)>(id);
        }

    private:
        aten::TriangleParameter param_;
        int32_t m_id{ -1 };
    };
}
