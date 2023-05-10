#pragma once

#include <atomic>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/geomparam.h"
#include "geometry/transformable.h"
#include "geometry/NoHitableMesh.h"
#include "geometry/vertex.h"

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
            const aten::vertex& v0,
            const aten::vertex& v1,
            const aten::vertex& v2,
            aten::hitrecord* rec,
            const aten::TriangleParameter& param,
            const aten::Intersection* isect);

        static void sample_pos_and_normal(
            const aten::vertex& v0,
            const aten::vertex& v1,
            const aten::vertex& v2,
            aten::SamplePosNormalPdfResult* result,
            aten::sampler* sampler);

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