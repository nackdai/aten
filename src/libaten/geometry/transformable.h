#pragma once

#include <vector>

#include "types.h"
#include "scene/hitable.h"
#include "math/mat4.h"
#include "geometry/geomparam.h"
#include "scene/hitable.h"

namespace aten
{
    class transformable : public hitable {
        friend class context;

    protected:
        transformable() = default;
        virtual ~transformable() {}

        transformable(ObjectType type)
        {
            m_param.type = type;
        }

    public:
        virtual void getSamplePosNormalArea(
            const context& ctxt,
            aten::SamplePosNormalPdfResult* result,
            const mat4& mtxL2W,
            sampler* sampler) const = 0;

        virtual void evalHitResult(
            const context& ctxt,
            const ray& r,
            const mat4& mtxL2W,
            hitrecord& rec,
            const Intersection& isect) const = 0;

        ObjectType getType() const
        {
            return m_param.type;
        }

        const ObjectParameter& getParam() const
        {
            return m_param;
        }

        ObjectParameter& getParam()
        {
            return m_param;
        }

        virtual void getMatrices(
            aten::mat4& mtxL2W,
            aten::mat4& mtxW2L) const
        {
            mtxL2W.identity();
            mtxW2L.identity();
        }

        int32_t id() const
        {
            return m_id;
        }

        virtual void collectTriangles(std::vector<aten::TriangleParameter>& triangles) const
        {
            // Nothing is done...
        }

    private:
        template <typename T>
        auto updateIndex(T id)
            -> std::enable_if_t<(std::is_signed<T>::value && !std::is_floating_point<T>::value) || std::is_same<T, std::size_t>::value, void>
        {
            m_id = static_cast<decltype(m_id)>(id);
        }

    protected:
        int32_t m_id{ -1 };

        ObjectParameter m_param;
    };
}
