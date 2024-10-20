#pragma once

#include <vector>

#include "types.h"
#include "scene/hitable.h"
#include "math/mat4.h"
#include "geometry/geomparam.h"

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
        ObjectType getType() const
        {
            return m_param.type;
        }

        const ObjectParameter& GetParam() const
        {
            return m_param;
        }

        ObjectParameter& GetParam()
        {
            return m_param;
        }

        virtual void getMatrices(
            aten::mat4& mtx_L2W,
            aten::mat4& mtx_W2L) const
        {
            mtx_L2W.identity();
            mtx_W2L.identity();
        }

        int32_t id() const
        {
            return m_id;
        }

        void setName(std::string_view name)
        {
            name_.assign(name);
        }

        const std::string& getName() const
        {
            return name_;
        }

        virtual void collectTriangles(std::vector<aten::TriangleParameter>& triangles) const
        {
            // Nothing is done...
        }

    private:
        template <class T>
        auto updateIndex(T id)
            -> std::enable_if_t<(std::is_signed<T>::value && !std::is_floating_point<T>::value) || std::is_same<T, std::size_t>::value, void>
        {
            m_id = static_cast<decltype(m_id)>(id);
        }

    protected:
        int32_t m_id{ -1 };

        ObjectParameter m_param;

        std::string name_;
    };
}
