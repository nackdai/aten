#include "geometry/transformable.h"

namespace aten
{
    transformable::transformable(GeometryType type)
    {
        m_param.type = type;
    }

    transformable::transformable(GeometryType type, const std::shared_ptr<material>& mtrl)
    {
        mtrl_ = mtrl;
        m_param.type = type;
    }
}
