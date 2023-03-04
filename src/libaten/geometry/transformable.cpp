#include "geometry/transformable.h"

namespace aten
{
    transformable::transformable(ObjectType type)
    {
        m_param.type = type;
    }

    transformable::transformable(ObjectType type, const std::shared_ptr<material>& mtrl)
    {
        mtrl_ = mtrl;
        m_param.type = type;
    }
}
