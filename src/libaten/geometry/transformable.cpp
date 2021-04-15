#include "geometry/transformable.h"

namespace aten
{
    void transformable::resetIdWhenAnyTransformableLeave(aten::transformable* obj)
    {
        obj->m_id = obj->m_listItem.currentIndex();
    }

    transformable::transformable()
    {
        m_listItem.init(this, resetIdWhenAnyTransformableLeave);
    }

    transformable::transformable(GeometryType type)
        : transformable()
    {
        m_param.type = type;
    }

    transformable::transformable(GeometryType type, material* mtrl)
        : transformable()
    {
        mtrl_.reset(mtrl);
        m_param.type = type;
    }

    transformable::transformable(GeometryType type, const std::shared_ptr<material>& mtrl)
        : transformable()
    {
        mtrl_ = mtrl;
        m_param.type = type;
    }

    transformable::~transformable()
    {
        m_listItem.leave();
    }
}
