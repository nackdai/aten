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

    transformable::~transformable()
    {
        m_listItem.leave();
    }
}
