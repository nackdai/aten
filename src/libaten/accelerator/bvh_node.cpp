#include <random>
#include <vector>

#include "accelerator/bvh.h"
#include "accelerator/bvh_node.h"
#include "geometry/object.h"
#include "geometry/transformable.h"

//#define TEST_NODE_LIST
//#pragma optimize( "", off)

namespace aten {
    bvhnode::bvhnode(const std::shared_ptr<bvhnode>& parent, hitable* item, bvh* bvh)
        : m_parent(parent.get())
    {
        m_children[0] = m_children[1] = m_children[2] = m_children[3] = nullptr;
        m_item = item;

        if (m_item) {
            m_item->setFuncNotifyChanged(std::bind(&bvhnode::itemChanged, this, std::placeholders::_1));
        }

        m_bvh = bvh;
    }

    bool bvhnode::hit(
        const context& ctxt,
        const ray& r,
        real t_min, real t_max,
        Intersection& isect) const
    {
#if 0
        if (m_item) {
            return m_item->hit(r, t_min, t_max, isect);
        }
#else
        if (m_childrenNum > 0) {
            bool isHit = false;

            for (int i = 0; i < m_childrenNum; i++) {
                Intersection isectTmp;
                isectTmp.t = AT_MATH_INF;
                auto res = m_children[i]->hit(ctxt, r, t_min, t_max, isectTmp);

                if (res) {
                    if (isectTmp.t < isect.t) {
                        isect = isectTmp;
                        t_max = isect.t;

                        isHit = true;
                    }
                }
            }

            return isHit;
        }
        else if (m_item) {
            return m_item->hit(ctxt, r, t_min, t_max, isect);
        }
#endif
        else {
            auto bbox = getBoundingbox();
            auto isHit = bbox.hit(r, t_min, t_max);

            if (isHit) {
                isHit = bvh::onHit(ctxt, this, r, t_min, t_max, isect);
            }

            return isHit;
        }
    }

    void bvhnode::drawAABB(
        aten::hitable::FuncDrawAABB func,
        const aten::mat4& mtxL2W) const
    {
        if (m_item && m_item->getHasObject()) {
            m_item->drawAABB(func, mtxL2W);
        }
        else {
            auto transofrmedBox = aten::aabb::transform(m_aabb, mtxL2W);

            aten::mat4 mtxScale;
            mtxScale.asScale(transofrmedBox.size());

            aten::mat4 mtxTrans;
            mtxTrans.asTrans(transofrmedBox.minPos());

            aten::mat4 mtx = mtxTrans * mtxScale;

            func(mtx);
        }
    }
}
