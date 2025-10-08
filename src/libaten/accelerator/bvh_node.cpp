#include <random>
#include <vector>

#include "accelerator/bvh.h"
#include "accelerator/bvh_node.h"
#include "geometry/PolygonObject.h"
#include "geometry/transformable.h"

//#define TEST_NODE_LIST
//#pragma optimize( "", off)

namespace aten {
    bvhnode::bvhnode(const std::shared_ptr<bvhnode>& parent, hitable* item, bvh* bvh)
        : parent_(parent.get())
    {
        children_[0] = children_[1] = children_[2] = children_[3] = nullptr;
        item_ = item;

        if (item_) {
            item_->setFuncNotifyChanged(std::bind(&bvhnode::itemChanged, this, std::placeholders::_1));
        }

        bvh_ = bvh;
    }

    bool bvhnode::hit(
        const context& ctxt,
        const ray& r,
        float t_min, float t_max,
        Intersection& isect) const
    {
#if 0
        if (item_) {
            return item_->hit(r, t_min, t_max, isect);
        }
#else
        if (children_num_ > 0) {
            bool is_hit = false;

            for (int32_t i = 0; i < children_num_; i++) {
                Intersection isectTmp;
                isectTmp.t = AT_MATH_INF;
                auto res = children_[i]->hit(ctxt, r, t_min, t_max, isectTmp);

                if (res) {
                    if (isectTmp.t < isect.t) {
                        isect = isectTmp;
                        t_max = isect.t;

                        is_hit = true;
                    }
                }
            }

            return is_hit;
        }
        else if (item_) {
            return item_->hit(ctxt, r, t_min, t_max, isect);
        }
#endif
        else {
            auto bbox = GetBoundingbox();
            auto is_hit = bbox.hit(r, t_min, t_max);

            if (is_hit) {
                is_hit = bvh::OnHit(ctxt, this, r, t_min, t_max, isect);
            }

            return is_hit;
        }
    }

    void bvhnode::DrawAABB(
        aten::hitable::FuncDrawAABB func,
        const aten::mat4& mtx_L2W) const
    {
        if (item_ && item_->getHasObject()) {
            item_->DrawAABB(func, mtx_L2W);
        }
        else {
            auto transofrmedBox = aten::aabb::transform(aabb_, mtx_L2W);

            aten::mat4 mtxScale;
            mtxScale.asScale(transofrmedBox.size());

            aten::mat4 mtxTrans;
            mtxTrans.asTrans(transofrmedBox.minPos());

            aten::mat4 mtx = mtxTrans * mtxScale;

            func(mtx);
        }
    }
}
