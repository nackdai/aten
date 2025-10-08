#pragma once

#include <functional>
#include <map>
#include <stack>

#include "accelerator/accelerator.h"
#include "geometry/PolygonObject.h"
#include "geometry/transformable.h"
#include "scene/hitable.h"

namespace aten {
    class bvh;

    /**
     * @brief Node in BVH tree.
     */
    class bvhnode {
        friend class bvh;

    public:
        bvhnode(const std::shared_ptr<bvhnode>& parent, hitable* item, bvh* bvh);
        virtual ~bvhnode() {}

        /**
         * @brief Test if a ray hits the node.
         */
        bool hit(
            const context& ctxt,
            const ray& r,
            float t_min, float t_max,
            Intersection& isect) const;

        /**
         * @brief Return a AABB which the node has.
         */
        const aabb& GetBoundingbox() const
        {
#if 0
            if (item_) {
                return item_->getTransformedBoundingBox();
            }
            return aabb_;
#else
            return aabb_;
#endif
        }

        /**
         * @brief Set AABB for the node.
         */
        void setBoundingBox(const aabb& bbox)
        {
            aabb_ = bbox;
        }

        /**
         * @brief Return if the node is leaf node in the tree.
         */
        bool isLeaf() const
        {
            return (!left_ && !right_);
        }

        /**
         * @brief Return a left child node.
         */
        std::shared_ptr<const bvhnode> getLeft() const
        {
            return left_;
        }
        bvhnode* getLeft()
        {
            return left_.get();
        }

        /**
        * @brief Return a right child node.
        */
        std::shared_ptr<const bvhnode> getRight() const
        {
            return right_;
        }
        bvhnode* getRight()
        {
            return right_.get();
        }

        void setParent(const std::shared_ptr<bvhnode>& parent)
        {
            parent_ = parent.get();
        }
        void setParent(bvhnode* parent)
        {
            parent_ = parent;
        }

        bvhnode* getParent()
        {
            return parent_;
        }

        /**
         * @brief Return a item which the node has.
         */
        hitable* getItem()
        {
            return item_;
        }

        /**
         * @brief Return an order to traversal tree for threaded bvh.
         */
        int32_t getTraversalOrder() const
        {
            return traverse_order_;
        }

        /**
         * @brief Set an order to traversal tree for threaded bvh.
         */
        void setTraversalOrder(int32_t order)
        {
            traverse_order_ = order;
        }

        /**
         * @brief Return the index of an external tree.
         */
        int32_t getExternalId() const
        {
            return external_id_;
        }

        /**
        * @brief Set the index of an external tree.
        */
        void setExternalId(int32_t exid)
        {
            external_id_ = exid;
        }

        /**
         * @brief Return the index of an external sub tree (it is for LOD).
         */
        int32_t getSubExternalId() const
        {
            return sub_external_id_;
        }

        /**
         * @brief Set the index of an external sub tree (it is for LOD).
         */
        void setSubExternalId(int32_t exid)
        {
            sub_external_id_ = exid;
        }

        /**
         * @brief Return count of children which the node has, for multi bvh.
         */
        int32_t getChildrenNum() const
        {
            return children_num_;
        }

        /**
         * @brief Set count of children which the node has, for multi bvh.
         */
        void setChildrenNum(int32_t num)
        {
            AT_ASSERT((0 <= num) && (num <= 4));
            children_num_ = num;
        }

        /**
         * @brief Get a pointer for children array.
         */
        hitable** getChildren()
        {
            return children_;
        }

        /**
         * @brief Register a child to the children array.
         */
        void registerChild(hitable* child, int32_t idx)
        {
            children_[idx] = child;
        }

    private:
        /**
         * @brief Set depth in the tree which the node belonges to.
         */
        void setDepth(int32_t depth)
        {
            depth_ = depth;
        }

        /**
        * @brief Propagete depth to children which belong to the node.
        */
        void propageteDepthToChildren()
        {
            if (left_) {
                left_->setDepth(depth_ + 1);
                left_->propageteDepthToChildren();
            }
            if (right_) {
                right_->setDepth(depth_ + 1);
                right_->propageteDepthToChildren();
            }
        }

        /**
         * @brief Return depth in the tree which the node belonges to.
         */
        int32_t getDepth() const
        {
            return depth_;
        }

        /**
         * @brief This function will be called when the item which the node has move/rotates/scales.
         */
        void itemChanged(const hitable* sender);

        /**
         * @brief Try to rotate the position in the tree.
         */
        void tryRotate(bvh* bvh);

        /**
         * @brief Re-fit children's AABB.
         */
        static void refitChildren(bvhnode* node, bool propagate);

        static void refitChildren(const std::shared_ptr<bvhnode>& node, bool propagate)
        {
            refitChildren(node.get(), propagate);
        }

        void setIsCandidate(bool c)
        {
            is_candidate_ = c;
        }
        bool isCandidate() const
        {
            return is_candidate_;
        }

        /**
         * @brief Draw AABB.
         */
        void DrawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtx_L2W) const;

    protected:
        std::shared_ptr<bvhnode> left_;
        std::shared_ptr<bvhnode> right_;

        // NOTE
        // To avoid circular reference, I can't use shared_ptr for this.
        // On the other hand, I need to replace this value frequently for bvh updating.
        // Therefore, it is difficult to define this as weak_ptr.
        bvhnode* parent_;

        aabb aabb_;

        union {
            struct {
                hitable* item_;
                hitable* padding[3];
            };
            hitable* children_[4];
        };

        // Order to traversal tree for threaded bvh
        int32_t traverse_order_{ -1 };

        // Index of an external tree
        int32_t external_id_{ -1 };

        // Count of children which the node has, for multi bvh.
        int32_t children_num_{ 0 };

        // Index of an external sub tree (it is for LOD).
        int32_t sub_external_id_{ -1 };

        // Depth in the tree which the node belonges to
        int32_t depth_{ 0 };

        // BVH which the node belongs to.
        bvh* bvh_{ nullptr };

        bool is_candidate_{ false };
    };
}
