#pragma once

#include <functional>
#include <map>
#include <stack>

#include "accelerator/accelerator.h"
#include "geometry/object.h"
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
            real t_min, real t_max,
            Intersection& isect) const;

        /**
         * @brief Return a AABB which the node has.
         */
        const aabb& getBoundingbox() const
        {
#if 0
            if (m_item) {
                return m_item->getTransformedBoundingBox();
            }
            return m_aabb;
#else
            return m_aabb;
#endif
        }

        /**
         * @brief Set AABB for the node.
         */
        void setBoundingBox(const aabb& bbox)
        {
            m_aabb = bbox;
        }

        /**
         * @brief Return if the node is leaf node in the tree.
         */
        bool isLeaf() const
        {
            return (!m_left && !m_right);
        }

        /**
         * @brief Return a left child node.
         */
        std::shared_ptr<const bvhnode> getLeft() const
        {
            return m_left;
        }
        bvhnode* getLeft()
        {
            return m_left.get();
        }

        /**
        * @brief Return a right child node.
        */
        std::shared_ptr<const bvhnode> getRight() const
        {
            return m_right;
        }
        bvhnode* getRight()
        {
            return m_right.get();
        }

        void setParent(const std::shared_ptr<bvhnode>& parent)
        {
            m_parent = parent.get();
        }
        void setParent(bvhnode* parent)
        {
            m_parent = parent;
        }

        bvhnode* getParent()
        {
            return m_parent;
        }

        /**
         * @brief Return a item which the node has.
         */
        hitable* getItem()
        {
            return m_item;
        }

        /**
         * @brief Return an order to traversal tree for threaded bvh.
         */
        int32_t getTraversalOrder() const
        {
            return m_traverseOrder;
        }

        /**
         * @brief Set an order to traversal tree for threaded bvh.
         */
        void setTraversalOrder(int32_t order)
        {
            m_traverseOrder = order;
        }

        /**
         * @brief Return the index of an external tree.
         */
        int32_t getExternalId() const
        {
            return m_externalId;
        }

        /**
        * @brief Set the index of an external tree.
        */
        void setExternalId(int32_t exid)
        {
            m_externalId = exid;
        }

        /**
         * @brief Return the index of an external sub tree (it is for LOD).
         */
        int32_t getSubExternalId() const
        {
            return m_subExternalId;
        }

        /**
         * @brief Set the index of an external sub tree (it is for LOD).
         */
        void setSubExternalId(int32_t exid)
        {
            m_subExternalId = exid;
        }

        /**
         * @brief Return count of children which the node has, for multi bvh.
         */
        int32_t getChildrenNum() const
        {
            return m_childrenNum;
        }

        /**
         * @brief Set count of children which the node has, for multi bvh.
         */
        void setChildrenNum(int32_t num)
        {
            AT_ASSERT((0 <= num) && (num <= 4));
            m_childrenNum = num;
        }

        /**
         * @brief Get a pointer for children array.
         */
        hitable** getChildren()
        {
            return m_children;
        }

        /**
         * @brief Register a child to the children array.
         */
        void registerChild(hitable* child, int32_t idx)
        {
            m_children[idx] = child;
        }

    private:
        /**
         * @brief Set depth in the tree which the node belonges to.
         */
        void setDepth(int32_t depth)
        {
            m_depth = depth;
        }

        /**
        * @brief Propagete depth to children which belong to the node.
        */
        void propageteDepthToChildren()
        {
            if (m_left) {
                m_left->setDepth(m_depth + 1);
                m_left->propageteDepthToChildren();
            }
            if (m_right) {
                m_right->setDepth(m_depth + 1);
                m_right->propageteDepthToChildren();
            }
        }

        /**
         * @brief Return depth in the tree which the node belonges to.
         */
        int32_t getDepth() const
        {
            return m_depth;
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
            m_isCandidate = c;
        }
        bool isCandidate() const
        {
            return m_isCandidate;
        }

        /**
         * @brief Draw AABB.
         */
        void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtxL2W) const;

    protected:
        std::shared_ptr<bvhnode> m_left;
        std::shared_ptr<bvhnode> m_right;

        // NOTE
        // To avoid circular reference, I can't use shared_ptr for this.
        // On the other hand, I need to replace this value frequently for bvh updating.
        // Therefore, it is difficult to define this as weak_ptr.
        bvhnode* m_parent;

        aabb m_aabb;

        union {
            struct {
                hitable* m_item;
                hitable* padding[3];
            };
            hitable* m_children[4];
        };

        // Order to traversal tree for threaded bvh
        int32_t m_traverseOrder{ -1 };

        // Index of an external tree
        int32_t m_externalId{ -1 };

        // Count of children which the node has, for multi bvh.
        int32_t m_childrenNum{ 0 };

        // Index of an external sub tree (it is for LOD).
        int32_t m_subExternalId{ -1 };

        // Depth in the tree which the node belonges to
        int32_t m_depth{ 0 };

        // BVH which the node belongs to.
        bvh* m_bvh{ nullptr };

        bool m_isCandidate{ false };
    };
}
