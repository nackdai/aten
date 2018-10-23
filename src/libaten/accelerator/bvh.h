#pragma once

#include <functional>
#include <stack>

#include "scene/hitable.h"
#include "geometry/transformable.h"
#include "geometry/object.h"
#include "accelerator/accelerator.h"

namespace aten {
    class bvh;

    /**
     * @brief Node in BVH tree.
     */
    class bvhnode {
        friend class bvh;

    private:
        bvhnode(bvhnode* parent, hitable* item, bvh* bvh);
        virtual ~bvhnode() {}

    public:
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
        bvhnode* getLeft()
        {
            return m_left;
        }

        /**
        * @brief Return a right child node.
        */
        bvhnode* getRight()
        {
            return m_right;
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
        int getTraversalOrder() const
        {
            return m_traverseOrder;
        }

        /**
         * @brief Set an order to traversal tree for threaded bvh.
         */
        void setTraversalOrder(int order)
        {
            m_traverseOrder = order;
        }

        /**
         * @brief Return the index of an external tree.
         */
        int getExternalId() const
        {
            return m_externalId;
        }

        /**
        * @brief Set the index of an external tree.
        */
        void setExternalId(int exid)
        {
            m_externalId = exid;
        }

        /**
         * @brief Return the index of an external sub tree (it is for LOD).
         */
        int getSubExternalId() const
        {
            return m_subExternalId;
        }

        /**
         * @brief Set the index of an external sub tree (it is for LOD).
         */
        void setSubExternalId(int exid)
        {
            m_subExternalId = exid;
        }

        /**
         * @brief Return count of children which the node has, for multi bvh.
         */
        int getChildrenNum() const
        {
            return m_childrenNum;
        }

        /**
         * @brief Set count of children which the node has, for multi bvh.
         */
        void setChildrenNum(int num)
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
        void registerChild(hitable* child, int idx)
        {
            m_children[idx] = child;
        }

    private:
        /**
         * @brief Set depth in the tree which the node belonges to.
         */
        void setDepth(int depth)
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
        int getDepth() const
        {
            return m_depth;
        }

        /**
         * @brief This function will be called when the item which the node has move/rotates/scales.
         */
        void itemChanged(hitable* sender);

        /**
         * @brief Try to rotate the position in the tree.
         */
        void tryRotate(bvh* bvh);

        /**
         * @brief Re-fit children's AABB.
         */
        static void refitChildren(bvhnode* node, bool propagate);

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
        bvhnode* m_left{ nullptr };
        bvhnode* m_right{ nullptr };

        bvhnode* m_parent{ nullptr };

        aabb m_aabb;

        union {
            struct {
                hitable* m_item;
                hitable* padding[3];
            };
            hitable* m_children[4];
        };

        // Order to traversal tree for threaded bvh
        int m_traverseOrder{ -1 };

        // Index of an external tree
        int m_externalId{ -1 };

        // Count of children which the node has, for multi bvh.
        int m_childrenNum{ 0 };

        // Index of an external sub tree (it is for LOD).
        int m_subExternalId{ -1 };

        // Depth in the tree which the node belonges to
        int m_depth{ 0 };

        // BVH which the node belongs to.
        bvh* m_bvh{ nullptr };

        bool m_isCandidate{ false };
    };

    //////////////////////////////////////////////

    /**
     * @brief Bounding Volume Hierarchies.
     */
    class bvh : public accelerator {
        friend class bvhnode;
        friend class accelerator;
        friend class ThreadedBVH;

    public:
        bvh() : accelerator(AccelType::Bvh) {}
        virtual ~bvh() {}

    public:
        /**
         * @brief Bulid structure tree from the specified list.
         */
        virtual void build(
            const context& ctxt,
            hitable** list,
            uint32_t num,
            aabb* bbox) override;

        /**
         * @brief Test if a ray hits a object.
         */
        virtual bool hit(
            const context& ctxt,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const override;

        /**
         * @brief Test if a ray hits a object.
         */
        virtual bool hit(
            const context& ctxt,
            const ray& r,
            real t_min, real t_max,
            bool enableLod,
            Intersection& isect) const override
        {
            return hit(ctxt, r, t_min, t_max, isect);
        }

        /**
         * @brief Return AABB.
         */
        virtual const aabb& getBoundingbox() const override
        {
            if (m_root) {
                return std::move(m_root->getBoundingbox());
            }
            return std::move(aabb());
        }

        /**
         * @brief Return the root node of the tree.
         */
        bvhnode* getRoot()
        {
            return m_root;
        }

        /**
         * @brief Return the root of the nested tree which the specified node has.
         */
        static bvhnode* getNestedNode(bvhnode* node);

        /**
         * @brief Convert the tree to the linear list.
         */
        template <typename _T>
        static void registerBvhNodeToLinearList(
            aten::bvhnode* root,
            aten::bvhnode* parentNode,
            aten::hitable* nestParent,
            const aten::mat4& mtxL2W,
            std::vector<_T>& listBvhNode,
            std::vector<aten::accelerator*>& listBvh,
            std::map<hitable*, aten::accelerator*>& nestedBvhMap,
            std::function<void(std::vector<_T>&, aten::bvhnode*, aten::hitable*, const aten::mat4&)> funcRegisterToList,
            std::function<void(aten::bvhnode*, int, int)> funcIfInstanceNode)
        {
            if (!root) {
                return;
            }

            auto pnode = root;

            auto original = pnode;

            // ネストしている場合にはネストさきのツリーのルートノードを取得.
            // ネストしていない場合は同じものが返ってくる.
            pnode = getNestedNode(original);

            if (pnode != original) {
                // ネストしている.
                {
                    original->setParent(parentNode);
                    original->setTraversalOrder((int)listBvhNode.size());
                    funcRegisterToList(listBvhNode, original, nestParent, mtxL2W);
                }

                // Register nested bvh.
                aten::hitable* originalItem = original->getItem();
                AT_ASSERT(originalItem->isInstance());

                // Register relation between instance and nested bvh.
                auto internalItem = const_cast<hitable*>(originalItem->getHasObject());
                auto internalSecondItem = const_cast<hitable*>(originalItem->getHasSecondObject());

                hitable* items[] = {
                    internalItem,
                    internalSecondItem,
                };

                int exids[2] = { -1, -1 };

                for (int i = 0; i < AT_COUNTOF(items); i++) {
                    auto item = items[i];

                    if (item == nullptr) {
                        break;
                    }

                    // TODO
                    auto obj = (AT_NAME::object*)item;

                    auto nestedBvh = obj->getInternalAccelerator();

                    auto found = std::find(listBvh.begin(), listBvh.end(), nestedBvh);
                    if (found == listBvh.end()) {
                        listBvh.push_back(nestedBvh);

                        int exid = (int)listBvh.size() - 1;
                        AT_ASSERT(exid >= 0);

                        exids[i] = exid;
                    }

                    if (i == 0) {
                        if (nestedBvh) {
                            nestedBvhMap.insert(std::pair<aten::hitable*, aten::accelerator*>(originalItem, nestedBvh));
                        }
                    }
                }

                if (funcIfInstanceNode) {
                    funcIfInstanceNode(original, exids[0], exids[1]);
                }
            }
            else {
                pnode->setParent(parentNode);
                pnode->setTraversalOrder((int)listBvhNode.size());
                funcRegisterToList(listBvhNode, pnode, nestParent, mtxL2W);

                aten::bvhnode* pleft = pnode->getLeft();
                aten::bvhnode* pright = pnode->getRight();

                registerBvhNodeToLinearList(pleft, pnode, nestParent, mtxL2W, listBvhNode, listBvh, nestedBvhMap, funcRegisterToList, funcIfInstanceNode);
                registerBvhNodeToLinearList(pright, pnode, nestParent, mtxL2W, listBvhNode, listBvh, nestedBvhMap, funcRegisterToList, funcIfInstanceNode);
            }
        }

        /**
         * @brief Draw all node's AABB in the structure tree.
         */
        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtxL2W) override;

        /**
         * @brief Update the tree.
         */
        virtual void update() override;

    private:
        /**
         * @brief Register the node which will be re-fitted.
         */
        void addToRefit(bvhnode* node)
        {
            m_refitNodes.push_back(node);
        }

        /**
         * @brief Test whether a ray is hit to a object.
         */
        static bool onHit(
            const context& ctxt,
            const bvhnode* root,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect);

        /**
         * @brief Build the tree with Sufrace Area Heuristic.
         */
        void buildBySAH(
            bvhnode* root,
            hitable** list,
            uint32_t num,
            int depth,
            bvhnode* parent);

        struct Candidate {
            bvhnode* node{ nullptr };
            bvhnode* instanceNode{ nullptr };

            Candidate(bvhnode* n, bvhnode* i = nullptr)
            {
                node = n;
                instanceNode = i;
            }
        };

        bool findCandidates(
            bvhnode* node,
            bvhnode* instanceNode,
            const frustum& f,
            std::stack<Candidate>* stack);

        bvhnode* traverse(
            bvhnode* root,
            const frustum& f);

    protected:
        // Root node.
        bvhnode* m_root{ nullptr };

        // Array of the node which will be re-fitted.
        std::vector<bvhnode*> m_refitNodes;
    };
}
