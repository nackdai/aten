#pragma once

#include <functional>
#include <map>
#include <stack>

#include "accelerator/accelerator.h"
#include "geometry/object.h"
#include "geometry/transformable.h"
#include "scene/hitable.h"

namespace aten {
    class bvhnode;

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
        virtual const aabb& getBoundingbox() const override;

        /**
         * @brief Return the root node of the tree.
         */
        std::shared_ptr<bvhnode> getRoot() const;

        /**
         * @brief Return the root node of the tree.
         */
        bvhnode* getRoot();

        /**
         * @brief Draw all node's AABB in the structure tree.
         */
        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtxL2W) override;

        /**
         * @brief Update the tree.
         */
        virtual void update(const context& ctxt) override;

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
            const std::shared_ptr<bvhnode>& root,
            hitable** list,
            uint32_t num,
            int32_t depth,
            const std::shared_ptr<bvhnode>& parent);

    protected:
        // Root node.
        std::shared_ptr<bvhnode> m_root;

        // Array of the node which will be re-fitted.
        std::vector<bvhnode*> m_refitNodes;
    };
}
