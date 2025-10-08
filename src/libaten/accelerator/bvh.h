#pragma once

#include <functional>
#include <map>
#include <stack>

#include "accelerator/accelerator.h"
#include "geometry/PolygonObject.h"
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
            float t_min, float t_max,
            Intersection& isect) const override;

        /**
         * @brief Test if a ray hits a object.
         */
        virtual bool HitWithLod(
            const context& ctxt,
            const ray& r,
            float t_min, float t_max,
            bool enableLod,
            Intersection& isect,
            aten::HitStopType hit_stop_type = aten::HitStopType::Closest) const override
        {
            bool is_hit = OnHit(ctxt, root_.get(), r, t_min, t_max, isect, hit_stop_type);
            return is_hit;
        }

        /**
         * @brief Return AABB.
         */
        virtual const aabb& GetBoundingbox() const override;

        /**
         * @brief Return the root node of the tree.
         */
        std::shared_ptr<bvhnode> GetRoot() const;

        /**
         * @brief Return the root node of the tree.
         */
        bvhnode* GetRoot();

        /**
         * @brief Draw all node's AABB in the structure tree.
         */
        virtual void DrawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtx_L2W) override;

        bool IsBuilt() const override;

        std::optional<aten::aabb> GetBoundingBox() const override;

        /**
         * @brief Update the tree.
         */
        virtual void update(const context& ctxt) override;

    private:
        /**
         * @brief Register the node which will be re-fitted.
         */
        void AddToRefit(bvhnode* node)
        {
            refit_nodes_.push_back(node);
        }

        /**
         * @brief Test whether a ray is hit to a object.
         */
        static bool OnHit(
            const context& ctxt,
            const bvhnode* root,
            const ray& r,
            float t_min, float t_max,
            Intersection& isect,
            aten::HitStopType hit_stop_type = aten::HitStopType::Closest);

        /**
         * @brief Build the tree with Sufrace Area Heuristic.
         */
        void BuildBySAH(
            const std::shared_ptr<bvhnode>& root,
            hitable** list,
            uint32_t num,
            int32_t depth,
            const std::shared_ptr<bvhnode>& parent);

    protected:
        // Root node.
        std::shared_ptr<bvhnode> root_;

        // Array of the node which will be re-fitted.
        std::vector<bvhnode*> refit_nodes_;
    };
}
