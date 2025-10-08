#pragma once

#include "scene/hitable.h"
#include "accelerator/bvh.h"

namespace aten {
    struct StacklessBvhNode {
        aten::vec3 boxmin_0;    ///< Child AABB min position.
        float parent{ -1 };        ///< Parent index.

        aten::vec3 boxmax_0;    ///< Child AABB max position.
        float sibling{ -1 };    ///< Sibling index.

        aten::vec3 boxmin_1;    ///< Child AABB min position.
        float child_0{ -1 };

        aten::vec3 boxmax_1;    ///< Child AABB max position.
        float child_1{ -1 };

        float object_id{ -1 };    ///< Object index.
        float primid{ -1 };        ///< Triangle index.
        float exid{ -1 };        ///< External bvh index.
        float meshid{ -1 };        ///< Mesh id.

        bool isLeaf() const
        {
            return (object_id >= 0 || primid >= 0);
        }
    };

    class StacklessBVH : public accelerator {
    public:
        StacklessBVH() : accelerator(AccelType::StacklessBvh) {}
        virtual ~StacklessBVH() {}

    public:
        virtual void build(
            const context& ctxt,
            hitable** list,
            uint32_t num,
            aabb* bbox = nullptr) override;

        virtual bool hit(
            const context& ctxt,
            const ray& r,
            float t_min, float t_max,
            Intersection& isect) const override;

        virtual bool HitWithLod(
            const context& ctxt,
            const ray& r,
            float t_min, float t_max,
            bool enableLod,
            Intersection& isect,
            aten::HitStopType hit_stop_type = aten::HitStopType::Closest) const override
        {
            return hit(ctxt, 0, m_listStacklessBvhNode, r, t_min, t_max, isect, hit_stop_type);
        }

        std::vector<std::vector<StacklessBvhNode>>& getNodes()
        {
            return m_listStacklessBvhNode;
        }

    private:
        struct StacklessBvhNodeEntry {
            bvhnode* node;
            hitable* nestParent;
            aten::mat4 mtx_L2W;

            StacklessBvhNodeEntry(bvhnode* n, hitable* p, const aten::mat4& m)
                : node(n), nestParent(p), mtx_L2W(m)
            {}
        };

        void registerBvhNodeToLinearList(
            bvhnode* root,
            bvhnode* parentNode,
            hitable* nestParent,
            const aten::mat4& mtx_L2W,
            std::vector<StacklessBvhNodeEntry>& listBvhNode,
            std::vector<accelerator*>& listBvh,
            std::map<hitable*, accelerator*>& nestedBvhMap);

        void registerThreadedBvhNode(
            const context& ctxt,
            bool isPrimitiveLeaf,
            const std::vector<StacklessBvhNodeEntry>& listBvhNode,
            std::vector<StacklessBvhNode>& listStacklessBvhNode);

        bool hit(
            const context& ctxt,
            int32_t exid,
            const std::vector<std::vector<StacklessBvhNode>>& listGpuBvhNode,
            const ray& r,
            float t_min, float t_max,
            Intersection& isect,
            aten::HitStopType hit_stop_type = aten::HitStopType::Closest) const;

    private:
        bvh bvh_;

        std::vector<std::vector<StacklessBvhNode>> m_listStacklessBvhNode;
    };
}
