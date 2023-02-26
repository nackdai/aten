#pragma once

#include "accelerator/bvh.h"

namespace aten {
    struct StacklessQbvhNode {
        float leftChildrenIdx{ -1 };
        float isLeaf{ false };
        float numChildren{ 0 };
        float parent{ -1 };

        float sib[4];

        float object_id{ -1 };    ///< Object index.
        float primid{ -1 };        ///< Triangle index.
        float exid{ -1 };        ///< External bvh index.
        float meshid{ -1 };        ///< Mesh id.

        aten::vec4 bminx;
        aten::vec4 bmaxx;

        aten::vec4 bminy;
        aten::vec4 bmaxy;

        aten::vec4 bminz;
        aten::vec4 bmaxz;

        StacklessQbvhNode()
        {
            isLeaf = false;
            numChildren = 0;
            sib[0] = sib[1] = sib[2] = -1;
        }

        StacklessQbvhNode(const StacklessQbvhNode& rhs)
        {
            leftChildrenIdx = rhs.leftChildrenIdx;
            isLeaf = rhs.isLeaf;
            numChildren = rhs.numChildren;

            sib[0] = rhs.sib[0];
            sib[1] = rhs.sib[1];
            sib[2] = rhs.sib[2];

            object_id = rhs.object_id;
            primid = rhs.primid;
            exid = rhs.exid;
            meshid = rhs.meshid;

            bminx = rhs.bminx;
            bmaxx = rhs.bmaxx;

            bminy = rhs.bminy;
            bmaxy = rhs.bmaxy;

            bminz = rhs.bminz;
            bmaxz = rhs.bmaxz;
        }
    };

    class transformable;

    class StacklessQbvh : public accelerator {
    public:
        StacklessQbvh() : accelerator(AccelType::StacklessQbvh) {}
        virtual ~StacklessQbvh() {}

    public:
        virtual void build(
            const context& ctxt,
            hitable** list,
            uint32_t num,
            aabb* bbox = nullptr) override final;

        virtual bool hit(
            const context& ctxt,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const override;

        virtual bool hit(
            const context& ctxt,
            const ray& r,
            real t_min, real t_max,
            bool enableLod,
            Intersection& isect) const override
        {
            return hit(ctxt, r, t_min, t_max, isect);
        }

        std::vector<std::vector<StacklessQbvhNode>>& getNodes()
        {
            return m_listQbvhNode;
        }
        std::vector<aten::mat4>& getMatrices()
        {
            return m_mtxs;
        }

    private:
        struct BvhNode {
            bvhnode* node;
            hitable* nestParent;
            aten::mat4 mtxL2W;

            BvhNode(bvhnode* n, hitable* p, const aten::mat4& m)
                : node(n), nestParent(p), mtxL2W(m)
            {}
        };

        void registerBvhNodeToLinearList(
            bvhnode* root,
            bvhnode* parentNode,
            hitable* nestParent,
            const aten::mat4& mtxL2W,
            std::vector<BvhNode>& listBvhNode,
            std::vector<accelerator*>& listBvh,
            std::map<hitable*, accelerator*>& nestedBvhMap);

        uint32_t convertFromBvh(
            const context& ctxt,
            bool isPrimitiveLeaf,
            std::vector<BvhNode>& listBvhNode,
            std::vector<StacklessQbvhNode>& listQbvhNode);

        void setQbvhNodeLeafParams(
            const context& ctxt,
            bool isPrimitiveLeaf,
            const BvhNode& bvhNode,
            StacklessQbvhNode& qbvhNode);

        void fillQbvhNode(
            StacklessQbvhNode& qbvhNode,
            std::vector<BvhNode>& listBvhNode,
            int32_t children[4],
            int32_t numChildren);

        int32_t getChildren(
            std::vector<BvhNode>& listBvhNode,
            int32_t bvhNodeIdx,
            int32_t children[4]);

        bool hit(
            const context& ctxt,
            int32_t exid,
            const std::vector<std::vector<StacklessQbvhNode>>& listQbvhNode,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const;

    private:
        bvh m_bvh;

        std::vector<std::vector<StacklessQbvhNode>> m_listQbvhNode;
        std::vector<aten::mat4> m_mtxs;
    };
}
