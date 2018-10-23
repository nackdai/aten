#pragma once

#include "accelerator/bvh.h"
#include "scene/context.h"

namespace aten {
    struct QbvhNode {
        union {
            aten::vec4 p0;
            struct {
                float leftChildrenIdx;
                float isLeaf;
                float numChildren;
                float padding;
            };
        };

        union {
            aten::vec4 p1;
            struct {
                float shapeid;    ///< Object index.
                float primid;    ///< Triangle index.
                float exid;        ///< External bvh index.
                float meshid;    ///< Mesh id.
            };
        };

        union {
            aten::vec4 bminx;
            aten::vec4 v0x;
        };

        union {
            aten::vec4 bmaxx;
            aten::vec4 v0y;
        };

        union {
            aten::vec4 bminy;
            aten::vec4 v0z;
        };

        union {
            aten::vec4 bmaxy;
            aten::vec4 e1x;
        };

        union {
            aten::vec4 bminz;
            aten::vec4 e1y;
        };

        union {
            aten::vec4 bmaxz;
            aten::vec4 e1z;
        };

#ifdef ENABLE_BVH_MULTI_TRIANGLES
        union {
            aten::vec4 p2;
            struct {
                float shapeidx[4];
            };
        };
        union {
            aten::vec4 p3;
            struct {
                float primidx[4];
            };
        };
#endif

        QbvhNode()
        {
            isLeaf = false;
            numChildren = 0;

            shapeid = -1;
            primid = -1;
            meshid = -1;
            exid = -1;
        }
        QbvhNode(const QbvhNode& rhs)
        {
            p0 = rhs.p0;
            p1 = rhs.p1;

#ifdef ENABLE_BVH_MULTI_TRIANGLES
            p2 = rhs.p2;
            p3 = rhs.p3;
#endif

            bminx = rhs.bminx;
            bmaxx = rhs.bmaxx;

            bminy = rhs.bminy;
            bmaxy = rhs.bmaxy;

            bminz = rhs.bminz;
            bmaxz = rhs.bmaxz;
        }
    };

    class transformable;

    class qbvh : public accelerator {
    public:
        qbvh() : accelerator(AccelType::Qbvh) {}
        virtual ~qbvh() {}

    public:
        virtual void build(
            const context& ctxt,
            hitable** list,
            uint32_t num,
            aabb* bbox) override final;

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

        std::vector<std::vector<QbvhNode>>& getNodes()
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
            std::vector<QbvhNode>& listQbvhNode);

        void setQbvhNodeLeafParams(
            const context& ctxt,
            bool isPrimitiveLeaf,
            const BvhNode& bvhNode,
            QbvhNode& qbvhNode);

        void fillQbvhNode(
            QbvhNode& qbvhNode,
            std::vector<BvhNode>& listBvhNode,
            int children[4],
            int numChildren);

        int getChildren(
            std::vector<BvhNode>& listBvhNode,
            int bvhNodeIdx,
            int children[4]);

        bool hit(
            const context& ctxt,
            int exid,
            const std::vector<std::vector<QbvhNode>>& listQbvhNode,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const;

    private:
        bvh m_bvh;

        std::vector<std::vector<QbvhNode>> m_listQbvhNode;
        std::vector<aten::mat4> m_mtxs;
    };
}
