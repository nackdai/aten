#pragma once

#include <map>

#include "scene/hitable.h"
#include "accelerator/bvh.h"

namespace aten
{
    /**
     * @brief Description for the node in threaded BVH.
     */
    struct ThreadedBvhNode {
        aten::vec3 boxmin;        ///< AABB min position.
        float hit{ -1 };        ///< Link index if ray hit.

        aten::vec3 boxmax;        ///< AABB max position.
        float miss{ -1 };        ///< Link index if ray miss.

        float shapeid{ -1 };    ///< Object index.
        float primid{ -1 };        ///< Triangle index.

        ///< External bvh index.
        union {
            float exid{ -1 };
            struct {
                uint32_t mainExid : 15;        ///< External bvh index.
                uint32_t lodExid : 15;        ///< LOD bvh index.
                uint32_t hasLod : 1;        ///< Flag if the node has LOD.
                uint32_t noExternal : 1;    ///< Flag if the node does not have external bvh.
            };
        };

        float meshid{ -1 };        ///< Mesh id.

        bool isLeaf() const
        {
            return (shapeid >= 0 || primid >= 0);
        }
    };

#define AT_BVHNODE_HAS_EXTERNAL(n)    (((n) & (1 << 31)) == 0)
#define AT_BVHNODE_HAS_LOD(n)    (((n) & (1 << 30)) > 0)
#define AT_BVHNODE_MAIN_EXID(n)    ((n) & 0x7fff)
#define AT_BVHNODE_LOD_EXID(n)    (((n) & (0x7fff << 15)) >> 15)

    /**
     * @brief Threaded Boundinf Volume Hierarchies.
     */
    class ThreadedBVH : public accelerator {
    public:
        ThreadedBVH() : accelerator(AccelType::ThreadedBvh) {}
        virtual ~ThreadedBVH() {}

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
         * @brief Draw all node's AABB in the structure tree.
         */
        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtxL2W) override final
        {
            m_bvh.drawAABB(func, mtxL2W);
        }

        /**
         * @brief Update the structure tree.
         */
        virtual void update(const context& ctxt) override;

        /**
         * @brief Return all nodes.
         */
        std::vector<std::vector<ThreadedBvhNode>>& getNodes()
        {
            return m_listThreadedBvhNode;
        }

        /**
         * @brief Return all nodes.
         */
        const std::vector<std::vector<ThreadedBvhNode>>& getNodes() const
        {
            return m_listThreadedBvhNode;
        }

        /**
         * @brief Return all matrices to transform the node.
         */
        const std::vector<aten::mat4>& getMatrices() const
        {
            return m_mtxs;
        }

        /**
         * @brief Tell not to build bottom layer.
         */
        void disableLayer()
        {
            m_enableLayer = false;
        }

        /**
         * @brief Return bottom layer list.
         */
        const std::vector<accelerator*>& getNestedAccel()
        {
            return m_nestedBvh;
        }

        static void dump(std::vector<ThreadedBvhNode>& nodes, const char* path);

    private:
        /**
         * @brief Build the tree for the bottom layer.
         */
        void buildAsNestedTree(
            const context& ctxt,
            hitable** list,
            uint32_t num,
            aabb* bbox);

        /**
         * @brief Build the tree for the top layer.
         */
        void buildAsTopLayerTree(
            const context& ctxt,
            hitable** list,
            uint32_t num,
            aabb* bbox);

        /**
         * @brief Temporary description for the node of threaded bvh.
         */
        struct ThreadedBvhNodeEntry {
            bvhnode* node;
            aten::mat4 mtxL2W;

            ThreadedBvhNodeEntry(bvhnode* n, const aten::mat4& m)
                : node(n), mtxL2W(m)
            {}
        };

        /**
         * @brief Convert the temporary description for the node of threaded bvh to final description.
         */
        void registerThreadedBvhNode(
            const context& ctxt,
            bool isPrimitiveLeaf,
            const std::vector<ThreadedBvhNodeEntry>& listBvhNode,
            std::vector<ThreadedBvhNode>& listThreadedBvhNode,
            std::vector<int>& listParentId);

        /**
         * @brief Set hit/miss traverse order.
         */
        void setOrder(
            const std::vector<ThreadedBvhNodeEntry>& listBvhNode,
            const std::vector<int>& listParentId,
            std::vector<ThreadedBvhNode>& listThreadedBvhNode);

        /**
         * @brief Test if a ray hits a object.
         */
        bool hit(
            const context& ctxt,
            int exid,
            const std::vector<std::vector<ThreadedBvhNode>>& listThreadedBvhNode,
            const ray& r,
            real t_min, real t_max,
            Intersection& isect) const;

        /**
         * @brief Convert the tree to the linear list.
         */
        void registerBvhNodeToLinearList(
            const context& ctxt,
            bvhnode* node,
            std::vector<ThreadedBvhNodeEntry>& nodes);

    private:
        bvh m_bvh;

        // Flag whether thereaded bvh will build bottom layer.
        bool m_enableLayer{ true };

        std::vector<std::vector<ThreadedBvhNode>> m_listThreadedBvhNode;
        std::vector<aten::mat4> m_mtxs;

        // List for bottom layer.
        std::vector<accelerator*> m_nestedBvh;

        std::map<int, accelerator*> m_mapNestedBvh;
    };
}
