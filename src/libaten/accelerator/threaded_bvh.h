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
        aten::vec3 boxmin;      ///< AABB min position.
        float hit{ -1 };        ///< Link index if ray hit.

        aten::vec3 boxmax;      ///< AABB max position.
        float miss{ -1 };       ///< Link index if ray miss.

        // TODO:
        // ThreadedBvhNode is converted from/to ThreadedSbvhNode.
        // It's too bad.
        // To keep the current situation temporarily, the following variables must be kept.
        // The order of the variabled must be kept as well.

        float object_id{ -1 };  ///< Object index.
        float primid{ -1 };     ///< Triangle index.

        union ExternalBvhIdxFlag {
            float exid{ -1 };   ///< External bvh index.
            struct {
                uint32_t mainExid : 15;     ///< External bvh index.
                uint32_t lodExid : 15;      ///< LOD bvh index.
                uint32_t hasLod : 1;        ///< Flag if the node has LOD.
                uint32_t noExternal : 1;    ///< Flag if the node does not have external bvh.
            };
        } ex_bvh;

        float meshid{ -1 };     ///< Mesh id.

        static AT_DEVICE_API bool isLeaf(const ThreadedBvhNode& node)
        {
            return (node.object_id >= 0 || node.primid >= 0);
        }

        static float ConstructExternalBvhIdxFlag(const int32_t exid, const int32_t subexid)
        {
            ExternalBvhIdxFlag flag;
            flag.noExternal = (exid < 0);
            flag.hasLod = (subexid >= 0);
            flag.mainExid = exid;
            flag.lodExid = (subexid >= 0 ? subexid : 0);
            return flag.exid;
        }

    };

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
            return hit(ctxt, 0, m_listThreadedBvhNode, r, t_min, t_max, isect, hit_stop_type);
        }

        /**
         * @brief Draw all node's AABB in the structure tree.
         */
        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtx_L2W) override final
        {
            m_bvh.drawAABB(func, mtx_L2W);
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

        static void dump(std::vector<ThreadedBvhNode>& nodes, std::string_view path);

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
            aten::mat4 mtx_L2W;
        };

        /**
         * @brief Convert the temporary description for the node of threaded bvh to final description.
         */
        void registerThreadedBvhNode(
            const context& ctxt,
            bool isPrimitiveLeaf,
            const std::vector<ThreadedBvhNodeEntry>& listBvhNode,
            std::vector<ThreadedBvhNode>& listThreadedBvhNode,
            std::vector<int32_t>& listParentId);

        /**
         * @brief Set hit/miss traverse order.
         */
        void setOrder(
            const std::vector<ThreadedBvhNodeEntry>& listBvhNode,
            const std::vector<int32_t>& listParentId,
            std::vector<ThreadedBvhNode>& listThreadedBvhNode);

        /**
         * @brief Test if a ray hits a object.
         */
        bool hit(
            const context& ctxt,
            int32_t exid,
            const std::vector<std::vector<ThreadedBvhNode>>& listThreadedBvhNode,
            const ray& r,
            float t_min, float t_max,
            Intersection& isect,
            aten::HitStopType hit_stop_type = aten::HitStopType::Closest) const;

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

        // Linear list to register bvh nodes.
        std::vector<std::vector<ThreadedBvhNode>> m_listThreadedBvhNode;

        // List for bottom layer.
        std::vector<accelerator*> m_nestedBvh;

        // Map to register the pair for bottom layer and its id.
        std::map<int32_t, accelerator*> m_mapNestedBvh;
    };
}
