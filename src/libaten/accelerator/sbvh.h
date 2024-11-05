#pragma once

#include "accelerator/threaded_bvh.h"

#define SBVH_TRIANGLE_NUM    (1)

namespace aten
{
    class context;

#define AT_DISABLE_VOXEL    (-1.0f)
#define AT_IS_VOXEL(v)    (v < AT_DISABLE_VOXEL)
#define AT_SET_VOXEL_DETPH(d)    (-(d) + AT_DISABLE_VOXEL)
#define AT_GET_VOXEL_DEPTH(d)    (aten::abs((d)) + AT_DISABLE_VOXEL)

    /**
     * @brief Description for the node in SBVH.
     */
    struct ThreadedSbvhNode {
        aten::vec3 boxmin;        ///< AABB min position.
        float hit{ -1 };        ///< Link index if ray hit.

        aten::vec3 boxmax;        ///< AABB max position.
        float miss{ -1 };        ///< Link index if ray miss.

#if (SBVH_TRIANGLE_NUM == 1)
        // NOTE
        // triidの位置をThreadedBvhNodeと合わせる.

        // NOTE
        // ThreadedBvhNode では isleaf の位置に object_id がいてGPUでは object_id を見てリーフノードかどうか判定している.
        // そのため、最初のfloatでリーフノードかどうかを判定するようにする.
        // isVoxel の部分は ThreadedBvhNode では exid なので、ここは常にマイナスになるようにする.
        // ただし、マイナスでありさえすればいいので、ここでは -1 より小さい数を判定の閾値にしつつ、depth値として扱う.

        float isleaf{ -1 };        ///< Flag if the node is leaf.
        float triid{ -1 };        ///< Index of the triangle.

        float voxeldepth{ AT_DISABLE_VOXEL };    ///< If hasVoxel < -1, the node is used as voxel.

        float mtrlid{ -1 };

        bool isLeaf() const
        {
            return (isleaf >= 0);
        }
#else
        float refIdListStart{ -1.0f };
        float refIdListEnd{ -1.0f };
        float parent{ -1.0f };
        float padding;

        bool isLeaf() const
        {
            return refIdListStart >= 0;
        }
#endif
    };

    // NOTE
    // GPGPU処理用に両方を同じメモリ空間上に格納するため、同じサイズでないといけない.
    AT_STATICASSERT(sizeof(ThreadedSbvhNode) == sizeof(ThreadedBvhNode));

    /**
     * @brief Spatial Splits in Bounding Volume Hierarchies.
     */
    class sbvh : public accelerator {
    public:
        // TODO
        static const int32_t VoxelDepth = 3;

    public:
        sbvh() : accelerator(AccelType::Sbvh) {}
        virtual ~sbvh() {}

    public:
        /**
         * @brief Bulid structure tree from the specified list.
         */
        virtual void build(
            const context& ctxt,
            hitable** list,
            uint32_t num,
            aabb* bbox) override final;

        /**
         * @brief Build voxels from the specified tree.
         */
        virtual void buildVoxel(const context& ctxt) override final;

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
            aten::HitStopType hit_stop_type = aten::HitStopType::Closest) const override;

        /**
         * @brief Export the built structure data.
         */
        virtual bool exportTree(
            const context& ctxt,
            std::string_view path) override final;

        /**
         * @brief Import the exported structure data.
         */
        virtual bool importTree(
            const context& ctxt,
            std::string_view path,
            int32_t offsetTriIdx) override final;

        /**
         * @brief Return the top layer acceleration structure.
         */
        ThreadedBVH& getTopLayer()
        {
            return m_bvh;
        }

        /**
         * @brief Draw all node's AABB in the structure tree.
         */
        virtual void drawAABB(
            aten::hitable::FuncDrawAABB func,
            const aten::mat4& mtx_L2W) override final;

        bool IsBuilt() const override
        {
            return !m_threadedNodes.empty() && !m_threadedNodes[0].empty();
        }

        std::optional<aten::aabb> GetBoundingBox() const override
        {
            if (IsBuilt()) {
                const auto& root = m_threadedNodes[0][0];
                return aten::aabb(root.boxmin, root.boxmax);
            }
            return std::nullopt;
        }

        /**
         * @brief Update the structure tree.
         */
        virtual void update(const context& ctxt) override final;

        /**
         * @brief Return all nodes.
         */
        const std::vector<std::vector<ThreadedSbvhNode>>& getNodes() const
        {
            return m_threadedNodes;
        }

        uint32_t getMaxDepth() const
        {
            return m_maxDepth;
        }

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
         * @brief Build the tree.
         */
        void onBuild(
            const context& ctxt,
            hitable** list,
            uint32_t num);

        /**
         * @brief Convert temporary description of sbvh node to final description of sbvh node.
         */
        void convert(
            std::vector<ThreadedSbvhNode>& nodes,
            int32_t offset,
            std::vector<int32_t>& indices) const;

        bool hit(
            const context& ctxt,
            int32_t exid,
            const ray& r,
            float t_min, float t_max,
            Intersection& isect,
            bool enableLod) const;

        /**
         * @brief Temporary description of sbvh node.
         */
        struct SBVHNode {
            SBVHNode() {}

            SBVHNode(const std::vector<uint32_t>&& indices, const aabb& box)
                : refIds(indices), bbox(box)
            {}

            bool isLeaf() const
            {
                return leaf;
            }

            void setChild(int32_t leftId, int32_t rightId)
            {
                leaf = false;
                left = leftId;
                right = rightId;
            }

            aabb bbox;

            // Indices for triangls which this node has.
            std::vector<uint32_t> refIds;

            // Child left;
            int32_t left{ -1 };

            // Child right;
            int32_t right{ -1 };

            int32_t parent{ -1 };
            int32_t depth{ 0 };

            bool leaf{ true };
            bool isTreeletRoot{ false };
        };

        // 分割情報.
        struct Bin {
            Bin() {}

            // bbox of bin.
            aabb bbox;

            // accumulated bbox of bin left/right.
            aabb accum;

            // references starting here.
            int32_t start{ 0 };

            // references ending here.
            int32_t end{ 0 };
        };

        // 分割三角形情報.
        struct Reference {
            Reference() = default;

            Reference(int32_t id) : triid(id) {}

            // 分割元の三角形インデックス.
            int32_t triid{ -1 };

            // 分割した後のAABB.
            aabb bbox;
        };

        /**
         * @brief Find a potential object split, but it does not split.
         * ざっくり分割テスト.
         * @param [in, out] node The node which we want to split.
         * @param [out] cost Cost to split.
         * @param [out] leftBB AABB of the potential left child nodes.
         * @param [out] rightBB AABB of the potential right child nodes.
         * @param [out] splitBinPos Index of the bin to split.
         * @param [out] axis Axis (xyz) along which the split was done.
         */
        void findObjectSplit(
            SBVHNode& node,
            float& cost,
            aabb& leftBB,
            aabb& rightBB,
            int32_t& splitBinPos,
            int32_t& axis);

        /**
         * @brief Find a potential spatial split, but it does not split.
         * より詳細分割テスト.
         * @param [in, out] node The node which we want to split.
         * @param [out] leftCount Count of triangles in the left child nodes.
         * @param [out] rightCount Count of triangles in the right child nodes.
         * @param [out] leftBB AABB of the potential left child nodes.
         * @param [out] rightBB AABB of the potential right child nodes.
         * @param [out] bestAxis Axis (xyz) along which the split was done.
         * @param [out] splitPlane Position of the axis along which the split was done.
        */
        void findSpatialSplit(
            SBVHNode& node,
            float& cost,
            int32_t& leftCount,
            int32_t& rightCount,
            aabb& leftBB,
            aabb& rightBB,
            int32_t& bestAxis,
            float& splitPlane);

        /**
         * @biref Do the binned sah split actually with the result of findSpatialSplit.
         * より詳細分割テストに基づいて実際に分割する.
         * @param[in, out] node The node which we want to split.
         * @param[in] splitPlane Position of the axis along which the split will run.
         * @param[in] axis Axis (xyz) along which the split will run.
         * @param[in] splitCost Cost to split.
         * @param[in] leftCount Count of triangles in the left child nodes.
         * @param[in] rightCount Count of triangles in the right child nodes.
         * @param[out] leftBB AABB of the potential left child nodes.
         * @param[out] rightBB AABB of the potential right child nodes.
         * @param[out] leftList Triangle indices list for the left children.
         * @param[out] rightList Triangle indices list for the right children.
         */
        void spatialSort(
            SBVHNode& node,
            float splitPlane,
            int32_t axis,
            float splitCost,
            int32_t leftCnt,
            int32_t rightCnt,
            aabb& leftBB,
            aabb& rightBB,
            std::vector<uint32_t>& leftList,
            std::vector<uint32_t>& rightList);

        /**
         * @biref Do the binned sah split actually with the result of findObjectSplit.
         * ざっくり分割テストに基づいて実際に分割する.
         * @param [in, out] node The node which we want to split.
         * @param [in] Index of the bin to split.
         * @param [in] axis Axis (xyz) along which the split will run.
         * @param [out] leftList Triangle indices list for the left children.
         * @param [out] rightList Triangle indices list for the right children.
         */
        void objectSort(
            SBVHNode& node,
            int32_t splitBin,
            int32_t axis,
            std::vector<uint32_t>& leftList,
            std::vector<uint32_t>& rightList);

        /**
         * @brief Convert the tree to the linear list.
         * @param [out] indices Node indices list.
         */
        void getOrderIndex(std::vector<int32_t>& indices) const;

        /**
         * @brief Make treelets from the tree.
         */
        void makeTreelet();

        /**
         * @brief Make treelet from the specified treelet root node.
         */
        void onMakeTreelet(
            uint32_t idx,
            const sbvh::SBVHNode& root);

    private:
        ThreadedBVH m_bvh;

        // 分割最大数.
        uint32_t m_numBins{ 16 };

        // ノード当たりの最大三角形数.
        uint32_t m_maxTriangles{ SBVH_TRIANGLE_NUM };

        uint32_t m_refIndexNum{ 0 };

        int32_t m_offsetTriIdx{ 0 };

        std::vector<SBVHNode> m_nodes;

        // 三角形情報リスト.
        // ここでいう三角形情報とは分割された or されていない三角形の情報.
        std::vector<Reference> m_refs;

        // For layer.
        std::vector<std::vector<ThreadedSbvhNode>> m_threadedNodes;
        std::vector<int32_t> m_refIndices;

        uint32_t m_maxDepth{ 0 };

        // Description for the treelet root.
        struct SbvhTreelet {
            // Node index in the tree.
            uint32_t idxInBvhTree;

            // Flag if the treelet is enabled.
            // If a treelet has a child which is light, it is disabled.
            bool enabled{ true };

            int32_t mtrlid{ -1 };

            // List of leaf children in the treelet.
            std::vector<uint32_t> leafChildren;

            // Triangle indices in the treelet.
            std::vector<uint32_t> tris;
        };

        // For voxelize.
        //  key : index in nodes.
        //  value : treelet.
        std::map<uint32_t, SbvhTreelet> m_treelets;

        // Flag if sbvh is imported from file.
        bool m_isImported{ false };
    };
}
