#pragma once

#include "accelerator/threaded_bvh.h"

#define SBVH_TRIANGLE_NUM	(1)

#define VOXEL_TEST

namespace aten
{
	/**
	 * @brief Description for the node in SBVH.
	 */
	struct ThreadedSbvhNode {
		aten::vec3 boxmin;		///< AABB min position.
		float hit{ -1 };		///< Link index if ray hit.

		aten::vec3 boxmax;		///< AABB max position.
		float miss{ -1 };		///< Link index if ray miss.

#if (SBVH_TRIANGLE_NUM == 1)
		// NOTE
		// triidの位置をThreadedBvhNodeと合わせる.

		// NOTE
		// ThreadedBvhNode では isleaf の位置に shapeid がいてGPUでは shapeid を見てリーフノードかどうか判定している.
		// そのため、最初のfloatでリーフノードかどうかを判定するようにする.
		// padding の部分は ThreadedBvhNode では exid なので、ここは常に -1 になるようにする.

		float isleaf{ -1 };		///< Flag if the node is leaf.
		float triid{ -1 };		///< Index of the triangle.
		float padding{ -1 };

#ifdef VOXEL_TEST
		float voxel{ -1 };		///< Offset to voxel from current node index.
#else
		float parent{ -1 };
#endif

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

	/**
	 * @brief Description for the treelet root.
	 */
	struct SbvhTreelet {
		uint32_t idxInBvhTree;	///< Node index in the tree.
		uint32_t depth;			///< Depth in the tree.
		std::vector<uint32_t> leafChildren;	///< List of leaf children in the treelet.
		std::vector<uint32_t> tris;			///< Triangle indices in the treelet.
	};

	/**
	 * @brief Desctiption for the voxel in the tree.
	 */
	struct BvhVoxel {
		aten::vec3 nml;			///< Averaged normal of the voxel.
		uint32_t nodeid{ 0 };	///< Node index in the tree.

		aten::vec3 clr;			///< Averaged color of the voxel.
		struct {
			uint32_t exid : 16;	///< Index of an external tree.
			uint32_t lod : 16;	///< LOD level.
		};		

		float radius;	///< Radius of the voxel.
		float area;		///< Area of the voxel.
		float reserved[2];
	};

	// NOTE
	// GPGPU処理用に両方を同じメモリ空間上に格納するため、同じサイズでないといけない.
	AT_STATICASSERT(sizeof(ThreadedSbvhNode) == sizeof(ThreadedBvhNode));
	AT_STATICASSERT(sizeof(BvhVoxel) == sizeof(ThreadedBvhNode));

	/**
	 * @brief Spatial Splits in Bounding Volume Hierarchies.
	 */
	class sbvh : public accelerator {
	public:
		sbvh() : accelerator(AccelType::Sbvh) {}
		virtual ~sbvh() {}

	public:
		/**
		 * @brief Bulid structure tree from the specified list.
		 */
		virtual void build(
			hitable** list,
			uint32_t num,
			aabb* bbox = nullptr) override final;

		/**
		 * @brief Build voxels from the specified tree.
		 */
		virtual void buildVoxel(
			uint32_t exid,
			uint32_t offset) override final;

		/**
		 * @brief Test if a ray hits a object.
		 */
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

		/**
		 * @brief Test if a ray hits a object.
		 */
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect,
			bool enableLod) const override;

		/**
		 * @brief Export the built structure data.
		 */
		virtual bool exportTree(const char* path) override final;

		/**
		 * @brief Import the exported structure data.
		 */
		virtual bool importTree(const char* path, int offsetTriIdx) override final;

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
			const aten::mat4& mtxL2W) override final;

		/**
		 * @brief Update the structure tree.
		 */
		virtual void update() override final;

		/**
		 * @brief Return all nodes.
		 */
		const std::vector<std::vector<ThreadedSbvhNode>>& getNodes() const
		{
			return m_threadedNodes;
		}
		
		/**
		 * @brief Return all matrices to transform the node.
		 */
		const std::vector<aten::mat4>& getMatrices() const
		{
			return m_bvh.getMatrices();
		}

		/**
		 * @brief Return all voxels.
		 */
		const std::vector<BvhVoxel>& getVoxels() const
		{
			return m_voxels;
		}

		real computeVoxelLodErrorMetric(
			uint32_t height, 
			real verticalFov, 
			real pixelOfError = real(-1));

		real getVoxelLodErrorMetric(real err);

		void setPixelOfError(real pixelOfError);

		void setVoxelLodErrorMetricMultiplyer(real multiplyer);

	private:
		/**
		 * @brief Build the tree for the bottom layer.
		 */
		void buildAsNestedTree(
			hitable** list,
			uint32_t num,
			aabb* bbox);

		/**
		 * @brief Build the tree for the top layer.
		 */
		void buildAsTopLayerTree(
			hitable** list,
			uint32_t num,
			aabb* bbox);

		/**
		 * @brief Build the tree.
		 */
		void onBuild(
			hitable** list,
			uint32_t num);

		/**
		 * @brief Convert temporary description of sbvh node to final description of sbvh node.
		 */
		void convert(
			std::vector<ThreadedSbvhNode>& nodes,
			int offset,
			std::vector<int>& indices) const;

		bool hit(
			int exid,
			const ray& r,
			real t_min, real t_max,
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

			void setChild(int leftId, int rightId)
			{
				leaf = false;
				left = leftId;
				right = rightId;
			}

			aabb bbox;

			// Indices for triangls which this node has.
			std::vector<uint32_t> refIds;

			// Child left;
			int left{ -1 };

			// Child right;
			int right{ -1 };

			int parent{ -1 };
			uint32_t depth{ 0 };

			int voxelIdx{ -1 };

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
			int start{ 0 };

			// references ending here.
			int end{ 0 };
		};

		// 分割三角形情報.
		struct Reference {
			Reference() {}

			Reference(int id) : triid(id) {}

			// 分割元の三角形インデックス.
			int triid;

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
			real& cost,
			aabb& leftBB,
			aabb& rightBB,
			int& splitBinPos,
			int& axis);

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
			real& cost,
			int& leftCount,
			int& rightCount,
			aabb& leftBB,
			aabb& rightBB,
			int& bestAxis,
			real& splitPlane);

		/**
		 * @biref Do the binned sah split actually with the result of findSpatialSplit.
		 * より詳細分割テストに基づいて実際に分割する.
		 * @param [in, out] node The node which we want to split.
		 * @param [in] splitPlane Position of the axis along which the split will run.
		 * @param [in] axis Axis (xyz) along which the split will run.
		 * @param [in] splitCost Cost to split.
		 * @param [in] leftCount Count of triangles in the left child nodes.
		 * @param [in] rightCount Count of triangles in the right child nodes.
		 * @param [out] leftBB AABB of the potential left child nodes.
		 * @param [out] rightBB AABB of the potential right child nodes.
		 * @param [out] leftList Triangle indices list for the left children.
		 * @param [out] rightList Triangle indices list for the right children.
		 */
		void spatialSort(
			SBVHNode& node,
			real splitPlane,
			int axis,
			real splitCost,
			int leftCnt,
			int rightCnt,
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
			int splitBin,
			int axis,
			std::vector<uint32_t>& leftList,
			std::vector<uint32_t>& rightList);

		/**
		 * @brief Convert the tree to the linear list.
		 * @param [out] indices Node indices list.
		 */
		void getOrderIndex(std::vector<int>& indices) const;

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

		/**
		 * @brief Copy voxel data to the node array.
		 * Voxelデータをノードの配列に入れるために、強制的にコピーを行う.
		 */
		void copyVoxelToNodeArray(std::vector<ThreadedSbvhNode>& dst);

	private:
		ThreadedBVH m_bvh;

		// 分割最大数.
		uint32_t m_numBins{ 16 };

		// ノード当たりの最大三角形数.
		uint32_t m_maxTriangles{ SBVH_TRIANGLE_NUM };

		uint32_t m_refIndexNum{ 0 };

		int m_offsetTriIdx{ 0 };

		std::vector<SBVHNode> m_nodes;

		// 三角形情報リスト.
		// ここでいう三角形情報とは分割された or されていない三角形の情報.
		std::vector<Reference> m_refs;

		// For layer.
		std::vector<std::vector<ThreadedSbvhNode>> m_threadedNodes;
		std::vector<int> m_refIndices;

		uint32_t m_maxDepth{ 0 };

		// For voxelize.
		float m_maxVoxelRadius{ 0.0f };
		std::vector<SbvhTreelet> m_treelets;
		std::vector<BvhVoxel> m_voxels;

		// For voxel hit test.
		real m_voxelLodErrorMetric{ real(1) };
		real m_voxelLodErrorMetricMultiplyer{ real(1) };
	};
}
