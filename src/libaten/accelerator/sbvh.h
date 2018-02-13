#pragma once

#include "accelerator/threaded_bvh.h"

#define SBVH_TRIANGLE_NUM	(1)

#define VOXEL_TEST

namespace aten
{
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

		float isleaf{ -1 };
		float triid{ -1 };
		float padding{ -1 };

#ifdef VOXEL_TEST
		// Offset to voxel from current node index.
		float voxel{ -1 };
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

	struct SbvhTreelet {
		uint32_t idxInBvhTree;
		uint32_t depth;
		std::vector<uint32_t> leafChildren;
		std::vector<uint32_t> tris;
	};

	struct BvhVoxel {
		aten::vec3 nml;
		uint32_t nodeid{ 0 };

		aten::vec3 clr;
		struct {
			uint32_t exid : 16;
			uint32_t lod : 16;
		};		

		float radius;
		float area;
		float reserved[2];
	};

	// NOTE
	// GPGPU処理用に両方を同じメモリ空間上に格納するため、同じサイズでないといけない.
	AT_STATICASSERT(sizeof(ThreadedSbvhNode) == sizeof(ThreadedBvhNode));
	AT_STATICASSERT(sizeof(BvhVoxel) == sizeof(ThreadedBvhNode));

	class sbvh : public accelerator {
	public:
		sbvh() : accelerator(AccelType::Sbvh) {}
		virtual ~sbvh() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num,
			aabb* bbox = nullptr) override final;

		virtual void buildVoxel(
			uint32_t exid,
			uint32_t offset) override final;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect,
			bool enableLod) const override;

		virtual bool exportTree(const char* path) override final;
		virtual bool importTree(const char* path, int offsetTriIdx) override final;

		ThreadedBVH& getTopLayer()
		{
			return m_bvh;
		}

		virtual void drawAABB(
			aten::hitable::FuncDrawAABB func,
			const aten::mat4& mtxL2W) override final;


		virtual void update() override final;

		const std::vector<std::vector<ThreadedSbvhNode>>& getNodes() const
		{
			return m_threadedNodes;
		}
		const std::vector<aten::mat4>& getMatrices() const
		{
			return m_bvh.getMatrices();
		}
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
		void buildInternal(
			hitable** list,
			uint32_t num);

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

		void findObjectSplit(
			SBVHNode& node,
			real& cost,
			aabb& leftBB,
			aabb& rightBB,
			int& splitBinPos,
			int& axis);

		void findSpatialSplit(
			SBVHNode& node,
			real& cost,
			int& leftCount,
			int& rightCount,
			aabb& leftBB,
			aabb& rightBB,
			int& bestAxis,
			real& splitPlane);

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

		void objectSort(
			SBVHNode& node,
			int splitBin,
			int axis,
			std::vector<uint32_t>& leftList,
			std::vector<uint32_t>& rightList);

		void getOrderIndex(std::vector<int>& indices) const;

		void makeTreelet();
		void onMakeTreelet(
			uint32_t idx,
			const sbvh::SBVHNode& root);

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
