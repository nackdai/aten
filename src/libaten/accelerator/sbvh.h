#pragma once

#include "accelerator/bvh.h"

namespace aten {
	class sbvh : public accelerator {
	public:
		sbvh() {}
		virtual ~sbvh() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num,
			aabb* bbox = nullptr) override final;

	private:
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

			bool leaf{ true };
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

			Reference(uint32_t id) : triid(id) {}

			// 分割元の三角形インデックス.
			uint32_t triid;

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

	private:
		bvh m_bvh;

		uint32_t m_numBins{ 16 };
		uint32_t m_maxTriangles{ 4 };

		std::vector<SBVHNode> m_nodes;

		std::vector<Reference> m_refs;
	};
}
