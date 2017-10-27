#pragma once

#include "accelerator/bvh.h"

namespace aten {
	struct StacklessQbvhNode {
		float leftChildrenIdx{ -1 };
		struct {
			uint32_t isLeaf : 1;
			uint32_t numChildren : 3;
		};
		float parent{ -1 };
		float leftSiblingIdx{ -1 };

		float shapeid{ -1 };	///< Object index.
		float primid{ -1 };	///< Triangle index.
		float exid{ -1 };		///< External bvh index.
		float meshid{ -1 };	///< Mesh id.

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
		}

		StacklessQbvhNode(const StacklessQbvhNode& rhs)
		{
			leftChildrenIdx = rhs.leftChildrenIdx;
			isLeaf = rhs.isLeaf;
			numChildren = rhs.numChildren;
			leftSiblingIdx = rhs.leftSiblingIdx;

			shapeid = rhs.shapeid;
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
		StacklessQbvh() {}
		virtual ~StacklessQbvh() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num) override final;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

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
			std::map<hitable*, std::vector<accelerator*>>& nestedBvhMap);

		uint32_t convertFromBvh(
			bool isPrimitiveLeaf,
			std::vector<BvhNode>& listBvhNode,
			std::vector<StacklessQbvhNode>& listQbvhNode);

		void setQbvhNodeLeafParams(
			bool isPrimitiveLeaf,
			const BvhNode& bvhNode,
			StacklessQbvhNode& qbvhNode);

		void fillQbvhNode(
			StacklessQbvhNode& qbvhNode,
			std::vector<BvhNode>& listBvhNode,
			int children[4],
			int numChildren);

		int getChildren(
			std::vector<BvhNode>& listBvhNode,
			int bvhNodeIdx,
			int children[4]);

		bool hit(
			int exid,
			const std::vector<std::vector<StacklessQbvhNode>>& listQbvhNode,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const;

	private:
		bvh m_bvh;

		int m_exid{ 1 };

		std::vector<std::vector<StacklessQbvhNode>> m_listQbvhNode;
		std::vector<aten::mat4> m_mtxs;
	};
}
