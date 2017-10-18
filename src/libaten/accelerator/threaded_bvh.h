#pragma once

#include "scene/hitable.h"
#include "accelerator/bvh.h"

namespace aten {
	struct ThreadedBvhNode {
		float hit{ -1 };		///< Link index if ray hit.
		float miss{ -1 };		///< Link index if ray miss.
		float parent{ -1 };		///< Parent node index.
		float padding0{ 0 };

		float shapeid{ -1 };	///< Object index.
		float primid{ -1 };		///< Triangle index.
		float exid{ -1 };		///< External bvh index.
		float meshid{ -1 };		///< Mesh id.

		aten::vec4 boxmin;		///< AABB min position.
		aten::vec4 boxmax;		///< AABB max position.

		bool isLeaf() const
		{
			return (shapeid >= 0 || primid >= 0);
		}
	};

	class ThreadedBVH : public accelerator {
	public:
		ThreadedBVH() {}
		virtual ~ThreadedBVH() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num) override;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

		std::vector<std::vector<ThreadedBvhNode>>& getNodes()
		{
			return m_listThreadedBvhNode;
		}
		std::vector<aten::mat4>& getMatrices()
		{
			return m_mtxs;
		}

		static void dump(std::vector<ThreadedBvhNode>& nodes, const char* path);

	private:
		struct ThreadedBvhNodeEntry {
			bvhnode* node;
			hitable* nestParent;
			aten::mat4 mtxL2W;

			ThreadedBvhNodeEntry(bvhnode* n, hitable* p, const aten::mat4& m)
				: node(n), nestParent(p), mtxL2W(m)
			{}
		};

		void registerBvhNodeToLinearList(
			bvhnode* root, 
			bvhnode* parentNode,
			hitable* nestParent,
			const aten::mat4& mtxL2W,
			std::vector<ThreadedBvhNodeEntry>& listBvhNode,
			std::vector<accelerator*>& listBvh,
			std::map<hitable*, std::vector<accelerator*>>& nestedBvhMap);

		void registerGpuBvhNode(
			bool isPrimitiveLeaf,
			std::vector<ThreadedBvhNodeEntry>& listBvhNode,
			std::vector<ThreadedBvhNode>& listGpuBvhNode);

		void setOrderForLinearBVH(
			std::vector<ThreadedBvhNodeEntry>& listBvhNode,
			std::vector<ThreadedBvhNode>& listGpuBvhNode);

		bool hit(
			int exid,
			const std::vector<std::vector<ThreadedBvhNode>>& listGpuBvhNode,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const;

	private:
		bvh m_bvh;

		int m_exid{ 1 };

		std::vector<std::vector<ThreadedBvhNode>> m_listThreadedBvhNode;
		std::vector<aten::mat4> m_mtxs;
	};
}
