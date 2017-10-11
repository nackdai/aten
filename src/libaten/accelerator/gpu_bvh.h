#pragma once

#include "scene/hitable.h"
#include "accelerator/bvh.h"

namespace aten {
	struct GPUBvhNode {
		float hit{ -1 };		///< Link index if ray hit.
		float miss{ -1 };		///< Link index if ray miss.
		float parent{ -1 };		///< Parent node index.
		float padding0;

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

	// TODO
	// テスト用に bvh の継承クラスで作るが、インスタシエイトする必要がないので、あとで変更する.
	class GPUBvh : public bvh {
	public:
		GPUBvh() {}
		virtual ~GPUBvh() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num) override;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

	private:
		void registerBvhNodeToLinearList(
			bvhnode* root, 
			std::vector<bvhnode*>& listBvhNode);

		void registerGpuBvhNode(
			std::vector<bvhnode*>& listBvhNode, 
			std::vector<GPUBvhNode>& listGpuBvhNode);

		void setOrderForLinearBVH(
			std::vector<bvhnode*>& listBvhNode,
			std::vector<GPUBvhNode>& listGpuBvhNode);

		void dump(std::vector<GPUBvhNode>& nodes, const char* path);

	private:
		std::vector<std::vector<GPUBvhNode>> m_nodes;
	};
}
