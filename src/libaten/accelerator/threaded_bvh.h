#pragma once

#include "scene/hitable.h"
#include "accelerator/bvh.h"

namespace aten {
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

		std::vector<std::vector<GPUBvhNode>>& getNodes()
		{
			return m_listGpuBvhNode;
		}
		std::vector<aten::mat4>& getMatrices()
		{
			return m_mtxs;
		}

		static void dump(std::vector<GPUBvhNode>& nodes, const char* path);

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
			std::vector<GPUBvhNode>& listGpuBvhNode);

		void setOrderForLinearBVH(
			std::vector<ThreadedBvhNodeEntry>& listBvhNode,
			std::vector<GPUBvhNode>& listGpuBvhNode);

		bool hit(
			int exid,
			const std::vector<std::vector<GPUBvhNode>>& listGpuBvhNode,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const;

	private:
		bvh m_bvh;

		int m_exid{ 1 };

		std::vector<std::vector<GPUBvhNode>> m_listGpuBvhNode;
		std::vector<aten::mat4> m_mtxs;
	};
}
