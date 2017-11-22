#pragma once

#include "scene/hitable.h"
#include "accelerator/bvh.h"

namespace aten {
	struct ThreadedBvhNode {
		aten::vec3 boxmin;		///< AABB min position.
		float hit{ -1 };		///< Link index if ray hit.

		aten::vec3 boxmax;		///< AABB max position.
		float miss{ -1 };		///< Link index if ray miss.

		float shapeid{ -1 };	///< Object index.
		float primid{ -1 };		///< Triangle index.
		float exid{ -1 };		///< External bvh index.
		float meshid{ -1 };		///< Mesh id.

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
			uint32_t num,
			aabb* bbox = nullptr) override;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

		virtual bool hitMultiLevel(
			const accelerator::ResultIntersectTestByFrustum& fisect,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override final;

		virtual accelerator::ResultIntersectTestByFrustum intersectTestByFrustum(const frustum& f) override final;

		virtual void drawAABB(
			aten::hitable::FuncDrawAABB func,
			const aten::mat4& mtxL2W) override final
		{
			m_bvh.drawAABB(func, mtxL2W);
		}

		const std::vector<std::vector<ThreadedBvhNode>>& getNodes() const
		{
			return m_listThreadedBvhNode;
		}
		const std::vector<aten::mat4>& getMatrices() const
		{
			return m_mtxs;
		}

		void disableLayer()
		{
			m_enableLayer = false;
		}

		const std::vector<accelerator*>& getNestedAccel()
		{
			return m_nestedBvh;
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

		void registerThreadedBvhNode(
			bool isPrimitiveLeaf,
			const std::vector<ThreadedBvhNodeEntry>& listBvhNode,
			std::vector<ThreadedBvhNode>& listThreadedBvhNode,
			std::vector<int>& listParentId);

		void setOrder(
			const std::vector<ThreadedBvhNodeEntry>& listBvhNode,
			const std::vector<int>& listParentId,
			std::vector<ThreadedBvhNode>& listThreadedBvhNode);

		bool hit(
			int exid,
			const std::vector<std::vector<ThreadedBvhNode>>& listThreadedBvhNode,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const;

		bool hitMultiLevel(
			int exid,
			int nodeid,
			int topid,
			const std::vector<std::vector<ThreadedBvhNode>>& listThreadedBvhNode,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const;

	private:
		bvh m_bvh;

		int m_exid{ 1 };

		bool m_enableLayer{ true };

		std::vector<std::vector<ThreadedBvhNode>> m_listThreadedBvhNode;
		std::vector<aten::mat4> m_mtxs;

		std::vector<accelerator*> m_nestedBvh;
	};
}
