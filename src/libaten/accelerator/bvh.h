#pragma once

#include "scene/hitable.h"
#include "accelerator/accelerator.h"

namespace aten {
	class transformable;

	class bvhnode : public hitable {
		friend class bvh;

	public:
		bvhnode() {}
		virtual ~bvhnode() {}

	private:
		bvhnode(
			bvhnode** list,
			uint32_t num)
		{
			build(list, num);
		}

	public:
		void build(
			bvhnode** list,
			uint32_t num);

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

		virtual aabb getBoundingbox() const override final
		{
			return std::move(m_aabb);
		}

		virtual bool isLeaf() const
		{
			return (!m_left && !m_right);
		}

		virtual const hitable* getHasObject() const
		{
			return nullptr;
		}

		virtual void getMatrices(
			aten::mat4& mtxL2W,
			aten::mat4& mtxW2L) const
		{
			AT_ASSERT(false);
		}

		int getTraversalOrder() const
		{
			return m_traverseOrder;
		}

	protected:
		void build(
			bvhnode** list,
			uint32_t num,
			bool needSort);

		virtual bool setBVHNodeParam(
			BVHNode& param,
			const bvhnode* parent,
			const int idx,
			std::vector<std::vector<BVHNode>>& nodes,
			const transformable* instanceParent,
			const aten::mat4& mtxL2W);

		virtual void registerToList(
			const int idx,
			std::vector<std::vector<bvhnode*>>& nodeList);

		virtual void getNodes(
			bvhnode*& left,
			bvhnode*& right);

		virtual bvhnode* getNode()
		{
			AT_ASSERT(false);
			return nullptr;
		}

	protected:
		bvhnode* m_left{ nullptr };
		bvhnode* m_right{ nullptr };
		aabb m_aabb;

		int m_traverseOrder{ -1 };
		int m_externalId{ -1 };
	};

	//////////////////////////////////////////////

	class bvh : public accelerator {
		friend class bvhnode;

	public:
		virtual void build(
			bvhnode** list,
			uint32_t num) override;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

		virtual aabb getBoundingbox() const override
		{
			if (m_root) {
				return std::move(m_root->getBoundingbox());
			}
			return std::move(aabb());
		}

		virtual void collectNodes(
			std::vector<std::vector<BVHNode>>& nodes,
			std::vector<aten::mat4>& mtxs) const override final;

		static void registerToList(
			bvhnode* root,
			const int idx,
			std::vector<std::vector<bvhnode*>>& nodeList);

		static void collectNodes(
			bvhnode* root,
			const int idx,
			std::vector<std::vector<BVHNode>>& nodes,
			const transformable* instanceParent,
			const aten::mat4& mtxL2W);

		static void dumpCollectedNodes(std::vector<BVHNode>& nodes, const char* path);

	private:
		static bool hit(
			const bvhnode* root,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect);

		static void buildBySAH(
			bvhnode* root,
			bvhnode** list,
			uint32_t num);

		static void collectNodes(
			bvhnode* node,
			const bvhnode* parent,
			const int idx,
			std::vector<std::vector<BVHNode>>& nodes,
			const transformable* instanceParent,
			const aten::mat4& mtxL2W);

	private:
		bvhnode* m_root{ nullptr };
	};
}
