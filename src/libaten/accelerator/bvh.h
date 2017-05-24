#pragma once

#include "scene/hitable.h"
#include "accelerator/accelerator.h"

namespace aten {
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
			hitrecord& rec) const override;

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

		virtual int collectInternalNodes(
			std::vector<std::vector<aten::BVHNode>>& nodes, 
			int order, 
			bvhnode* parent,
			const aten::mat4& mtxL2W = aten::mat4())
		{
			// Nothing is done...
			return order;
		}

	private:
		void build(
			bvhnode** list,
			uint32_t num,
			bool needSort);

		virtual void setBVHNodeParamInCollectNodes(BVHNode& param)
		{
			// Nothing is done...
		}

	protected:
		bvhnode* m_left{ nullptr };
		bvhnode* m_right{ nullptr };
		aabb m_aabb;

		int m_traverseOrder{ -1 };
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
			hitrecord& rec) const override;

		virtual aabb getBoundingbox() const override
		{
			if (m_root) {
				return std::move(m_root->getBoundingbox());
			}
			return std::move(aabb());
		}

		virtual void collectNodes(std::vector<std::vector<BVHNode>>& nodes) const override final;

		static int setTraverseOrder(bvhnode* root, int curOrder);
		static void collectNodes(bvhnode* root, std::vector<BVHNode>& nodes, bvhnode* parent);
		static void dumpCollectedNodes(std::vector<BVHNode>& nodes, const char* path);

	private:
		static bool hit(
			const bvhnode* root,
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec);

		static void buildBySAH(
			bvhnode* root,
			bvhnode** list,
			uint32_t num);

	private:
		bvhnode* m_root{ nullptr };
	};
}
