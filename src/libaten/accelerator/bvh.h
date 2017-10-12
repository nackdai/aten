#pragma once

#include "scene/hitable.h"
#include "accelerator/accelerator.h"

namespace aten {
	class transformable;

	class bvhnode {
		friend class bvh;

	public:
		bvhnode(bvhnode* parent) : m_parent(parent) {}
		virtual ~bvhnode() {}

	private:
		bvhnode(bvhnode* parent, hitable* item)
			: m_item(item), m_parent(parent)
		{}

	public:
		bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const;

		const aabb& getBoundingbox() const
		{
			if (m_item) {
				return m_item->getBoundingbox();
			}
			return m_aabb;
		}
		void setBoundingBox(const aabb& bbox)
		{
			m_aabb = bbox;
		}

		bool isLeaf() const
		{
			return (!m_left && !m_right);
		}

		bvhnode* getLeft()
		{
			return m_left;
		}
		bvhnode* getRight()
		{
			return m_right;
		}

		bvhnode* getParent()
		{
			return m_parent;
		}

		hitable* getItem()
		{
			return m_item;
		}

		int getTraversalOrder() const
		{
			return m_traverseOrder;
		}
		void setTraversalOrder(int order)
		{
			m_traverseOrder = order;
		}

		int getExternalId() const
		{
			return m_externalId;
		}
		void setExternalId(int exid)
		{
			m_externalId = exid;
		}

	protected:
		bvhnode* m_left{ nullptr };
		bvhnode* m_right{ nullptr };

		bvhnode* m_parent{ nullptr };

		aabb m_aabb;

		hitable* m_item{ nullptr };

		int m_traverseOrder{ -1 };
		int m_externalId{ -1 };
	};

	//////////////////////////////////////////////

	class bvh : public accelerator {
		friend class bvhnode;
		friend class accelerator;
		friend class GPUBvh;

		static std::vector<bvh*> s_bvhList;

	public:
		bvh() {}
		virtual ~bvh() {}

	public:
		virtual void build(
			hitable** list,
			uint32_t num) override;

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const override;

		virtual const aabb& getBoundingbox() const override
		{
			if (m_root) {
				return std::move(m_root->getBoundingbox());
			}
			return std::move(aabb());
		}

		bvhnode* getRoot()
		{
			return m_root;
		}

		static int registerToList(bvh* b);
		static std::vector<bvh*>& bvh::getBvhList();

	private:
		static bool hit(
			const bvhnode* root,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect);

		static void buildBySAH(
			bvhnode* root,
			hitable** list,
			uint32_t num);

	protected:
		bvhnode* m_root{ nullptr };
	};
}
