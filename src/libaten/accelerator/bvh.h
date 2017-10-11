#pragma once

#include "scene/hitable.h"
#include "accelerator/accelerator.h"

namespace aten {
	class transformable;

	class bvhnode {
		friend class bvh;

	public:
		bvhnode() {}
		virtual ~bvhnode() {}

	private:
		bvhnode(hitable* item) : m_item(item) {}

	public:
		void build(
			hitable** list,
			uint32_t num);

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

		int getTraversalOrder() const
		{
			return m_traverseOrder;
		}

	protected:
		bvhnode* m_left{ nullptr };
		bvhnode* m_right{ nullptr };

		aabb m_aabb;

		hitable* m_item{ nullptr };

		int m_traverseOrder{ -1 };
		int m_externalId{ -1 };
	};

	//////////////////////////////////////////////

	class bvh : public accelerator {
		friend class bvhnode;

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

	private:
		bvhnode* m_root{ nullptr };
	};
}
