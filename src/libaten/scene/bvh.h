#pragma once

#include "scene/hitable.h"
#include "scene/accel.h"
#include "sampler/random.h"

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

	private:
		void build(
			bvhnode** list,
			uint32_t num,
			bool needSort);

	protected:
		bvhnode* m_left{ nullptr };
		bvhnode* m_right{ nullptr };
		aabb m_aabb;

		int m_traverseOrder{ -1 };
	};

	//////////////////////////////////////////////

	class bvh : public accel {
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

		virtual void collectNodes(std::vector<BVHNode>& nodes) const override final;

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
