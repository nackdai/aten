#include "scene/bvh.h"

#include <random>

namespace aten {
	int compareX(const void* a, const void* b)
	{
		const hitable* ah = *(hitable**)a;
		const hitable* bh = *(hitable**)b;

		auto left = ah->getBoundingbox();
		auto right = bh->getBoundingbox();

		if (left.minPos().x < right.minPos().x) {
			return -1;
		}
		else {
			return 1;
		}
	}

	int compareY(const void* a, const void* b)
	{
		hitable* ah = *(hitable**)a;
		hitable* bh = *(hitable**)b;

		auto left = ah->getBoundingbox();
		auto right = bh->getBoundingbox();

		if (left.minPos().y < right.minPos().y) {
			return -1;
		}
		else {
			return 1;
		}
	}

	int compareZ(const void* a, const void* b)
	{
		hitable* ah = *(hitable**)a;
		hitable* bh = *(hitable**)b;

		auto left = ah->getBoundingbox();
		auto right = bh->getBoundingbox();

		if (left.minPos().z < right.minPos().z) {
			return -1;
		}
		else {
			return 1;
		}
	}

	void bvhnode::build(
		bvhnode** list,
		uint32_t num)
	{
		build(list, num, true);
	}

	void bvhnode::build(
		bvhnode** list,
		uint32_t num,
		bool needSort)
	{
		if (needSort) {
			// TODO
			int axis = (int)(::rand() % 3);

			switch (axis) {
			case 0:
				::qsort(list, num, sizeof(hitable*), compareX);
				break;
			case 1:
				::qsort(list, num, sizeof(hitable*), compareY);
				break;
			default:
				::qsort(list, num, sizeof(hitable*), compareZ);
				break;
			}
		}

		if (num == 1) {
			m_left = list[0];
		}
		else if (num == 2) {
			m_left = list[0];
			m_right = list[1];
		}
		else {
			m_left = new bvhnode(list, num / 2);
			m_right = new bvhnode(list + num / 2, num - num / 2);
		}

		if (m_left && m_right) {
			auto boxLeft = m_left->getBoundingbox();
			auto boxRight = m_right->getBoundingbox();

			m_aabb = aabb::merge(boxLeft, boxRight);
		}
		else {
			auto boxLeft = m_left->getBoundingbox();

			m_aabb = boxLeft;
		}
	}

	bool bvhnode::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		auto isHit = m_aabb.hit(r, t_min, t_max);

		if (isHit) {
			isHit = bvh::hit(this, r, t_min, t_max, rec);
		}

		return isHit;
	}

	///////////////////////////////////////////////////////

	void bvh::build(
		bvhnode** list,
		uint32_t num)
	{
		// TODO
		int axis = (int)(::rand() % 3);

		switch (axis) {
		case 0:
			::qsort(list, num, sizeof(hitable*), compareX);
			break;
		case 1:
			::qsort(list, num, sizeof(hitable*), compareY);
			break;
		default:
			::qsort(list, num, sizeof(hitable*), compareZ);
			break;
		}

		m_root = new bvhnode();
		m_root->build(&list[0], num, false);
	}

	bool bvh::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		bool isHit = hit(m_root, r, t_min, t_max, rec);
		return isHit;
	}

	bool bvh::hit(
		const bvhnode* root,
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec)
	{
		// NOTE
		// https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-ii-tree-traversal-gpu/

		// TODO
		// stack size.
		static const uint32_t stacksize = 64;
		bvhnode* stackbuf[stacksize] = { nullptr };
		bvhnode** stack = &stackbuf[0];

		// push.
		*stack++ = nullptr;

		// For debug.
		int stackpos = 1;

		auto node = root;

		do {
			auto left = node->m_left;
			auto right = node->m_right;

			hitrecord recRight;

			bool isHitLeft = false;
			bool isHitRight = false;

			if (left) {
				isHitLeft = left->getBoundingbox().hit(r, t_min, t_max);
			}
			if (right) {
				isHitRight = right->getBoundingbox().hit(r, t_min, t_max);
			}

			if (isHitLeft || isHitRight) {
				int xxx = 0;
			}

			if (isHitLeft && left->isLeaf()) {
				hitrecord recLeft;
				if (left->hit(r, t_min, t_max, recLeft)) {
					if (recLeft.t < rec.t) {
						rec = recLeft;
					}
				}
			}
			if (isHitRight && right->isLeaf()) {
				hitrecord recRight;
				if (right->hit(r, t_min, t_max, recRight)) {
					if (recRight.t < rec.t) {
						rec = recRight;
					}
				}
			}

			bool traverseLeft = isHitLeft && !left->isLeaf();
			bool traverseRight = isHitRight && !right->isLeaf();

			if (!traverseLeft && !traverseRight) {
				node = *(--stack);
				stackpos -= 1;
			}
			else {
				node = traverseLeft ? left : right;

				if (traverseLeft && traverseRight) {
					*(stack++) = right;
					stackpos += 1;
				}
			}
			AT_ASSERT(0 <= stackpos && stackpos < stacksize);
		} while (node != nullptr);

		return (rec.obj != nullptr);
	}
}
