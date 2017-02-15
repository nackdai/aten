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

	void bvh::build(
		hitable** list,
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

		if (num == 1) {
			m_left = list[0];
		}
		else if (num == 2) {
			m_left = list[0];
			m_right = list[1];
		}
		else {
			m_left = new bvh(list, num / 2);
			m_right = new bvh(list + num / 2, num - num / 2);
		}

		if (m_left && m_right) {
			auto boxLeft = m_left->getBoundingbox();
			auto boxRight = m_right->getBoundingbox();

			m_aabb = aabb::surrounding_box(boxLeft, boxRight);
		}
		else {
			auto boxLeft = m_left->getBoundingbox();

			m_aabb = boxLeft;
		}
	}

	bool bvh::hit(
		const ray& r,
		real t_min, real t_max,
		hitrecord& rec) const
	{
		auto isHit = m_aabb.hit(r, t_min, t_max);

		if (isHit) {
			hitrecord recLeft;
			hitrecord recRight;

			bool isHitLeft = false;
			bool isHitRight = false;

			if (m_left) {
				isHitLeft = m_left->hit(r, t_min, t_max, recLeft);
			}
			if (m_right) {
				isHitRight = m_right->hit(r, t_min, t_max, recRight);
			}

			if (isHitLeft && isHitRight) {
				if (recLeft.t < recRight.t) {
					rec = recLeft;
				}
				else {
					rec = recRight;
				}
				return true;
			}
			else if (isHitLeft) {
				rec = recLeft;
				return true;
			}
			else if (isHitRight) {
				rec = recRight;
				return true;
			}
			else {
				return false;
			}
		}

		return false;
	}
}
