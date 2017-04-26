#include "kernel/bvh.cuh"
#include "kernel/intersect.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"

#define STACK_SIZE	(64)

AT_DEVICE_API bool intersectBVH(
	const Context& ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord& rec)
{
	aten::BVHNode* stackbuf[STACK_SIZE] = { nullptr };
	aten::BVHNode** stack = &stackbuf[0];

	// push terminator.
	*stack++ = nullptr;

	int stackpos = 1;

	aten::BVHNode* node = &ctxt.nodes[0];

	do {
		aten::BVHNode* left = node->left >= 0 ? &ctxt.nodes[node->left] : nullptr;
		aten::BVHNode* right = node->right >= 0 ? &ctxt.nodes[node->right] : nullptr;

		bool traverseLeft = false;
		bool traverseRight = false;

		if (left) {
			if (left->isLeaf()) {
				aten::hitrecord recLeft;
				const auto& leftobj = ctxt.shapes[left->shapeid];

				if (intersectShape(leftobj, r, t_min, t_max, recLeft)) {
					if (recLeft.t < rec.t) {
						rec = recLeft;
						rec.obj = (void*)&leftobj;
						rec.mtrlid = leftobj.mtrl.idx;
					}
				}
			}
			else {
				traverseLeft = left->bbox.hit(r, t_min, t_max);
			}
		}
		if (right) {
			if (right->isLeaf()) {
				aten::hitrecord recRight;
				const auto& rightobj = ctxt.shapes[right->shapeid];

				if (intersectShape(rightobj, r, t_min, t_max, recRight)) {
					if (recRight.t < rec.t) {
						rec = recRight;
						rec.obj = (void*)&rightobj;
						rec.mtrlid = rightobj.mtrl.idx;
					}
				}
			}
			else {
				traverseRight = right->bbox.hit(r, t_min, t_max);
			}
		}

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
		AT_ASSERT(0 <= stackpos && stackpos < STACK_SIZE);

		if (stackpos >= STACK_SIZE) {
			return false;
		}
	} while (node != nullptr);

	return (rec.obj != nullptr);
}