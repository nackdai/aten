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
	int nestedStackPos = -1;

	aten::BVHNode* node = &ctxt.nodes[0];

	aten::ray transformedRay = r;
	bool isNested = false;

	do {
		aten::BVHNode* left = node->left >= 0 ? &ctxt.nodes[node->left] : nullptr;
		aten::BVHNode* right = node->right >= 0 ? &ctxt.nodes[node->right] : nullptr;

		bool traverseLeft = false;
		bool traverseRight = false;

		real left_t = AT_MATH_INF;
		real right_t = AT_MATH_INF;

		if (left) {
			if (left->isLeaf()) {
				aten::hitrecord recLeft;
				bool isHitLeft = false;
				const auto& leftobj = ctxt.shapes[left->shapeid];

				if (left->nestid >= 0) {
					traverseLeft = left->bbox.hit(r, t_min, t_max, &left_t);

					if (traverseLeft) {
						left = &ctxt.nodes[left->nestid];
						nestedStackPos = stackpos;
					}
				}
				else {
					if (left->primid >= 0) {
					}
					else {
						isHitLeft = intersectShape(leftobj, isNested ? transformedRay : r, t_min, t_max, recLeft);
					}
				}

				if (isHitLeft) {
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
				bool isHitRight = false;
				const auto& rightobj = ctxt.shapes[right->shapeid];

				if (right->nestid >= 0) {
					traverseRight = right->bbox.hit(r, t_min, t_max, &right_t);

					if (traverseRight && right_t < left_t) {
						right = &ctxt.nodes[right->nestid];
						nestedStackPos = stackpos;
					}
				}
				else {
					if (right->primid >= 0) {

					}
					else {
						isHitRight = intersectShape(rightobj, isNested ? transformedRay : r, t_min, t_max, recRight);
					}
				}

				if (isHitRight) {
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
			if (nestedStackPos == stackpos) {
				nestedStackPos = -1;
				isNested = false;
			}

			node = *(--stack);
			stackpos -= 1;
		}
		else {
			node = traverseLeft ? left : right;

			if (traverseLeft && traverseRight) {
				*(stack++) = right;
				stackpos += 1;
			}

			if (!isNested && nestedStackPos >= 0) {
				const auto& param = ctxt.shapes[node->shapeid];
				transformedRay = param.mtxW2L.applyRay(r);
				isNested = true;
			}
		}
		AT_ASSERT(0 <= stackpos && stackpos < STACK_SIZE);

		if (stackpos >= STACK_SIZE) {
			return false;
		}
	} while (node != nullptr);

	return (rec.obj != nullptr);
}