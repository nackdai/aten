#include "kernel/bvh.cuh"
#include "kernel/intersect.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda/helper_math.h"

#define STACK_SIZE	(64)

AT_DEVICE_API bool intersectBVH(
	const Context* ctxt,
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

	aten::BVHNode* node = &ctxt->nodes[0];

	aten::ray transformedRay = r;
	bool isNested = false;

	aten::BVHNode* left = nullptr;
	aten::BVHNode* right = nullptr;

	bool traverseLeft = false;
	bool traverseRight = false;

	real left_t = AT_MATH_INF;
	real right_t = AT_MATH_INF;

	aten::hitrecord recTmp;
	bool isHit = false;

	do {
		left = node->left >= 0 ? &ctxt->nodes[node->left] : nullptr;
		right = node->right >= 0 ? &ctxt->nodes[node->right] : nullptr;

		traverseLeft = false;
		traverseRight = false;

		left_t = AT_MATH_INF;
		right_t = AT_MATH_INF;

		if (left) {
			isHit = false;

			if (left->isLeaf()) {
				const auto& leftobj = ctxt->shapes[left->shapeid];

				if (left->nestid >= 0) {
					traverseLeft = left->bbox.hit(r, t_min, t_max, &left_t);

					if (traverseLeft) {
						left = &ctxt->nodes[left->nestid];
						nestedStackPos = stackpos;
					}
				}
				else {
					if (left->primid >= 0) {
						// hit test primitive...
						const auto& prim = ctxt->prims[left->primid];
						isHit = intersectShape(leftobj, &prim, ctxt, transformedRay, t_min, t_max, recTmp);

						if (isHit) {
							recTmp.p = leftobj.mtxL2W.apply(recTmp.p);
							recTmp.normal = normalize(leftobj.mtxL2W.applyXYZ(recTmp.normal));

							const auto& v0 = ctxt->vertices[prim.idx[0]];
							const auto& v1 = ctxt->vertices[prim.idx[1]];

							real orignalLen = (v1.pos - v0.pos).length();

							real scaledLen = 0;
							{
								auto p0 = leftobj.mtxL2W.apply(v0.pos);
								auto p1 = leftobj.mtxL2W.apply(v1.pos);

								scaledLen = (p1 - p0).length();
							}

							real ratio = scaledLen / orignalLen;
							ratio = ratio * ratio;

							recTmp.area = leftobj.area * ratio;
							recTmp.mtrlid = prim.mtrlid;
						}
					}
					else {
						isHit = intersectShape(leftobj, nullptr, ctxt, isNested ? transformedRay : r, t_min, t_max, recTmp);
						recTmp.mtrlid = leftobj.mtrl.idx;
					}
				}

				if (isHit) {
					if (recTmp.t < rec.t) {
						rec = recTmp;
						rec.obj = (void*)&leftobj;
					}
				}
			}
			else {
				traverseLeft = left->bbox.hit(isNested ? transformedRay : r, t_min, t_max);
			}
		}
		if (right) {
			isHit = false;

			if (right->isLeaf()) {
				const auto& rightobj = ctxt->shapes[right->shapeid];

				if (right->nestid >= 0) {
					traverseRight = right->bbox.hit(r, t_min, t_max, &right_t);

					if (traverseRight && right_t < left_t) {
						right = &ctxt->nodes[right->nestid];
						nestedStackPos = stackpos;
					}
				}
				else {
					if (right->primid >= 0) {
						// hit test primitive...
						const auto& prim = ctxt->prims[right->primid];
						isHit = intersectShape(rightobj, &prim, ctxt, transformedRay, t_min, t_max, recTmp);

						if (isHit) {
							recTmp.p = rightobj.mtxL2W.apply(recTmp.p);
							recTmp.normal = normalize(rightobj.mtxL2W.applyXYZ(recTmp.normal));

							const auto& v0 = ctxt->vertices[prim.idx[0]];
							const auto& v1 = ctxt->vertices[prim.idx[1]];

							real orignalLen = (v1.pos - v0.pos).length();

							real scaledLen = 0;
							{
								auto p0 = rightobj.mtxL2W.apply(v0.pos);
								auto p1 = rightobj.mtxL2W.apply(v1.pos);

								scaledLen = (p1 - p0).length();
							}

							real ratio = scaledLen / orignalLen;
							ratio = ratio * ratio;

							recTmp.area = rightobj.area * ratio;
							recTmp.mtrlid = prim.mtrlid;
						}
					}
					else {
						isHit = intersectShape(rightobj, nullptr, ctxt, isNested ? transformedRay : r, t_min, t_max, recTmp);
						recTmp.mtrlid = rightobj.mtrl.idx;
					}
				}

				if (isHit) {
					if (recTmp.t < rec.t) {
						rec = recTmp;
						rec.obj = (void*)&rightobj;
					}
				}
			}
			else {
				traverseRight = right->bbox.hit(isNested ? transformedRay : r, t_min, t_max);
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
				const auto& param = ctxt->shapes[node->shapeid];
				transformedRay.org = param.mtxW2L.apply(r.org);
				transformedRay.dir = param.mtxW2L.applyXYZ(r.dir);
				transformedRay.dir = normalize(transformedRay.dir);
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