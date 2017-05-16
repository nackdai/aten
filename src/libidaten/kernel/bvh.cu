#include "kernel/bvh.cuh"
#include "kernel/intersect.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

#define STACK_SIZE	(64)

AT_DEVICE_API bool intersectBVH(
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord* rec)
{
	int stackbuf[STACK_SIZE];

	stackbuf[0] = 0;

	int stackpos = 1;
	int nestedStackPos = -1;

	aten::ray transformedRay = r;
	bool isNested = false;
	aten::hitrecord recTmp;
	bool isHit = false;

	int nodeid = -1;
	aten::BVHNode* node = nullptr;

	while (stackpos > 0) {
		if (stackpos == nestedStackPos) {
			nestedStackPos = -1;
			isNested = false;
			transformedRay = r;
		}

		nodeid = stackbuf[stackpos - 1];
		stackpos--;

		if (nodeid >= 0) {
			node = &ctxt->nodes[nodeid];

			if (node->isLeaf()) {
				if (node->nestid >= 0) {
					if (aten::aabb::hit(transformedRay, node->boxmin, node->boxmax, t_min, t_max)) {
						nestedStackPos = isNested ? nestedStackPos : stackpos;
						stackbuf[stackpos++] = (int)node->nestid;

						if (!isNested) {
							const auto& param = ctxt->shapes[(int)node->shapeid];
							transformedRay.org = param.mtxW2L.apply(r.org);
							transformedRay.dir = param.mtxW2L.applyXYZ(r.dir);
							transformedRay.dir = normalize(transformedRay.dir);
							isNested = true;
						}
					}
				}
				else {
					isHit = false;

					const auto* s = &ctxt->shapes[(int)node->shapeid];

					if (node->primid >= 0) {
						const auto& prim = ctxt->prims[(int)node->primid];
						isHit = intersectShape(s, &prim, ctxt, transformedRay, t_min, t_max, &recTmp);
						recTmp.mtrlid = prim.mtrlid;
					}
					else {
						isHit = intersectShape(s, nullptr, ctxt, transformedRay, t_min, t_max, &recTmp);
						recTmp.mtrlid = s->mtrl.idx;
					}

					if (isHit) {
						if (recTmp.t < rec->t) {
							*rec = recTmp;
							rec->obj = (void*)s;
						}
					}
				}
			}
			else {
				if (aten::aabb::hit(transformedRay, node->boxmin, node->boxmax, t_min, t_max)) {
					stackbuf[stackpos++] = (int)node->left;
					stackbuf[stackpos++] = (int)node->right;

					if (stackpos > STACK_SIZE) {
						//AT_ASSERT(false);
						return false;
					}
				}
			}
		}
	}

	isHit = (rec->obj != nullptr);

	if (isHit) {
		evalHitResult(ctxt, (aten::ShapeParameter*)rec->obj, r, rec);
	}

	return isHit;
}