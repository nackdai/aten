#include "kernel/bvh.cuh"
#include "kernel/intersect.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"

#define STACK_SIZE	(64)

struct BVHRay : public aten::ray {
	aten::vec3 inv;
	int sign[3];

	__device__ BVHRay(const aten::ray& r)
	{
		org = r.org;
		dir = r.dir;

		inv = real(1) / dir;

		sign[0] = (inv.x < real(0) ? 1 : 0);
		sign[1] = (inv.y < real(0) ? 1 : 0);
		sign[2] = (inv.z < real(0) ? 1 : 0);
	}
};

__device__ bool intersectAABB(
	const BVHRay* ray,
	const float4* aabb,
	real& t_result)
{
	// NOTE
	// https://github.com/hpicgs/cgsee/wiki/Ray-Box-Intersection-on-the-GPU

	float tmin, tmax, tymin, tymax, tzmin, tzmax;

	tmin = (aabb[ray->sign[0]].x - ray->org.x) * ray->inv.x;
	tmax = (aabb[1 - ray->sign[0]].x - ray->org.x) * ray->inv.x;
	
	tymin = (aabb[ray->sign[1]].y - ray->org.y) * ray->inv.y;
	tymax = (aabb[1 - ray->sign[1]].y - ray->org.y) * ray->inv.y;
	
	tzmin = (aabb[ray->sign[2]].z - ray->org.z) * ray->inv.z;
	tzmax = (aabb[1 - ray->sign[2]].z - ray->org.z) * ray->inv.z;
	
	tmin = max(max(tmin, tymin), tzmin);
	tmax = min(min(tmax, tymax), tzmax);

	if (tmin > tmax) {
		return false;
	}

	t_result = tmin;

	return true;
}

__device__ bool intersectBVH(
	cudaTextureObject_t nodes,
	int* exid,
	int* shapeid,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::hitrecord* rec,
	aten::hitrecordOption* recOpt)
{
	int stackbuf[STACK_SIZE];

	stackbuf[0] = 0;

	int stackpos = 1;

	aten::hitrecord recTmp;
	aten::hitrecordOption recOptTmp;

	bool isHit = false;

	real hitt = AT_MATH_INF;

	*exid = -1;
	int tmpexid = -1;

	*shapeid = -1;
	int tmpshapeid = -1;

	int nodeid = -1;
	float4 node;	// x:left, y:right
	float4 attrib;	// x:shapeid, y:primid, z:nestid

	float4 aabb[2];

	BVHRay bvhray(r);
	real t = AT_MATH_INF;

	while (stackpos > 0) {
		nodeid = stackbuf[stackpos - 1];
		stackpos--;

		if (nodeid >= 0) {
			node = tex1Dfetch<float4>(nodes, 4 * nodeid + 0);	// x : left, y: right
			attrib = tex1Dfetch<float4>(nodes, 4 * nodeid + 1);	// x : shapeid, y : primgid, z : nestid, w : exid
			aabb[0] = tex1Dfetch<float4>(nodes, 4 * nodeid + 2);
			aabb[1] = tex1Dfetch<float4>(nodes, 4 * nodeid + 3);

			if (node.x < 0 && node.y < 0) {
				if (attrib.z >= 0) {
					if (intersectAABB(&bvhray, aabb, t)) {
						stackbuf[stackpos++] = (int)attrib.z;
						tmpshapeid = (int)attrib.x;
					}
				}
				else {
					isHit = false;
					tmpexid = -1;

					const auto* s = &ctxt->shapes[(int)attrib.x];

					if (attrib.w >= 0) {	// exid
						t = AT_MATH_INF;
						isHit = intersectAABB(&bvhray, aabb, t);
						recTmp.t = t;
						tmpexid = attrib.w;
					}
#if 0
					else if (attrib.y >= 0) {	// primid
						const auto& prim = ctxt->prims[(int)attrib.y];
						isHit = intersectShape(s, &prim, ctxt, r, t_min, t_max, &recTmp, &recOptTmp);
						recTmp.mtrlid = prim.mtrlid;
					}
#endif
					else {
						// TODO
						// Only sphere...
						//isHit = intersectShape(s, nullptr, ctxt, r, t_min, t_max, &recTmp, &recOptTmp);
						isHit = hitSphere(s, nullptr, ctxt, r, t_min, t_max, &recTmp, &recOptTmp);
						recTmp.mtrlid = s->mtrl.idx;
					}

					if (isHit) {
						if (recTmp.t < hitt) {
							hitt = recTmp.t;
							*exid = tmpexid;
							*shapeid = tmpshapeid;
						}
						if (tmpexid < 0) {
							if (recTmp.t < rec->t) {
								*rec = recTmp;
								*recOpt = recOptTmp;
								rec->obj = (void*)s;
							}
						}
					}
				}
			}
			else {
				if (intersectAABB(&bvhray, aabb, t)) {
					stackbuf[stackpos++] = (int)node.x;
					stackbuf[stackpos++] = (int)node.y;

					if (stackpos > STACK_SIZE) {
						//AT_ASSERT(false);
						return false;
					}
				}
			}
		}
	}

	return (rec->obj != nullptr);
}

__device__ bool intersectBVH(
	cudaTextureObject_t nodes,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::hitrecord* rec,
	aten::hitrecordOption* recOpt)
{
	int stackbuf[STACK_SIZE];

	stackbuf[0] = 0;

	int stackpos = 1;

	aten::hitrecord recTmp;
	aten::hitrecordOption recOptTmp;

	bool isHit = false;

	int nodeid = -1;
	float4 node;	// x:left, y:right
	float4 attrib;	// x:shapeid, y:primid, z:nestid

	float4 _boxmin;
	float4 _boxmax;

	aten::vec3 boxmin;
	aten::vec3 boxmax;

	while (stackpos > 0) {
		nodeid = stackbuf[stackpos - 1];
		stackpos--;

		if (nodeid >= 0) {
			node = tex1Dfetch<float4>(nodes, 4 * nodeid + 0);	// x : left, y: right
			attrib = tex1Dfetch<float4>(nodes, 4 * nodeid + 1);	// x : shapeid, y : primgid, z : nestid, w : exid
			_boxmin = tex1Dfetch<float4>(nodes, 4 * nodeid + 2);
			_boxmax = tex1Dfetch<float4>(nodes, 4 * nodeid + 3);

			boxmin = aten::make_float3(_boxmin.x, _boxmin.y, _boxmin.z);
			boxmax = aten::make_float3(_boxmax.x, _boxmax.y, _boxmax.z);

			if (node.x < 0 && node.y < 0) {
				isHit = false;

				const auto* s = &ctxt->shapes[(int)attrib.x];
				const aten::ShapeParameter* realShape = (s->shapeid >= 0 ? &ctxt->shapes[s->shapeid] : s);

				const auto& prim = ctxt->prims[(int)attrib.y];

				isHit = hitTriangle(realShape, &prim, ctxt, r, t_min, t_max, &recTmp, &recOptTmp);
				recTmp.mtrlid = prim.mtrlid;

				if (recTmp.t < rec->t) {
					*rec = recTmp;
					*recOpt = recOptTmp;
					rec->obj = (void*)s;
				}
			}
			else {
				if (aten::aabb::hit(r, boxmin, boxmax, t_min, t_max)) {
					stackbuf[stackpos++] = (int)node.x;
					stackbuf[stackpos++] = (int)node.y;

					if (stackpos > STACK_SIZE) {
						//AT_ASSERT(false);
						return false;
					}
				}
			}
		}
	}

	return (rec->obj != nullptr);
}


__device__ bool intersectBVH(
	const Context* ctxt,
	const aten::ray& r,
	float t_min, float t_max,
	aten::hitrecord* rec)
{
	int exid = -1;
	int shapeid = -1;

	aten::hitrecordOption recOpt;

	bool isHit = intersectBVH(
		ctxt->nodes[0],
		&exid,
		&shapeid,
		ctxt,
		r,
		t_min, t_max,
		rec, &recOpt);

	if (exid >= 0) {
		aten::hitrecord recTmp;

		const auto& param = ctxt->shapes[shapeid];

		auto mtxW2L = ctxt->matrices[param.mtxid * 2 + 1];

		aten::ray transformedRay;
		transformedRay.org = mtxW2L.apply(r.org);
		transformedRay.dir = mtxW2L.applyXYZ(r.dir);
		transformedRay.dir = normalize(transformedRay.dir);

		if (intersectBVH(ctxt->nodes[exid], ctxt, transformedRay, t_min, t_max, &recTmp, &recOpt)) {
			if (recTmp.t < rec->t) {
				*rec = recTmp;
				isHit = true;
			}
		}
	}

	if (isHit) {
		evalHitResult(ctxt, (aten::ShapeParameter*)rec->obj, r, rec, &recOpt);
	}

	return isHit;
}