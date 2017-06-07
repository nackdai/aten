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

enum IntersectType {
	Closest,
	Closer,
	Any,
};

struct BVHCandidate {
	int exid;
	int shapeid;
};

__device__ bool intersectBVH(
	cudaTextureObject_t nodes,
	BVHCandidate* candidates,
	int& candidateNum,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::Intersection* isect,
	IntersectType type = IntersectType::Closest)
{
	int stackbuf[STACK_SIZE];

	stackbuf[0] = 0;

	int stackpos = 1;

	aten::Intersection isectTmp;

	real hitt = t_max;
	isect->t = t_max;

	int tmpexid = -1;

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

		tmpshapeid = -1;

		if (nodeid >= 0) {
			node = tex1Dfetch<float4>(nodes, 4 * nodeid + 0);	// x : left, y: right
			attrib = tex1Dfetch<float4>(nodes, 4 * nodeid + 1);	// x : shapeid, y : primgid, z : nestid, w : exid
			aabb[0] = tex1Dfetch<float4>(nodes, 4 * nodeid + 2);
			aabb[1] = tex1Dfetch<float4>(nodes, 4 * nodeid + 3);

			auto boxmin = aten::make_float3(aabb[0].x, aabb[0].y, aabb[0].z);
			auto boxmax = aten::make_float3(aabb[1].x, aabb[1].y, aabb[1].z);

			if (node.x < 0 && node.y < 0) {
				if (attrib.z >= 0) {
					//if (intersectAABB(&bvhray, aabb, t)) {
					if (aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t)) {
						stackbuf[stackpos++] = (int)attrib.z;
						tmpshapeid = (int)attrib.x;
					}
				}
				else {
					tmpexid = -1;

					bool isHit = false;

					const auto* s = &ctxt->shapes[(int)attrib.x];

					if (attrib.w >= 0) {	// exid
						isectTmp.t = AT_MATH_INF;
						//isHit = intersectAABB(&bvhray, aabb, isectTmp.t);
						isHit = aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &isectTmp.t);
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
						isectTmp.t = AT_MATH_INF;
						hitSphere(s, nullptr, ctxt, r, t_min, t_max, &isectTmp);
						isectTmp.mtrlid = s->mtrl.idx;
					}

#if 0
					if (isectTmp.t < hitt)
#else
					if (isHit)
#endif
					{
						hitt = isectTmp.t;

						if (tmpexid >= 0) {
							candidates[candidateNum].exid = tmpexid;
							candidates[candidateNum].shapeid = (int)attrib.x;
							candidateNum++;
						}
					}
					if (tmpexid < 0) {
						if (isectTmp.t < isect->t) {
							*isect = isectTmp;
							isect->objid = (int)attrib.x;

							if (type == IntersectType::Closer) {
								return true;
							}
						}
					}
				}
			}
			else {
				//if (intersectAABB(&bvhray, aabb, t)) {
				if (aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t)) {
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

	return (isect->objid >= 0);
}

__device__ bool intersectBVH(
	cudaTextureObject_t nodes,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::Intersection* isect,
	IntersectType type = IntersectType::Closest)
{
	int stackbuf[10];

	stackbuf[0] = 0;

	int stackpos = 1;

	aten::Intersection isectTmp;

	int nodeid = -1;
	float4 node;	// x:left, y:right
	float4 attrib;	// x:shapeid, y:primid, z:nestid

	float4 _boxmin;
	float4 _boxmax;

	aten::vec3 boxmin;
	aten::vec3 boxmax;

	isect->t = t_max;

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
				const auto* s = &ctxt->shapes[(int)attrib.x];
				const aten::ShapeParameter* realShape = (s->shapeid >= 0 ? &ctxt->shapes[s->shapeid] : s);

				const auto& prim = ctxt->prims[(int)attrib.y];

				isectTmp.t = AT_MATH_INF;
				hitTriangle(realShape, &prim, ctxt, r, t_min, t_max, &isectTmp);
				isectTmp.mtrlid = prim.mtrlid;

				if (isectTmp.t < isect->t) {
					*isect = isectTmp;
					isect->objid = (int)attrib.x;
					isect->primid = (int)attrib.y;

					if (type == IntersectType::Closer) {
						return true;
					}
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

	return (isect->objid >= 0);
}


__device__ bool intersectBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect)
{
	float t_min = AT_MATH_EPSILON;
	float t_max = AT_MATH_INF;

	BVHCandidate candidates[STACK_SIZE];
	int candidateNum = 0;

	bool isHit = intersectBVH(
		ctxt->nodes[0],
		candidates, candidateNum,
		ctxt,
		r,
		t_min, t_max,
		isect);

	for (int i = 0; i < candidateNum; i++) {
		const auto& c = candidates[i];

		aten::Intersection isectTmp;

		const auto& param = ctxt->shapes[c.shapeid];

		auto mtxW2L = ctxt->matrices[param.mtxid * 2 + 1];

		aten::ray transformedRay;
		transformedRay.dir = mtxW2L.applyXYZ(r.dir);
		transformedRay.dir = normalize(transformedRay.dir);
		transformedRay.org = mtxW2L.apply(r.org) + AT_MATH_EPSILON * transformedRay.dir;

		if (intersectBVH(ctxt->nodes[c.exid], ctxt, transformedRay, t_min, t_max, &isectTmp)) {
			if (isectTmp.t < isect->t) {
				*isect = isectTmp;
				isHit = true;
			}
		}
	}

	return isHit;
}

__device__ bool intersectCloserBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect,
	const float t_max)
{
	float t_min = AT_MATH_EPSILON;

	BVHCandidate candidates[STACK_SIZE];
	int candidateNum = 0;

	bool isHit = intersectBVH(
		ctxt->nodes[0],
		candidates, candidateNum,
		ctxt,
		r,
		t_min, t_max,
		isect,
		IntersectType::Closer);

	aten::ray transformedRay;
	int previd = -1;

	for(int i = 0; i < candidateNum; i++) {
		const auto& c = candidates[i];

		aten::Intersection isectTmp;

		const auto& param = ctxt->shapes[c.shapeid];

		auto mtxW2L = ctxt->matrices[param.mtxid * 2 + 1];

		if (c.shapeid != previd)
		{
			transformedRay.dir = mtxW2L.applyXYZ(r.dir);
			transformedRay.dir = normalize(transformedRay.dir);
			transformedRay.org = mtxW2L.apply(r.org) + AT_MATH_EPSILON * transformedRay.dir;
		}
		previd = c.shapeid;

		if (intersectBVH(ctxt->nodes[c.exid], ctxt, transformedRay, t_min, t_max, &isectTmp, IntersectType::Closer)) {
			if (isectTmp.t < isect->t) {
				*isect = isectTmp;
				isHit = true;
			}
		}
	}

	return isHit;
}