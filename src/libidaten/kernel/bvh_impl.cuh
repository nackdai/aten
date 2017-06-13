#include "kernel/idatendefs.cuh"

AT_CUDA_INLINE __device__ bool intersectAABB(
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

AT_CUDA_INLINE __device__ bool intersectBVHTriangles(
	cudaTextureObject_t nodes,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::Intersection* isect,
	IntersectType type = IntersectType::Closest)
{
	aten::Intersection isectTmp;

	int nodeid = 0;
	float4 node;	// x:left, y:right
	float4 attrib;	// x:shapeid, y:primid, z:nestid

	float4 _boxmin;
	float4 _boxmax;

	aten::vec3 boxmin;
	aten::vec3 boxmax;

	isect->t = t_max;

	for (;;) {
		if (nodeid < 0) {
			break;
		}

		node = tex1Dfetch<float4>(nodes, 4 * nodeid + 0);	// x : hit, y: miss
		attrib = tex1Dfetch<float4>(nodes, 4 * nodeid + 1);	// x : shapeid, y : primgid, z : exid
		_boxmin = tex1Dfetch<float4>(nodes, 4 * nodeid + 2);
		_boxmax = tex1Dfetch<float4>(nodes, 4 * nodeid + 3);

		boxmin = aten::vec3(_boxmin.x, _boxmin.y, _boxmin.z);
		boxmax = aten::vec3(_boxmax.x, _boxmax.y, _boxmax.z);

		bool isHit = false;

		if (attrib.x >= 0 || attrib.y >= 0) {
			const auto& prim = ctxt->prims[(int)attrib.y];

			isectTmp.t = AT_MATH_INF;
			isHit = hitTriangle(&prim, ctxt, r, &isectTmp);
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
			isHit = aten::aabb::hit(r, boxmin, boxmax, t_min, t_max);
		}

		if (isHit) {
			nodeid = (int)node.x;
		}
		else {
			nodeid = (int)node.y;
		}
	}

	return (isect->objid >= 0);
}

AT_CUDA_INLINE __device__ bool intersectBVH(
	cudaTextureObject_t nodes,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::Intersection* isect,
	IntersectType type = IntersectType::Closest)
{
	aten::Intersection isectTmp;

	real hitt = t_max;
	isect->t = t_max;

	int tmpexid = -1;

	int nodeid = 0;
	float4 node;	// x:left, y:right
	float4 attrib;	// x:shapeid, y:primid, z:nestid

	float4 aabb[2];

	real t = AT_MATH_INF;

	for (;;) {
		if (nodeid < 0) {
			break;
		}

		node = tex1Dfetch<float4>(nodes, 4 * nodeid + 0);	// x : hit, y: miss
		attrib = tex1Dfetch<float4>(nodes, 4 * nodeid + 1);	// x : shapeid, y : primgid, z : exid
		aabb[0] = tex1Dfetch<float4>(nodes, 4 * nodeid + 2);
		aabb[1] = tex1Dfetch<float4>(nodes, 4 * nodeid + 3);

		auto boxmin = aten::vec3(aabb[0].x, aabb[0].y, aabb[0].z);
		auto boxmax = aten::vec3(aabb[1].x, aabb[1].y, aabb[1].z);

		bool isHit = false;

		if (attrib.x >= 0 || attrib.y >= 0) {
			// Leaf.
			tmpexid = -1;

			const auto* s = &ctxt->shapes[(int)attrib.x];

			if (attrib.z >= 0) {	// exid
				aten::ray transformedRay;

				if (s->mtxid >= 0) {
					auto mtxW2L = ctxt->matrices[s->mtxid * 2 + 1];
					transformedRay.dir = mtxW2L.applyXYZ(r.dir);
					transformedRay.dir = normalize(transformedRay.dir);
					transformedRay.org = mtxW2L.apply(r.org) + AT_MATH_EPSILON * transformedRay.dir;
				}
				else {
					transformedRay = r;
				}

				isHit = intersectBVHTriangles(ctxt->nodes[(int)attrib.z], ctxt, transformedRay, t_min, t_max, &isectTmp, type);
			}
			else {
				// TODO
				// Only sphere...
				//isHit = intersectShape(s, nullptr, ctxt, r, t_min, t_max, &recTmp, &recOptTmp);
				isectTmp.t = AT_MATH_INF;
				isHit = hitSphere(s, r, t_min, t_max, &isectTmp);
				isectTmp.mtrlid = s->mtrl.idx;
			}

			if (isHit) {
				if (isectTmp.t < isect->t) {
					*isect = isectTmp;
					isect->objid = (int)attrib.x;

					if (type == IntersectType::Closer) {
						return true;
					}
				}
			}
		}
		else {
			isHit = aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t);
		}

		if (isHit) {
			nodeid = (int)node.x;
		}
		else {
			nodeid = (int)node.y;
		}
	}

	return (isect->objid >= 0);
}

AT_CUDA_INLINE __device__ bool intersectBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect)
{
	float t_min = AT_MATH_EPSILON;
	float t_max = AT_MATH_INF;

	bool isHit = intersectBVH(
		ctxt->nodes[0],
		ctxt,
		r,
		t_min, t_max,
		isect);

	return isHit;
}

AT_CUDA_INLINE __device__ bool intersectCloserBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect,
	const float t_max)
{
	float t_min = AT_MATH_EPSILON;

	bool isHit = intersectBVH(
		ctxt->nodes[0],
		ctxt,
		r,
		t_min, t_max,
		isect,
		IntersectType::Closer);

	return isHit;
}