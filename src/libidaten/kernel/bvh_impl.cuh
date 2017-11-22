#include "kernel/idatendefs.cuh"

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectBVHTriangles(
	cudaTextureObject_t nodes,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	aten::Intersection isectTmp;

	int nodeid = 0;
	
	float4 node0;	// xyz: boxmin, z: hit
	float4 node1;	// xyz: boxmax, z: hit
	float4 attrib;	// x:shapeid, y:primid, z:exid,	w:meshid

	float4 boxmin;
	float4 boxmax;

	float t = AT_MATH_INF;

	isect->t = t_max;

	while (nodeid >= 0) {
		node0 = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);	// xyz : boxmin, z: hit
		node1 = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);	// xyz : boxmin, z: hit
		attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);	// x : shapeid, y : primid, z : exid, w : meshid

		boxmin = make_float4(node0.x, node0.y, node0.z, 1.0f);
		boxmax = make_float4(node1.x, node1.y, node1.z, 1.0f);

		bool isHit = false;

		if (attrib.y >= 0) {
			int primidx = (int)attrib.y;
			aten::PrimitiveParamter prim;
			prim.v0 = ((aten::vec4*)ctxt->prims)[primidx * aten::PrimitiveParamter_float4_size + 0];
			prim.v1 = ((aten::vec4*)ctxt->prims)[primidx * aten::PrimitiveParamter_float4_size + 1];

			isectTmp.t = AT_MATH_INF;
			isHit = hitTriangle(&prim, ctxt, r, &isectTmp);

			bool isIntersect = (Type == idaten::IntersectType::Any
				? isHit
				: isectTmp.t < isect->t);

			if (isIntersect) {
				*isect = isectTmp;
				isect->objid = (int)attrib.x;
				isect->primid = (int)attrib.y;
				isect->mtrlid = prim.mtrlid;

				//isect->meshid = (int)attrib.w;
				isect->meshid = prim.gemoid;

				t_max = isect->t;

				if (Type == idaten::IntersectType::Closer
					|| Type == idaten::IntersectType::Any)
				{
					return true;
				}
			}
		}
		else {
			isHit = hitAABB(r.org, r.dir, boxmin, boxmax, t_min, t_max, &t);
		}

		if (isHit) {
			nodeid = (int)node0.w;
		}
		else {
			nodeid = (int)node1.w;
		}
	}

	return (isect->objid >= 0);
}

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectBVH(
	cudaTextureObject_t nodes,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	aten::Intersection isectTmp;

	isect->t = t_max;

	int nodeid = 0;
	
	float4 node0;	// xyz: boxmin, z: hit
	float4 node1;	// xyz: boxmax, z: hit
	float4 attrib;	// x:shapeid, y:primid, z:exid,	w:meshid

	float4 boxmin;
	float4 boxmax;

	real t = AT_MATH_INF;

	while (nodeid >= 0) {
		node0 = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);	// xyz : boxmin, z: hit
		node1 = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);	// xyz : boxmin, z: hit
		attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);	// x : shapeid, y : primid, z : exid, w : meshid

		boxmin = make_float4(node0.x, node0.y, node0.z, 1.0f);
		boxmax = make_float4(node1.x, node1.y, node1.z, 1.0f);

		bool isHit = false;

		if (attrib.x >= 0) {
			// Leaf.
			const auto* s = &ctxt->shapes[(int)attrib.x];

			if (attrib.z >= 0) {	// exid
				//if (aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t)) {
				if (hitAABB(r.org, r.dir, boxmin, boxmax, t_min, t_max, &t)) {
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

					isHit = intersectBVHTriangles<Type>(ctxt->nodes[(int)attrib.z], ctxt, transformedRay, t_min, t_max, &isectTmp);
				}
			}
			else {
				// TODO
				// Only sphere...
				//isHit = intersectShape(s, nullptr, ctxt, r, t_min, t_max, &recTmp, &recOptTmp);
				isectTmp.t = AT_MATH_INF;
				isHit = hitSphere(s, r, t_min, t_max, &isectTmp);
				isectTmp.mtrlid = s->mtrl.idx;
			}

			bool isIntersect = (Type == idaten::IntersectType::Any
				? isHit
				: isectTmp.t < isect->t);

			if (isIntersect) {
				*isect = isectTmp;
				isect->objid = (int)attrib.x;
				isect->meshid = (isect->meshid < 0 ? (int)attrib.w : isect->meshid);

				t_max = isect->t;

				if (Type == idaten::IntersectType::Closer
					|| Type == idaten::IntersectType::Any)
				{
					return true;
				}
			}
		}
		else {
			//isHit = aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t);
			isHit = hitAABB(r.org, r.dir, boxmin, boxmax, t_min, t_max, &t);
		}

		if (isHit) {
			nodeid = (int)node0.w;
		}
		else {
			nodeid = (int)node1.w;
		}
	}

	return (isect->objid >= 0);
}

AT_CUDA_INLINE __device__ bool intersectBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect,
	float t_max/*= AT_MATH_INF*/)
{
	float t_min = AT_MATH_EPSILON;

	bool isHit = intersectBVH<idaten::IntersectType::Closest>(
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

	bool isHit = intersectBVH<idaten::IntersectType::Closer>(
		ctxt->nodes[0],
		ctxt,
		r,
		t_min, t_max,
		isect);

	return isHit;
}

AT_CUDA_INLINE __device__ bool intersectAnyBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect)
{
	float t_min = AT_MATH_EPSILON;
	float t_max = AT_MATH_INF;

	bool isHit = intersectBVH<idaten::IntersectType::Any>(
		ctxt->nodes[0],
		ctxt,
		r,
		t_min, t_max,
		isect);

	return isHit;
}
