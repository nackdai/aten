#include "kernel/idatendefs.cuh"

AT_CUDA_INLINE __device__ int SkipCode(int mask, int pos)
{
	return (((mask >> (pos + 1))) | (mask << (3 - pos))) & 7;
}

AT_CUDA_INLINE __device__ int bitScan(int n)
{
	// NOTE
	// http://www.techiedelight.com/bit-hacks-part-3-playing-rightmost-set-bit-number/

	// if number is odd, return 1
	if (n & 1) {
		//return 1;
		return 0;
	}

	// unset rightmost bit and xor with number itself
	n = n ^ (n & (n - 1));

	// find the position of the only set bit in the result
	// we can directly return log2(n) + 1 from the function
#if 0
	int pos = 0;
	while (n)
	{
		n = n >> 1;
		pos++;
	}
#else
	int pos = (int)(log2f((float)n) + 1);
#endif
	return pos - 1;
}

AT_CUDA_INLINE __device__ int SkipCodeNext(int code)
{
	int n = bitScan(code);
	int newCode = code >> (n + 1);
	return newCode ^ code;
}

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectStacklessQBVHTriangles(
	cudaTextureObject_t nodes,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	aten::Intersection isectTmp;

	int nodeid = 0;
	uint32_t bitstack = 0;

	int skipCode = 0;
	
	for (;;) {
		// x : leftChildrenIdx, y ; isLeaf, z : numChildren, w : parent
		float4 node = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);

		// x: shapeid, y : primid, z : exid, w : meshid
		float4 attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);

		float4 bminx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 3);
		float4 bmaxx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 4);
		float4 bminy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 5);
		float4 bmaxy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 6);
		float4 bminz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 7);
		float4 bmaxz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 8);

		int leftChildrenIdx = (int)node.x;
		int isLeaf = (int)node.y;
		int numChildren = (int)node.z;
		int parent = (int)node.w;

		bool isHit = false;

		if (isLeaf) {
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
			aten::vec4 intersectT;
			int hitMask = hit4AABBWith1Ray(
				&intersectT,
				r.org, r.dir,
				bminx, bmaxx,
				bminy, bmaxy,
				bminz, bmaxz,
				t_min, t_max);

			if (hitMask > 0) {
				bitstack = bitstack << 3;

				if (hitMask == 1) {
					nodeid = leftChildrenIdx + 0;
				}
				else if (hitMask == 2) {
					nodeid = leftChildrenIdx + 1;
				}
				else if (hitMask == 4) {
					nodeid = leftChildrenIdx + 2;
				}
				else if (hitMask == 8) {
					nodeid = leftChildrenIdx + 3;
				}
				else {
					int nearId_a = (intersectT.x < intersectT.y ? 0 : 1);
					int nearId_b = (intersectT.z < intersectT.w ? 2 : 3);

					int nearPos = (intersectT[nearId_a] < intersectT[nearId_b] ? nearId_a : nearId_b);

					nodeid = leftChildrenIdx + nearPos;

					skipCode = SkipCode(hitMask, nearPos);
					bitstack = bitstack | skipCode;
				}

				continue;
			}
		}

		while ((skipCode = (bitstack & 7)) == 0) {
			if (bitstack == 0) {
				return (isect->objid >= 0);
			}

			nodeid = (int)node.w;	// parent
			bitstack = bitstack >> 3;

			node = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);
		}

		float4 sib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);

		auto siblingPos = bitScan(skipCode);

		nodeid = (int)((float*)&sib)[siblingPos];

		int n = SkipCodeNext(skipCode);
		bitstack = bitstack ^ n;
	}

	return (isect->objid >= 0);
}

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectStacklessQBVH(
	cudaTextureObject_t nodes,
	const Context* ctxt,
	const aten::ray r,
	float t_min, float t_max,
	aten::Intersection* isect)
{
	aten::Intersection isectTmp;

	isect->t = t_max;

	int nodeid = 0;
	uint32_t bitstack = 0;

	int skipCode = 0;

	for (;;) {
		// x : leftChildrenIdx, y ; isLeaf, z : numChildren, w : parent
		float4 node = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);

		// x: shapeid, y : primid, z : exid, w : meshid
		float4 attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);

		float4 bminx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 3);
		float4 bmaxx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 4);
		float4 bminy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 5);
		float4 bmaxy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 6);
		float4 bminz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 7);
		float4 bmaxz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 8);

		int leftChildrenIdx = (int)node.x;
		int isLeaf = (int)node.y;
		int numChildren = (int)node.z;

		bool isHit = false;

		if (isLeaf) {
			// Leaf.
			const auto* s = &ctxt->shapes[(int)attrib.x];

			if (attrib.z >= 0) {
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

				isHit = intersectStacklessQBVHTriangles<Type>(
					ctxt->nodes[(int)attrib.z], ctxt, transformedRay, t_min, t_max, &isectTmp);
			}
			else {
				// TODO
				// Only sphere...
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
			aten::vec4 intersectT;
			int hitMask = hit4AABBWith1Ray(
				&intersectT,
				r.org, r.dir,
				bminx, bmaxx,
				bminy, bmaxy,
				bminz, bmaxz,
				t_min, t_max);

			if (hitMask > 0) {
				bitstack = bitstack << 3;

				if (hitMask == 1) {
					nodeid = leftChildrenIdx + 0;
				}
				else if (hitMask == 2) {
					nodeid = leftChildrenIdx + 1;
				}
				else if (hitMask == 4) {
					nodeid = leftChildrenIdx + 2;
				}
				else if (hitMask == 8) {
					nodeid = leftChildrenIdx + 3;
				}
				else {
					int nearId_a = (intersectT.x < intersectT.y ? 0 : 1);
					int nearId_b = (intersectT.z < intersectT.w ? 2 : 3);

					int nearPos = (intersectT[nearId_a] < intersectT[nearId_b] ? nearId_a : nearId_b);

					nodeid = leftChildrenIdx + nearPos;

					skipCode = SkipCode(hitMask, nearPos);
					bitstack = bitstack | skipCode;
				}

				continue;
			}
		}

		while ((skipCode = (bitstack & 7)) == 0) {
			if (bitstack == 0) {
				return (isect->objid >= 0);
			}

			nodeid = (int)node.w;	// parent
			bitstack = bitstack >> 3;

			node = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);
		}

		float4 sib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);

		auto siblingPos = bitScan(skipCode);

		nodeid = (int)((float*)&sib)[siblingPos];

		int n = SkipCodeNext(skipCode);
		bitstack = bitstack ^ n;
	}

	return (isect->objid >= 0);
}

AT_CUDA_INLINE __device__ bool intersectClosestStacklessQBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect)
{
	float t_min = AT_MATH_EPSILON;
	float t_max = AT_MATH_INF;

	bool isHit = intersectStacklessQBVH<idaten::IntersectType::Closest>(
		ctxt->nodes[0],
		ctxt,
		r,
		t_min, t_max,
		isect);

	return isHit;
}

AT_CUDA_INLINE __device__ bool intersectCloserStacklessQBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect,
	const float t_max)
{
	float t_min = AT_MATH_EPSILON;

	bool isHit = intersectStacklessQBVH<idaten::IntersectType::Closer>(
		ctxt->nodes[0],
		ctxt,
		r,
		t_min, t_max,
		isect);

	return isHit;
}

AT_CUDA_INLINE __device__ bool intersectAnyStacklessQBVH(
	const Context* ctxt,
	const aten::ray& r,
	aten::Intersection* isect)
{
	float t_min = AT_MATH_EPSILON;
	float t_max = AT_MATH_INF;

	bool isHit = intersectStacklessQBVH<idaten::IntersectType::Any>(
		ctxt->nodes[0],
		ctxt,
		r,
		t_min, t_max,
		isect);

	return isHit;
}
