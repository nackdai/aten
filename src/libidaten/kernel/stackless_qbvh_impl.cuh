#include "kernel/idatendefs.cuh"

AT_CUDA_INLINE __device__ int32_t SkipCode(int32_t mask, int32_t pos)
{
    return (((mask >> (pos + 1))) | (mask << (3 - pos))) & 7;
}

AT_CUDA_INLINE __device__ int32_t bitScan(int32_t n)
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
    int32_t pos = 0;
    while (n)
    {
        n = n >> 1;
        pos++;
    }
#else
    int32_t pos = (int32_t)(log2f((float)n) + 1);
#endif
    return pos - 1;
}

AT_CUDA_INLINE __device__ int32_t SkipCodeNext(int32_t code)
{
    int32_t n = bitScan(code);
    int32_t newCode = code >> (n + 1);
    return newCode ^ code;
}

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectStacklessQBVHTriangles(
    cudaTextureObject_t nodes,
    const idaten::Context* ctxt,
    const aten::ray r,
    float t_min, float t_max,
    aten::Intersection* isect)
{
    aten::Intersection isectTmp;

    int32_t nodeid = 0;
    uint32_t bitstack = 0;

    int32_t skipCode = 0;

    for (;;) {
        // x : leftChildrenIdx, y ; isLeaf, z : numChildren, w : parent
        float4 node = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);

        // x: object_id, y : primid, z : exid, w : meshid
        float4 attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);

        float4 bminx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 3);
        float4 bmaxx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 4);
        float4 bminy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 5);
        float4 bmaxy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 6);
        float4 bminz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 7);
        float4 bmaxz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 8);

        int32_t leftChildrenIdx = (int32_t)node.x;
        int32_t isLeaf = (int32_t)node.y;
        int32_t numChildren = (int32_t)node.z;
        int32_t parent = (int32_t)node.w;

        bool isHit = false;

        if (isLeaf) {
            int32_t primidx = (int32_t)attrib.y;
            aten::TriangleParameter prim;
            prim.v0 = ((aten::vec4*)ctxt->prims)[primidx * aten::PrimitiveParamter_float4_size + 0];
            prim.v1 = ((aten::vec4*)ctxt->prims)[primidx * aten::PrimitiveParamter_float4_size + 1];

            isectTmp.t = AT_MATH_INF;
            isHit = hitTriangle(&prim, ctxt, r, &isectTmp);

            bool isIntersect = (Type == idaten::IntersectType::Any
                ? isHit
                : isectTmp.t < isect->t);

            if (isIntersect) {
                *isect = isectTmp;
                isect->objid = (int32_t)attrib.x;
                isect->triangle_id = (int32_t)attrib.y;
                isect->mtrlid = prim.mtrlid;

                //isect->meshid = (int32_t)attrib.w;
                isect->meshid = prim.mesh_id;

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
            int32_t hitMask = hit4AABBWith1Ray(
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
                    int32_t nearId_a = (intersectT.x < intersectT.y ? 0 : 1);
                    int32_t nearId_b = (intersectT.z < intersectT.w ? 2 : 3);

                    int32_t nearPos = (intersectT[nearId_a] < intersectT[nearId_b] ? nearId_a : nearId_b);

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

            nodeid = (int32_t)node.w;    // parent
            bitstack = bitstack >> 3;

            node = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);
        }

        float4 sib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);

        auto siblingPos = bitScan(skipCode);

        nodeid = (int32_t)((float*)&sib)[siblingPos];

        int32_t n = SkipCodeNext(skipCode);
        bitstack = bitstack ^ n;
    }

    return (isect->objid >= 0);
}

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectStacklessQBVH(
    cudaTextureObject_t nodes,
    const idaten::Context* ctxt,
    const aten::ray r,
    float t_min, float t_max,
    aten::Intersection* isect)
{
    aten::Intersection isectTmp;

    isect->t = t_max;

    int32_t nodeid = 0;
    uint32_t bitstack = 0;

    int32_t skipCode = 0;

    for (;;) {
        // x : leftChildrenIdx, y ; isLeaf, z : numChildren, w : parent
        float4 node = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);

        // x: object_id, y : primid, z : exid, w : meshid
        float4 attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);

        float4 bminx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 3);
        float4 bmaxx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 4);
        float4 bminy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 5);
        float4 bmaxy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 6);
        float4 bminz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 7);
        float4 bmaxz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 8);

        int32_t leftChildrenIdx = (int32_t)node.x;
        int32_t isLeaf = (int32_t)node.y;
        int32_t numChildren = (int32_t)node.z;

        bool isHit = false;

        if (isLeaf) {
            // Leaf.
            const auto* s = &ctxt->shapes[(int32_t)attrib.x];

            if (attrib.z >= 0) {
                aten::ray transformedRay;

                if (s->mtx_id >= 0) {
                    auto mtxW2L = ctxt->matrices[s->mtx_id * 2 + 1];
                    transformedRay.dir = mtxW2L.applyXYZ(r.dir);
                    transformedRay.dir = normalize(transformedRay.dir);
                    transformedRay.org = mtxW2L.apply(r.org) + AT_MATH_EPSILON * transformedRay.dir;
                }
                else {
                    transformedRay = r;
                }

                isHit = intersectStacklessQBVHTriangles<Type>(
                    ctxt->nodes[(int32_t)attrib.z], ctxt, transformedRay, t_min, t_max, &isectTmp);
            }
            else {
                // TODO
                // Only sphere...
                isectTmp.t = AT_MATH_INF;
                isHit = hitSphere(s, r, t_min, t_max, &isectTmp);
                isectTmp.mtrlid = s->sphere.mtrl_id;
            }

            bool isIntersect = (Type == idaten::IntersectType::Any
                ? isHit
                : isectTmp.t < isect->t);

            if (isIntersect) {
                *isect = isectTmp;
                isect->objid = (int32_t)attrib.x;
                isect->meshid = (isect->meshid < 0 ? (int32_t)attrib.w : isect->meshid);

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
            int32_t hitMask = hit4AABBWith1Ray(
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
                    int32_t nearId_a = (intersectT.x < intersectT.y ? 0 : 1);
                    int32_t nearId_b = (intersectT.z < intersectT.w ? 2 : 3);

                    int32_t nearPos = (intersectT[nearId_a] < intersectT[nearId_b] ? nearId_a : nearId_b);

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

            nodeid = (int32_t)node.w;    // parent
            bitstack = bitstack >> 3;

            node = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);
        }

        float4 sib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);

        auto siblingPos = bitScan(skipCode);

        nodeid = (int32_t)((float*)&sib)[siblingPos];

        int32_t n = SkipCodeNext(skipCode);
        bitstack = bitstack ^ n;
    }

    return (isect->objid >= 0);
}

AT_CUDA_INLINE __device__ bool intersectClosestStacklessQBVH(
    const idaten::Context* ctxt,
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
    const idaten::Context* ctxt,
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
    const idaten::Context* ctxt,
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
