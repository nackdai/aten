#include "kernel/idatendefs.cuh"

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectQBVHTriangles(
    int32_t* stack,
    int32_t beginStackPos,
    cudaTextureObject_t nodes,
    const idaten::Context* ctxt,
    const aten::ray r,
    float t_min, float t_max,
    aten::Intersection* isect)
{
    aten::Intersection isectTmp;

    int32_t stackpos = beginStackPos + 1;

    stack[stackpos] = 0;
    stackpos++;

    while (stackpos > beginStackPos) {
        int32_t nodeid = stack[stackpos - 1];
        stackpos -= 1;

        // x : leftChildIdx, y : isLeaf, z : numChildren
        float4 node = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);

        // x: object_id, y : primid, z : exid, w : meshid
        float4 attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);

        float4 bminx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);
        float4 bmaxx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 3);
        float4 bminy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 4);
        float4 bmaxy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 5);
        float4 bminz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 6);
        float4 bmaxz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 7);

        int32_t leftChildrenIdx = (int32_t)node.x;
        int32_t isLeaf = (int32_t)node.y;
        int32_t numChildren = (int32_t)node.z;

        bool isHit = false;

        if (isLeaf) {
            int32_t primidx = (int32_t)attrib.y;
            aten::TriangleParameter prim;
            prim.v0 = ((aten::vec4*)ctxt->prims)[primidx * aten::TriangleParamter_float4_size + 0];
            prim.v1 = ((aten::vec4*)ctxt->prims)[primidx * aten::TriangleParamter_float4_size + 1];

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
            int32_t res = hit4AABBWith1Ray(
                &intersectT,
                r.org, r.dir,
                bminx, bmaxx,
                bminy, bmaxy,
                bminz, bmaxz,
                t_min, t_max);

            // Stack hit children.
            if (res > 0) {
#pragma unroll
                for (int32_t i = 0; i < 4; i++) {
                    if ((res & (1 << i)) && intersectT[i] < t_max) {
                        stack[stackpos] = leftChildrenIdx + i;
                        stackpos++;
                    }
                }
            }
        }
    }

    return (isect->objid >= 0);
}

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectQBVH(
    cudaTextureObject_t nodes,
    const idaten::Context* ctxt,
    const aten::ray r,
    float t_min, float t_max,
    aten::Intersection* isect)
{
    static const int32_t stacksize = 64;
    int32_t stack[stacksize];

    aten::Intersection isectTmp;

    isect->t = t_max;

    int32_t stackpos = 0;
    stack[stackpos] = 0;
    stackpos++;

    while (stackpos > 0) {
        int32_t nodeid = stack[stackpos - 1];
        stackpos -= 1;

        // x : leftChildIdx, y : isLeaf, z : numChildren
        float4 node = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);

        // x: object_id, y : primid, z : exid, w : meshid
        float4 attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);

        float4 bminx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);
        float4 bmaxx = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 3);
        float4 bminy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 4);
        float4 bmaxy = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 5);
        float4 bminz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 6);
        float4 bmaxz = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 7);

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

                isHit = intersectQBVHTriangles<Type>(
                    stack, stackpos,
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
            int32_t res = hit4AABBWith1Ray(
                &intersectT,
                r.org, r.dir,
                bminx, bmaxx,
                bminy, bmaxy,
                bminz, bmaxz,
                t_min, t_max);

            // Stack hit children.
            if (res > 0) {
#pragma unroll
                for (int32_t i = 0; i < 4; i++) {
                    if ((res & (1 << i)) && intersectT[i] < t_max) {
                        stack[stackpos] = leftChildrenIdx + i;
                        stackpos++;
                    }
                }
            }
        }
    }

    return (isect->objid >= 0);
}

AT_CUDA_INLINE __device__ bool intersectQBVH(
    const idaten::Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    const float t_max/*= AT_MATH_INF*/)
{
    float t_min = AT_MATH_EPSILON;

    bool isHit = intersectQBVH<idaten::IntersectType::Closest>(
        ctxt->nodes[0],
        ctxt,
        r,
        t_min, t_max,
        isect);

    return isHit;
}

AT_CUDA_INLINE __device__ bool intersectCloserQBVH(
    const idaten::Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    const float t_max)
{
    float t_min = AT_MATH_EPSILON;

    bool isHit = intersectQBVH<idaten::IntersectType::Closer>(
        ctxt->nodes[0],
        ctxt,
        r,
        t_min, t_max,
        isect);

    return isHit;
}

AT_CUDA_INLINE __device__ bool intersectAnyQBVH(
    const idaten::Context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect)
{
    float t_min = AT_MATH_EPSILON;
    float t_max = AT_MATH_INF;

    bool isHit = intersectQBVH<idaten::IntersectType::Any>(
        ctxt->nodes[0],
        ctxt,
        r,
        t_min, t_max,
        isect);

    return isHit;
}
