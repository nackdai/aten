#include "defs.h"

template <idaten::IntersectType Type>
AT_INLINE_RELEASE __device__ bool intersectBVHTriangles(
    cudaTextureObject_t nodes,
    const idaten::context* ctxt,
    const aten::ray r,
    float t_min, float t_max,
    aten::Intersection* isect)
{
    aten::Intersection isectTmp;

    int32_t nodeid = 0;

    float4 node0;    // xyz: boxmin, z: hit
    float4 node1;    // xyz: boxmax, z: hit
    float4 attrib;   // x:object_id, y:primid, z:exid, w:meshid

    float3 boxmin;
    float3 boxmax;

    float t = AT_MATH_INF;

    isect->t = t_max;

    while (nodeid >= 0) {
        node0 = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);    // xyz : boxmin, z: hit
        node1 = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);    // xyz : boxmin, z: hit
        attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);    // x : object_id, y : primid, z : exid, w : meshid

        boxmin = make_float3(node0.x, node0.y, node0.z);
        boxmax = make_float3(node1.x, node1.y, node1.z);

        bool isHit = false;

        if (attrib.y >= 0) {
            int32_t primidx = (int32_t)attrib.y;
            aten::TriangleParameter prim;
            prim.v0 = ((aten::vec4*)ctxt->prims)[primidx * aten::TriangleParamter_float4_size + 0];
            prim.v1 = ((aten::vec4*)ctxt->prims)[primidx * aten::TriangleParamter_float4_size + 1];

            isectTmp.t = AT_MATH_INF;
            isHit = AT_NAME::triangle::hit(prim, *ctxt, r, &isectTmp);

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
            isHit = aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t);
        }

        if (isHit) {
            nodeid = (int32_t)node0.w;
        }
        else {
            nodeid = (int32_t)node1.w;
        }
    }

    return (isect->objid >= 0);
}

#define ENABLE_PLANE_LOOP_BVH

template <idaten::IntersectType Type>
AT_INLINE_RELEASE __device__ bool intersectBVH(
    const idaten::context* ctxt,
    const aten::ray r,
    float t_min, float t_max,
    aten::Intersection* isect)
{
    aten::Intersection isectTmp;

    isect->t = t_max;

    int32_t nodeid = 0;

    float4 node0;    // xyz: boxmin, z: hit
    float4 node1;    // xyz: boxmax, z: hit
    float4 attrib;    // x:object_id, y:primid, z:exid,    w:meshid

    float3 boxmin;
    float3 boxmax;

    float t = AT_MATH_INF;

    cudaTextureObject_t node = ctxt->nodes[0];
    aten::ray transformedRay = r;

    int32_t toplayerHit = -1;
    int32_t toplayerMiss = -1;
    int32_t objid = 0;
    int32_t meshid = 0;

    while (nodeid >= 0) {
        node0 = tex1Dfetch<float4>(node, aten::GPUBvhNodeSize * nodeid + 0);    // xyz : boxmin, z: hit
        node1 = tex1Dfetch<float4>(node, aten::GPUBvhNodeSize * nodeid + 1);    // xyz : boxmin, z: hit
        attrib = tex1Dfetch<float4>(node, aten::GPUBvhNodeSize * nodeid + 2);    // x : object_id, y : primid, z : exid, w : meshid

        boxmin = make_float3(node0.x, node0.y, node0.z);
        boxmax = make_float3(node1.x, node1.y, node1.z);

        bool isHit = false;

#ifdef ENABLE_PLANE_LOOP_BVH
        if (attrib.x >= 0 || attrib.y >= 0) {
            // Leaf.
            if (attrib.z >= 0) {    // exid
                //if (aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t)) {
                {
                    const auto* s = &ctxt->GetObject(static_cast<uint32_t>(attrib.x));

                    if (s->mtx_id >= 0) {
                        auto mtx_W2L = ctxt->GetMatrix(s->mtx_id + 1);
                        transformedRay.dir = mtx_W2L.applyXYZ(r.dir);
                        transformedRay.dir = normalize(transformedRay.dir);
                        transformedRay.org = mtx_W2L.apply(r.org) + AT_MATH_EPSILON * transformedRay.dir;
                    }
                    else {
                        transformedRay = r;
                    }

                    int32_t exid = __float_as_int(attrib.z);
                    exid = AT_BVHNODE_MAIN_EXID(exid);

                    node = ctxt->nodes[exid];

                    objid = (int32_t)attrib.x;
                    meshid = (int32_t)attrib.w;

                    toplayerHit = (int32_t)node0.w;
                    toplayerMiss = (int32_t)node1.w;

                    nodeid = 0;

                    continue;
                }
            }
            else if (attrib.y >= 0) {
                int32_t primidx = (int32_t)attrib.y;
                aten::TriangleParameter prim;
                prim.v0 = ((aten::vec4*)ctxt->prims)[primidx * aten::TriangleParamter_float4_size + 0];
                prim.v1 = ((aten::vec4*)ctxt->prims)[primidx * aten::TriangleParamter_float4_size + 1];

                isectTmp.t = AT_MATH_INF;
                isHit = AT_NAME::triangle::hit(prim, *ctxt, transformedRay, &isectTmp);

                bool isIntersect = (Type == idaten::IntersectType::Any
                    ? isHit
                    : isectTmp.t < isect->t);

                if (isIntersect) {
                    *isect = isectTmp;
                    isect->objid = objid;
                    isect->triangle_id = (int32_t)attrib.y;
                    isect->mtrlid = prim.mtrlid;

                    isect->meshid = prim.mesh_id;
                    isect->meshid = (isect->meshid < 0 ? meshid : isect->meshid);

                    t_max = isect->t;

                    if (Type == idaten::IntersectType::Closer
                        || Type == idaten::IntersectType::Any)
                    {
                        return true;
                    }
                }
            }
        }
        else {
            isHit = aten::aabb::hit(transformedRay, boxmin, boxmax, t_min, t_max, &t);
        }

        if (isHit) {
            nodeid = (int32_t)node0.w;
        }
        else {
            nodeid = (int32_t)node1.w;
        }

        if (nodeid < 0 && toplayerHit >= 0) {
            nodeid = isHit ? toplayerHit : toplayerMiss;
            toplayerHit = -1;
            toplayerMiss = -1;
            node = ctxt->nodes[0];
            transformedRay = r;
        }
#else
        if (attrib.x >= 0) {
            // Leaf.
            const auto* s = &ctxt->shapes[(int32_t)attrib.x];

            if (attrib.z >= 0) {    // exid
                if (aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t)) {
                    aten::ray transformedRay;

                    if (s->mtx_id >= 0) {
                        const auto& mtx_W2L = ctxt->GetMatrix(s->mtx_id + 1);
                        transformedRay.dir = mtx_W2L.applyXYZ(r.dir);
                        transformedRay.dir = normalize(transformedRay.dir);
                        transformedRay.org = mtx_W2L.apply(r.org) + AT_MATH_EPSILON * transformedRay.dir;
                    }
                    else {
                        transformedRay = r;
                    }

                    isHit = intersectBVHTriangles<Type>(ctxt->nodes[(int32_t)attrib.z], ctxt, transformedRay, t_min, t_max, &isectTmp);
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
            isHit = aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t);
        }

        if (isHit) {
            nodeid = (int32_t)node0.w;
        }
        else {
            nodeid = (int32_t)node1.w;
        }
#endif
    }

    return (isect->objid >= 0);
}

AT_INLINE_RELEASE __device__ bool intersectBVH(
    const idaten::context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    float t_max/*= AT_MATH_INF*/,
    bool enableLod/*= false*/,
    int32_t depth/*= -1*/)
{
	(void)enableLod;
	(void)depth;

    float t_min = AT_MATH_EPSILON;

    bool isHit = intersectBVH<idaten::IntersectType::Closest>(
        ctxt,
        r,
        t_min, t_max,
        isect);

    return isHit;
}

AT_INLINE_RELEASE __device__ bool intersectCloserBVH(
    const idaten::context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    const float t_max,
    bool enableLod/*= false*/,
    int32_t depth/*= -1*/)
{
    (void)enableLod;
    (void)depth;

    float t_min = AT_MATH_EPSILON;

    bool isHit = intersectBVH<idaten::IntersectType::Closer>(
        ctxt,
        r,
        t_min, t_max,
        isect);

    return isHit;
}

AT_INLINE_RELEASE __device__ bool intersectAnyBVH(
    const idaten::context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    bool enableLod/*= false*/,
    int32_t depth/*= -1*/)
{
    (void)enableLod;
    (void)depth;

    float t_min = AT_MATH_EPSILON;
    float t_max = AT_MATH_INF;

    bool isHit = intersectBVH<idaten::IntersectType::Any>(
        ctxt,
        r,
        t_min, t_max,
        isect);

    return isHit;
}
