#include "kernel/idatendefs.cuh"

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectSBVHTriangles(
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
    float4 attrib;    // x:object_id, y:primid, z:exid,    w:meshid

    float4 boxmin;
    float4 boxmax;

    float t = AT_MATH_INF;

    isect->t = t_max;

    while (nodeid >= 0) {
        node0 = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 0);    // xyz : boxmin, z: hit
        node1 = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 1);    // xyz : boxmax, z: miss
        attrib = tex1Dfetch<float4>(nodes, aten::GPUBvhNodeSize * nodeid + 2);    // x : parent, y : triid, z : padding, w : padding

        boxmin = make_float4(node0.x, node0.y, node0.z, 1.0f);
        boxmax = make_float4(node1.x, node1.y, node1.z, 1.0f);

        bool isHit = false;

        if (attrib.y >= 0) {
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
                isect->objid = -1;
                isect->triangle_id = primidx;
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
            isHit = hitAABB(r.org, r.dir, boxmin, boxmax, t_min, t_max, &t);
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

#define ENABLE_PLANE_LOOP_SBVH

template <idaten::IntersectType Type>
AT_CUDA_INLINE __device__ bool intersectSBVH(
    const idaten::context* ctxt,
    const aten::ray r,
    float t_min, float t_max,
    aten::Intersection* isect,
    bool enableLod,
    int32_t depth)
{
    aten::Intersection isectTmp;

    isect->t = t_max;
    isect->isVoxel = false;

    int32_t nodeid = 0;

    float4 node0;    // xyz: boxmin, z: hit
    float4 node1;    // xyz: boxmax, z: hit
    float4 attrib;    // x:object_id, y:primid, z:exid,    w:meshid

    float4 boxmin;
    float4 boxmax;

    real t = AT_MATH_INF;

    cudaTextureObject_t node = ctxt->nodes[0];
    aten::ray transformedRay = r;

    bool isTraverseRootTree = true;

    int32_t toplayerHit = -1;
    int32_t toplayerMiss = -1;
    int32_t objid = 0;
    int32_t meshid = 0;

    while (nodeid >= 0) {
        node0 = tex1Dfetch<float4>(node, aten::GPUBvhNodeSize * nodeid + 0);    // xyz : boxmin, z: hit
        node1 = tex1Dfetch<float4>(node, aten::GPUBvhNodeSize * nodeid + 1);    // xyz : boxmin, z: hit
        attrib = tex1Dfetch<float4>(node, aten::GPUBvhNodeSize * nodeid + 2);    // x : object_id, y : primid, z : exid, w : meshid

        boxmin = make_float4(node0.x, node0.y, node0.z, 1.0f);
        boxmax = make_float4(node1.x, node1.y, node1.z, 1.0f);

        bool isHit = false;

#ifdef ENABLE_PLANE_LOOP_SBVH
        if (attrib.x >= 0) {
            // Leaf.
            const auto* s = &ctxt->GetObject(static_cast<uint32_t>(attrib.x));

            if (attrib.z >= 0) {    // exid
                if (s->mtx_id >= 0) {
                    const auto& mtxW2L = ctxt->GetMatrix(s->mtx_id * 2 + 1);
                    transformedRay.dir = mtxW2L.applyXYZ(r.dir);
                    transformedRay.dir = normalize(transformedRay.dir);
                    transformedRay.org = mtxW2L.apply(r.org) + AT_MATH_EPSILON * transformedRay.dir;
                }
                else {
                    transformedRay = r;
                }

                int32_t exid = __float_as_int(attrib.z);
                bool hasLod = AT_BVHNODE_HAS_LOD(exid);
                exid = hasLod && enableLod ? AT_BVHNODE_LOD_EXID(exid) : AT_BVHNODE_MAIN_EXID(exid);
                //exid = AT_BVHNODE_MAIN_EXID(exid);

                node = ctxt->nodes[exid];

                objid = (int32_t)attrib.x;
                meshid = (int32_t)attrib.w;

                toplayerHit = (int32_t)node0.w;
                toplayerMiss = (int32_t)node1.w;

                isHit = true;
                node0.w = 0.0f;

                isTraverseRootTree = false;
            }
            else if (attrib.y >= 0) {
                int32_t primidx = (int32_t)attrib.y;
                aten::TriangleParameter prim;
                prim.v0 = ((aten::vec4*)ctxt->prims)[primidx * aten::TriangleParamter_float4_size + 0];
                prim.v1 = ((aten::vec4*)ctxt->prims)[primidx * aten::TriangleParamter_float4_size + 1];

                isectTmp.t = AT_MATH_INF;
                isHit = hitTriangle(&prim, ctxt, transformedRay, &isectTmp);

                bool isIntersect = (Type == idaten::IntersectType::Any
                    ? isHit
                    : isectTmp.t < isect->t);

                if (isIntersect) {
                    *isect = isectTmp;
                    isect->objid = objid;
                    isect->triangle_id = primidx;
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
#if 1
        // TODO
        else if (enableLod && !isTraverseRootTree && AT_IS_VOXEL(attrib.z)) {
            // Voxel
            const int32_t voxeldepth = (int32_t)AT_GET_VOXEL_DEPTH(attrib.z);

            if (voxeldepth == 3) {
                aten::vec3 nml;
                isHit = hitAABB(transformedRay.org, transformedRay.dir, boxmin, boxmax, t_min, t_max, &t, &nml);

                bool isIntersect = (Type == idaten::IntersectType::Any
                    ? isHit
                    : isHit && t < isect->t);

                if (isIntersect) {
                    isectTmp.isVoxel = true;

                    // TODO
                    // Normal have to be transformed by L2W matrix.

                    isectTmp.t = t;

                    isectTmp.nml_x = nml.x;
                    isectTmp.nml_y = nml.y;
                    isectTmp.nml_z = nml.z;

                    isectTmp.mtrlid = attrib.w;

                    isectTmp.objid = objid;

                    if (isectTmp.t < isect->t) {
                        *isect = isectTmp;
                        t_max = isect->t;
                    }

                    // LODにヒットしたので、子供（詳細）は探索しないようにする.
                    isHit = false;

                    if (Type == idaten::IntersectType::Closer
                        || Type == idaten::IntersectType::Any)
                    {
                        return true;
                    }
                }
            }
        }
#endif
        else {
            //isHit = aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t);
            isHit = hitAABB(transformedRay.org, transformedRay.dir, boxmin, boxmax, t_min, t_max, &t);
        }

        if (isHit) {
            nodeid = (int32_t)node0.w;
        }
        else {
            nodeid = (int32_t)node1.w;
        }

        if (nodeid < 0) {
            nodeid = isHit ? toplayerHit : toplayerMiss;
            toplayerHit = -1;
            toplayerMiss = -1;
            node = ctxt->nodes[0];
            transformedRay = r;
            isTraverseRootTree = true;
        }

#else
        if (attrib.x >= 0) {
            // Leaf.
            const auto* s = &ctxt->GetObject(static_cast<uint32_t>(attrib.x));

            if (attrib.z >= 0) {    // exid
                                    //if (aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t)) {
                if (hitAABB(r.org, r.dir, boxmin, boxmax, t_min, t_max, &t)) {
                    if (s->mtx_id >= 0) {
                        const auto& mtxW2L = ctxt->GetMatrix(s->mtx_id * 2 + 1);
                        transformedRay.dir = mtxW2L.applyXYZ(r.dir);
                        transformedRay.dir = normalize(transformedRay.dir);
                        transformedRay.org = mtxW2L.apply(r.org) + AT_MATH_EPSILON * transformedRay.dir;
                    }
                    else {
                        transformedRay = r;
                    }

                    isHit = intersectSBVHTriangles<Type>(ctxt->nodes[(int32_t)attrib.z], ctxt, transformedRay, t_min, t_max, &isectTmp);
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
            //isHit = aten::aabb::hit(r, boxmin, boxmax, t_min, t_max, &t);
            isHit = hitAABB(r.org, r.dir, boxmin, boxmax, t_min, t_max, &t);
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

AT_CUDA_INLINE __device__ bool intersectClosestSBVH(
    const idaten::context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    float t_max/*= AT_MATH_INF*/,
    bool enableLod/*= false*/,
    int32_t depth/*= -1*/)
{
    float t_min = AT_MATH_EPSILON;

    bool isHit = intersectSBVH<idaten::IntersectType::Closest>(
        ctxt,
        r,
        t_min, t_max,
        isect,
        enableLod,
        depth);

    return isHit;
}

AT_CUDA_INLINE __device__ bool intersectCloserSBVH(
    const idaten::context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    const float t_max,
    bool enableLod/*= false*/,
    int32_t depth/*= -1*/)
{
    float t_min = AT_MATH_EPSILON;

    bool isHit = intersectSBVH<idaten::IntersectType::Closer>(
        ctxt,
        r,
        t_min, t_max,
        isect,
        enableLod,
        depth);

    return isHit;
}

AT_CUDA_INLINE __device__ bool intersectAnySBVH(
    const idaten::context* ctxt,
    const aten::ray& r,
    aten::Intersection* isect,
    bool enableLod/*= false*/,
    int32_t depth/*= -1*/)
{
    float t_min = AT_MATH_EPSILON;
    float t_max = AT_MATH_INF;

    bool isHit = intersectSBVH<idaten::IntersectType::Any>(
        ctxt,
        r,
        t_min, t_max,
        isect,
        enableLod,
        depth);

    return isHit;
}
