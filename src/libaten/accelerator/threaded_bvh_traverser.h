#pragma once

#include "defs.h"

#include "accelerator/threaded_bvh.h"
#include "scene/hit_parameter.h"
#include "renderer/pathtracing/pt_params.h"

#ifdef __CUDACC__
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "kernel/device_scene_context.cuh"
#else
#include "scene/host_scene_context.h"
#endif

#include "math/cuda_host_common_math.h"

namespace aten {
    enum IntersectType {
        Closest,
        Closer,
        Any,
    };
}

namespace AT_NAME
{
    // Threaded BVH Traverser.
    // This class can work for both ThreadedBVH and SBVH.
    class ThreadedBvhTraverser {
    private:
        ThreadedBvhTraverser() = default;
        ~ThreadedBvhTraverser() = default;

        ThreadedBvhTraverser(const ThreadedBvhTraverser&) = delete;
        ThreadedBvhTraverser(ThreadedBvhTraverser&&) = delete;
        ThreadedBvhTraverser& operator=(const ThreadedBvhTraverser&) = delete;
        ThreadedBvhTraverser& operator=(ThreadedBvhTraverser&&) = delete;

        template <class ReturnType>
        static inline AT_DEVICE_API ReturnType GetBvhNodes(
            const AT_NAME::context& ctxt,
            int32_t id)
        {
#ifdef __CUDACC__
            return ctxt.GetBvhNodes(id);
#else
            return &ctxt.GetBvhNodes(id);
#endif
        }

        template <class NodesListType>
        static inline AT_DEVICE_API void GetAsThreadedBvhNode(
            aten::ThreadedBvhNode& node,
            int32_t nodeid,
            NodesListType nodes_list
        )
        {
#ifdef __CUDACC__
            // xyz : boxmin, z: hit
            auto node0 = tex1Dfetch<float4>(nodes_list, aten::GPUBvhNodeSize * nodeid + 0);

            // xyz : boxmax, z: miss
            auto node1 = tex1Dfetch<float4>(nodes_list, aten::GPUBvhNodeSize * nodeid + 1);

            // x : object_id, y : primid, z : exid, w : meshid
            auto attrib = tex1Dfetch<float4>(nodes_list, aten::GPUBvhNodeSize * nodeid + 2);

            node.boxmin = aten::vec3(node0.x, node0.y, node0.z);
            node.hit = node0.w;

            node.boxmax = aten::vec3(node1.x, node1.y, node1.z);
            node.miss = node1.w;

            node.object_id = attrib.x;
            node.primid = attrib.y;
            node.exid = attrib.z;
            node.meshid = attrib.w;
#else
            node = *reinterpret_cast<const aten::ThreadedBvhNode*>(&(*nodes_list)[nodeid]);
#endif
        }

        static inline AT_DEVICE_API int32_t float_as_int(float v)
        {
#ifdef __CUDACC__
            return __float_as_int(v);
#else
            return aten::floatAsInt(v);
#endif
        }

    public:
        template <aten::IntersectType Type>
        static AT_DEVICE_API bool Traverse(
            aten::Intersection& isect,
            const AT_NAME::context& ctxt,
            const aten::ray r,
            float t_min, float t_max,
            bool enable_lod,
            int32_t depth
        )
        {
            float hit_t = AT_MATH_INF;

            using BvhNodesTypeBase = decltype(std::declval<AT_NAME::context>().GetBvhNodes(0));
            using BvhNodeType = std::conditional_t<std::is_reference_v<BvhNodesTypeBase>,
                std::add_pointer_t<std::remove_reference_t<BvhNodesTypeBase>>,
                BvhNodesTypeBase>;
            BvhNodeType node_list = GetBvhNodes<BvhNodeType>(ctxt, 0);

            aten::ray transformed_ray = r;

            int32_t nodeid = 0;

            // To memorize object id and mesh id for bottom layer traversal.
            int32_t objid = -1;
            int32_t meshid = -1;

            // To memorize top layer status for returning to the top layer from the bottom layer.
            bool is_traversing_top_layer = true;
            int32_t top_layer_hit = -1;
            int32_t top_layer_miss = -1;

            while (nodeid >= 0)
            {
                bool is_hit = false;

                aten::ThreadedBvhNode node;
                GetAsThreadedBvhNode(node, nodeid, node_list);

                if (node.isLeaf()) {
                    // Leaf.
                    const auto& obj = ctxt.GetObject(static_cast<uint32_t>(node.object_id));

                    if (node.exid >= 0) {
                        // Hit the bottom later bbox.
                        // Start to traverse the bottom layer.
                        if (obj.mtx_id >= 0) {
                            const auto& mtx_W2L = ctxt.GetMatrix(obj.mtx_id + 1);
                            transformed_ray = mtx_W2L.applyRay(r);
                        }
                        else {
                            transformed_ray = r;
                        }

                        // Get index of the bottom layer bvh node list
                        int32_t exid = aten::floatAsInt(node.exid);
                        bool has_lod = AT_BVHNODE_HAS_LOD(exid);
                        exid = has_lod && enable_lod ? AT_BVHNODE_LOD_EXID(exid) : AT_BVHNODE_MAIN_EXID(exid);

                        // Get the node list of the bottom layer bvh.
                        node_list = GetBvhNodes<BvhNodeType>(ctxt, exid);

                        // Memorize object id and mesh id for bottom layer traversal.
                        objid = node.object_id;
                        meshid = node.meshid;

                        // Memorize top layer status for returning to the top layer from the bottom layer.
                        top_layer_hit = static_cast<int32_t>(node.hit);
                        top_layer_miss = static_cast<int32_t>(node.miss);

                        // Set hit to true to enter the bottom layer.
                        is_hit = true;

                        // Start from the root node of the bottom layer.
                        node.hit = 0;

                        is_traversing_top_layer = false;
                    }
                    else if (node.primid >= 0) {
                        // Hit triangle.
                        const auto& prim = ctxt.GetTriangle(node.primid);

                        aten::Intersection isect_tmp;
                        isect_tmp.t = AT_MATH_INF;

                        is_hit = AT_NAME::triangle::hit(prim, ctxt, transformed_ray, &isect_tmp);

                        // Even though hit to the traiangle, check if we can treat it as hit for Any/Closer mode.
                        // If Any, we always treat hit result as the result.
                        // If Closer, we treat hit result as the result only when it's closer than previous hit.
                        bool is_intersect = (Type == aten::IntersectType::Any
                            ? is_hit
                            : isect_tmp.t < isect.t);

                        if (is_intersect) {
                            isect = isect_tmp;
                            isect.objid = objid;
                            isect.triangle_id = static_cast<int32_t>(node.primid);
                            isect.mtrlid = static_cast<int32_t>(prim.mtrlid);

                            isect.meshid = static_cast<int32_t>(prim.mesh_id);
                            isect.meshid = (isect.meshid < 0 ? meshid : isect.meshid);

                            t_max = isect.t;

                            if constexpr (Type == aten::IntersectType::Closer
                                || Type == aten::IntersectType::Any)
                            {
                                return true;
                            }
                        }
                    }
                }
                else {
                    is_hit = aten::aabb::hit(transformed_ray, node.boxmin, node.boxmax, t_min, t_max, &hit_t);
                }

                // Pick next node based on hit/miss.
                if (is_hit) {
                    nodeid = node.hit;
                }
                else {
                    nodeid = node.miss;
                }

                if (nodeid < 0) {
                    // Return to the top layer.
                    nodeid = is_hit ? top_layer_hit : top_layer_miss;
                    top_layer_hit = -1;
                    top_layer_miss = -1;

                    node_list = GetBvhNodes<BvhNodeType>(ctxt, 0);

                    transformed_ray = r;
                    is_traversing_top_layer = true;
                }
            }

            return (isect.objid >= 0);
        }
    };
}
