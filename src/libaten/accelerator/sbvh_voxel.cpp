#include <algorithm>
#include <iterator>
#include <numeric>

#include <omp.h>

#include "accelerator/sbvh.h"

//#pragma optimize( "", off)

namespace aten
{
    void sbvh::makeTreelet()
    {
        if (m_isImported) {
            // Imported already.
            return;
        }

        // NOTE
        // rootÉmÅ[ÉhÇÕëŒè€äO.

        // Find the node for treelet root.
        for (uint32_t i = 1; i < (uint32_t)m_nodes.size(); i++) {
            auto* node = &m_nodes[i];

            // Check if the node is treelet root.
            bool isTreeletRoot = (((node->depth % VoxelDepth) == 0) && !node->isLeaf());

            if (isTreeletRoot) {
                node->isTreeletRoot = true;

                // Make treelet from thd found treelet node.
                onMakeTreelet(i, *node);
            }
        }
    }

    void sbvh::onMakeTreelet(
        uint32_t idx,
        const sbvh::SBVHNode& root)
    {
        auto it = m_treelets.insert(std::make_pair(idx, SbvhTreelet()));
        auto& treelet = it.first->second;

        treelet.idxInBvhTree = idx;

        int stack[128] = { 0 };

        stack[0] = root.left;
        stack[1] = root.right;
        int stackpos = 2;

        while (stackpos > 0) {
            int idx = stack[stackpos - 1];
            stackpos -= 1;

            const auto& sbvhNode = m_nodes[idx];

            if (sbvhNode.isLeaf()) {
                treelet.leafChildren.push_back(idx);

                const auto refid = sbvhNode.refIds[0];
                const auto& ref = m_refs[refid];
                const auto triid = ref.triid + m_offsetTriIdx;
                treelet.tris.push_back(triid);
            }
            else {
                stack[stackpos++] = sbvhNode.right;
                stack[stackpos++] = sbvhNode.left;
            }
        }
    }

    bool barycentric(
        const aten::vec3& v1,
        const aten::vec3& v2,
        const aten::vec3& v3,
        const aten::vec3& p, float &lambda1, float &lambda2)
    {
        // NOTE
        // https://blogs.msdn.microsoft.com/rezanour/2011/08/07/barycentric-coordinates-and-point-in-triangle-tests/

        // Prepare our barycentric variables
        auto u = v2 - v1;
        auto v = v3 - v1;
        auto w = p - v1;

        auto vCrossW = cross(v, w);
        auto vCrossU = cross(v, u);

        auto uCrossW = cross(u, w);
        auto uCrossV = cross(u, v);

        // At this point, we know that r and t and both > 0.
        // Therefore, as long as their sum is <= 1, each must be less <= 1
        float denom = length(uCrossV);
        lambda1 = length(vCrossW) / denom;
        lambda2 = length(uCrossW) / denom;

        return lambda1 >= 0.0f && lambda2 >= 0.0f && lambda1 + lambda2 <= 1.0f;
    }

    void sbvh::buildVoxel(const context& ctxt)
    {
        const auto& faces = aten::face::faces();
        const auto& vertices = ctxt.getVertices();

        for (auto it = m_treelets.begin(); it != m_treelets.end(); it++) {
            auto& treelet = it->second;

            treelet.enabled = true;

            auto& sbvhNode = m_nodes[treelet.idxInBvhTree];

            std::map<int, real> mtrlMap;

            for (const auto tid : treelet.tris) {
                const auto& triparam = faces[tid]->getParam();

                auto found = mtrlMap.find(triparam.mtrlid);

                if (found != mtrlMap.end()) {
                    found->second += triparam.area;
                }
                else {
                    mtrlMap.insert(std::make_pair(triparam.mtrlid, triparam.area));
                }
            }

            int mtrlCandidateId = -1;
            real maxArea = real(-1);

            for (auto it : mtrlMap) {
                auto mtrlid = it.first;
                auto area = it.second;

                if (area >= maxArea) {
                    maxArea = area;
                    mtrlCandidateId = mtrlid;
                }
            }

            AT_ASSERT(mtrlCandidateId >= 0);

            treelet.mtrlid = mtrlCandidateId;
        }
    }
}
