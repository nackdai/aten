#include "accelerator/sbvh.h"

#include <omp.h>

#include <algorithm>
#include <iterator>
#include <numeric>

//#pragma optimize( "", off)

namespace aten
{
	void sbvh::makeTreelet()
	{
		static const int VoxelDepth = 3;
		static const real VoxelVolumeThreshold = real(0.5);

		int stack[128] = { 0 };

		aabb wholeBox = m_nodes[0].bbox;

		// NOTE
		// rootノードは対象外.
		for (size_t i = 1; i < m_nodes.size(); i++) {
			auto* node = &m_nodes[i];

			bool isTreeletRoot = (((node->depth % VoxelDepth) == 0) && !node->isLeaf());

			if (isTreeletRoot) {
				auto ratio = wholeBox.computeRatio(node->bbox);

				std::map<uint32_t, SBVHNode*> treeletRoots;

				if (ratio < VoxelVolumeThreshold) {
					if (!node->isTreeletRoot) {
						treeletRoots.insert(std::pair<uint32_t, SBVHNode*>(i, node));
						node->isTreeletRoot = true;
					}
				}
				else {
					stack[0] = node->left;
					stack[1] = node->right;
					int stackpos = 2;

					bool enableTraverseToChild = true;
					
					while (stackpos > 0) {
						auto idx = stack[stackpos - 1];
						stackpos -= 1;

						if (idx < 0) {
							return;
						}

						if (idx == node->right) {
							// 分岐点に戻ったので、子供の探索を許す.
							enableTraverseToChild = true;
						}

						auto* n = &m_nodes[idx];

						if (!n->isLeaf()) {
							ratio = wholeBox.computeRatio(n->bbox);
							if (ratio < VoxelVolumeThreshold) {
								if (!n->isTreeletRoot) {
									treeletRoots.insert(std::pair<uint32_t, SBVHNode*>(idx, n));
									n->isTreeletRoot = true;

									// Treeletのルート候補は見つけたので、これ以上は子供を探索しない.
									enableTraverseToChild = false;
								}
							}
							else if (enableTraverseToChild) {
								stack[stackpos++] = n->left;
								stack[stackpos++] = n->right;
							}
						}
					}
				}

				for (auto it = treeletRoots.begin(); it != treeletRoots.end(); it++) {
					auto idx = it->first;
					auto node = it->second;

					onMakeTreelet(idx, *node);
				}
			}
		}
	}

	void sbvh::onMakeTreelet(
		uint32_t idx,
		const sbvh::SBVHNode& root)
	{
		m_treelets.push_back(SbvhTreelet());
		auto& treelet = m_treelets[m_treelets.size() - 1];

		treelet.idxInBvhTree = idx;
		treelet.depth = root.depth;

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
#if 1
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
#else
		auto f1 = v1 - p;
		auto f2 = v2 - p;
		auto f3 = v3 - p;
		auto c = cross(v2 - v1, v3 - v1);
		float area = length(c);
		lambda1 = length(cross(f2, f3));
		lambda2 = length(cross(f3, f1));

		lambda1 /= area;
		lambda2 /= area;
#endif

		return lambda1 >= 0.0f && lambda2 >= 0.0f && lambda1 + lambda2 <= 1.0f;
	}

	void sbvh::buildVoxel(
		uint32_t exid,
		uint32_t offset)
	{
		const auto& faces = aten::face::faces();
		const auto& vertices = aten::VertexManager::getVertices();
		const auto& mtrls = aten::material::getMaterials();

		for (size_t i = 0; i < m_treelets.size(); i++) {
			const auto& treelet = m_treelets[i];

			auto& sbvhNode = m_nodes[treelet.idxInBvhTree];

			// Speficfy having voxel.
			sbvhNode.voxelIdx = i + offset;

			auto center = sbvhNode.bbox.getCenter();

			BvhVoxel voxel;
			{
				voxel.normal = aten::vec3(0);
				voxel.color = aten::vec3(0);
			}

			voxel.nodeid = treelet.idxInBvhTree;
			voxel.exid = exid;
			voxel.depth = sbvhNode.depth;

			for (const auto tid : treelet.tris) {
				const auto triparam = faces[tid]->param;

				const auto& v0 = vertices[triparam.idx[0]];
				const auto& v1 = vertices[triparam.idx[1]];
				const auto& v2 = vertices[triparam.idx[2]];

				float lambda1, lambda2;

				if (!barycentric(v0.pos, v1.pos, v2.pos, center, lambda1, lambda2)) {
					lambda1 = std::min<float>(std::max<float>(lambda1, 0.0f), 1.0f);
					lambda2 = std::min<float>(std::max<float>(lambda2, 0.0f), 1.0f);
					float tau = lambda1 + lambda2;
					if (tau > 1.0f) {
						lambda1 /= tau;
						lambda2 /= tau;
					}
				}

				float lambda3 = 1.0f - lambda1 - lambda2;

				auto normal = v0.nml * lambda1 + v1.nml * lambda2 + v2.nml * lambda3;
				if (triparam.needNormal > 0) {
					auto e01 = v1.pos - v0.pos;
					auto e02 = v2.pos - v0.pos;

					e01.w = e02.w = real(0);

					normal = normalize(cross(e01, e02));
				}

				auto uv = v0.uv * lambda1 + v1.uv * lambda2 + v2.uv * lambda3;

				const auto mtrl = mtrls[triparam.mtrlid];
				auto color = mtrl->sampleAlbedoMap(uv.x, uv.y);
				color *= mtrl->color();

				voxel.normal += normal;
				voxel.color += color;
			}

			int cnt = (int)treelet.tris.size();

			voxel.normal /= cnt;
			voxel.color /= cnt;

			voxel.normal = normalize(voxel.normal);

			m_voxels.push_back(voxel);
		}
	}
}
