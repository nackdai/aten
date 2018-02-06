#include "accelerator/sbvh.h"

#include <omp.h>

#include <algorithm>
#include <iterator>
#include <numeric>

#pragma optimize( "", off)

namespace aten
{
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

	void sbvh::buildVoxel()
	{
		const auto& faces = aten::face::faces();
		const auto& vertices = aten::VertexManager::getVertices();
		const auto& mtrls = aten::material::getMaterials();

		for (size_t i = 0; i < m_treelets.size(); i++) {
			const auto& treelet = m_treelets[i];

			const auto& sbvhNode = m_nodes[treelet.idxInBvhTree];

			auto center = sbvhNode.bbox.getCenter();

			BvhVoxel voxel;
			{
				voxel.normal = aten::vec3(0);
				voxel.color = aten::vec3(0);
			}

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

				voxel.normal += normal;
				voxel.color += color;
			}

			int cnt = (int)treelet.tris.size();

			voxel.normal /= cnt;
			voxel.color /= cnt;

			m_voxels.push_back(voxel);
		}
	}
}
