#include "kernel/Skinning.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda/helper_math.h"
#include "cuda/cudautil.h"

__global__ void computeSkinning(
	uint32_t indexNum,
	const idaten::Skinning::Vertex* __restrict__ vertices,
	const uint32_t* __restrict__ indices,
	const aten::mat4* __restrict__ matrices,
	aten::vertex* dst)
{
	const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= indexNum) {
		return;
	}

	const auto vtxIdx = indices[idx];
	const auto* vtx = &vertices[vtxIdx];

	aten::vec4 srcPos = vtx->position;
	aten::vec4 srcNml = aten::vec4(vtx->normal, 0);

	aten::vec4 dstPos(0);
	aten::vec4 dstNml(0);

	for (int i = 0; i < 4; i++) {
		int idx = int(vtx->blendIndex[i]);
		float weight = vtx->blendWeight[i];

		// TODO
		// DeformPrimitives単位で０ベースのインデックス値になっているので、通しインデックスに変換する必要がある.
		aten::mat4 mtx = matrices[idx];

		dstPos += weight * mtx * vtx->position;
		dstNml += weight * mtx * srcNml;
	}

	dstNml = normalize(dstNml);

	dst[idx].pos = aten::vec4(dstPos.x, dstPos.y, dstPos.z, 1);
	dst[idx].nml = dstNml;
	dst[idx].uv = aten::vec3(vtx->uv[0], vtx->uv[1], 0);
}

namespace idaten
{
	void Skinning::init(
		Skinning::Vertex* vertices,
		uint32_t vtxNum,
		uint32_t* indices,
		uint32_t idxNum)
	{
		m_vertices.init(vtxNum);
		m_vertices.writeByNum(vertices, vtxNum);

		m_indices.init(idxNum);
		m_indices.writeByNum(indices, idxNum);
	}

	void Skinning::update(
		aten::mat4* matrices,
		uint32_t mtxNum)
	{
		if (m_matrices.bytes() == 0) {
			m_matrices.init(mtxNum);
		}

		AT_ASSERT(m_matrices.maxNum() >= mtxNum);

		m_matrices.writeByNum(matrices, mtxNum);
	}

	void Skinning::compute()
	{
		const auto idxNum = m_indices.maxNum();
	}
}