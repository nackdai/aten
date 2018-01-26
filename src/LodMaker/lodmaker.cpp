#include "lodmaker.h"

struct QVertex {
	aten::vertex v;
	uint32_t idx;
	uint32_t grid[3];
	uint64_t hash;

	QVertex() {}

	QVertex(
		aten::vertex _v, 
		uint32_t i, 
		uint64_t h,
		uint32_t gx, uint32_t gy, uint32_t gz)
		: v(_v), idx(i), hash(h)
	{
		grid[0] = gx;
		grid[1] = gy;
		grid[2] = gz;
	}

	const QVertex& operator=(const QVertex& rhs)
	{
		v = rhs.v;
		idx = rhs.idx;
		grid[0] = rhs.grid[0];
		grid[1] = rhs.grid[1];
		grid[2] = rhs.grid[2];
		hash = rhs.hash;
	}
};

// NOTE
// gridX * gridY * gridZ が uint64_t を超えないようにする

void LodMaker::make(
	std::vector<aten::vertex>& dstVertices,
	std::vector<std::vector<int>>& dstIndices,
	const aten::aabb& bound,
	const std::vector<aten::vertex>& vertices,
	const std::vector<std::vector<aten::face*>>& triGroups,
	int gridX,
	int gridY,
	int gridZ)
{
	auto bmin = bound.minPos();
	auto range = bound.size();

	aten::vec3 scale(
		(gridX - 1) / range.x,
		(gridY - 1) / range.y,
		(gridZ - 1) / range.z);

	std::vector<std::vector<QVertex>> qvtxs(triGroups.size());
	std::vector<std::vector<uint32_t>> sortedIndices(triGroups.size());

	for (uint32_t i = 0; i < triGroups.size(); i++) {
		const auto tris = triGroups[i];

		qvtxs[i].reserve(tris.size());

		for (uint32_t n = 0; n < tris.size(); n++) {
			const auto tri = tris[n];

			for (int t = 0; t < 3; t++) {
				uint32_t idx = tri->param.idx[t];
				const auto& v = vertices[idx];

				auto grid = ((aten::vec3)v.pos - bmin) * scale + real(0.5);

				uint32_t gx = (uint32_t)grid.x;
				uint32_t gy = (uint32_t)grid.y;
				uint32_t gz = (uint32_t)grid.z;

				uint64_t hash = gz * (gridX * gridY) + gy * gridX + gx;

				qvtxs[i].push_back(QVertex(v, idx, hash, gx, gy, gz));
			}
		}
	}

	// 同じグリッドに入っている頂点の順になるようにソートする.

	for (uint32_t i = 0; i < qvtxs.size(); i++)
	{
		std::sort(
			qvtxs[i].begin(), qvtxs[i].end(),
			[](const QVertex& q0, const QVertex& q1)
		{
			return q0.hash > q1.hash;
		});

		sortedIndices.reserve(triGroups[i].size());

		uint32_t num = (uint32_t)qvtxs[i].size();

		// インデックスも頂点にあわせて並べ替える.
		for (uint32_t n = 0; n < num; n++) {
			const auto& q = qvtxs[i][n];

			// 元々のインデックスの位置に新しいインデックス値を入れる.
			sortedIndices[i][q.idx] = n;
		}
	}

	// 同じグリッド内に入っている頂点の平均値を計算して、１つの頂点にしてしまう.
	// UVは崩れるが、所詮セカンダリバウンスに使うので、そこは気にしない.

	for (uint32_t i = 0; i < qvtxs.size(); i++)
	{
		auto start = qvtxs[i].begin();

		while (start != qvtxs[i].end())
		{
			auto end = start;

			// グリッドが異なる頂点になるまで探す.
			while (end != qvtxs[i].end() && start->hash == end->hash) {
				end++;
			}

			// TODO
			// 平均計算.

			for (auto q = start; q != end; q++) {
				if (q == start) {
					dstVertices.push_back(q->v);
				}
				else {
					// 異なる位置の点の場合.

					// 前の点と比較する.
					auto prev = q - 1;

					auto v0 = q->v.pos;
					auto v1 = prev->v.pos;

					bool isEqual = (memcmp(&v0, &v1, sizeof(v0)) == 0);

					if (!isEqual) {
						dstVertices.push_back(q->v);
					}
				}

				// インデックス更新.
				q->idx = (uint32_t)dstVertices.size() - 1;
			}

			// 次の開始位置を更新
			start = end;
		}
	}

	// LODされた結果のインデックスを格納.

	dstIndices.resize(triGroups.size());

	for (uint32_t i = 0; i < triGroups.size(); i++) {
		const auto tris = triGroups[i];

		dstIndices[i].reserve(tris.size() * 3);

		for (uint32_t n = 0; n < tris.size(); n++) {
			const auto tri = tris[n];

			for (int t = 0; t < 3; t++) {
				uint32_t idx = tri->param.idx[t];

				auto sortedIdx = sortedIndices[i][idx];
				auto newIdx = qvtxs[i][sortedIdx].idx;

				dstIndices[i].push_back(newIdx);
			}
		}
	}
}
