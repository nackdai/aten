#include "geometry/objshape.h"
#include "math/intersect.h"
#include "accelerator/accelerator.h"
#include "geometry/vertex.h"

#include <iterator>

namespace AT_NAME
{
	void objshape::build()
	{
		aten::vec3 boxmin(AT_MATH_INF, AT_MATH_INF, AT_MATH_INF);
		aten::vec3 boxmax(-AT_MATH_INF, -AT_MATH_INF, -AT_MATH_INF);

		param.area = 0;

		for (const auto f : faces) {
			f->build(this);
			param.area += f->param.area;

			const auto& faabb = f->getBoundingbox();

			boxmin = aten::min(faabb.minPos(), boxmin);
			boxmax = aten::max(faabb.maxPos(), boxmax);
		}

		m_aabb.init(boxmin, boxmax);

		// For rasterize rendering.
		{
			std::vector<uint32_t> idx;
			idx.reserve(faces.size() * 3);

			for (const auto f : faces) {
				idx.push_back(f->param.idx[0]);
				idx.push_back(f->param.idx[1]);
				idx.push_back(f->param.idx[2]);
			}

			m_ib.init((uint32_t)idx.size(), &idx[0]);
		}
	}

	void objshape::addFace(face* f)
	{
		int idx0 = f->param.idx[0];
		int idx1 = f->param.idx[1];
		int idx2 = f->param.idx[2];

		int baseIdx = std::min(idx0, std::min(idx1, idx2));
		m_baseIdx = std::min(baseIdx, m_baseIdx);

		faces.push_back(f);

		m_baseTriIdx = std::min(f->id, m_baseTriIdx);
	}

	void objshape::draw(
		aten::hitable::FuncPreDraw func,
		const aten::mat4& mtxL2W,
		int parentId)
	{
		if (func) {
			func(mtxL2W, parentId, m_baseTriIdx);
		}

		auto vb = VertexManager::getVB();

		auto triNum = (uint32_t)faces.size();

		m_ib.draw(vb, 0, triNum);
	}
}
