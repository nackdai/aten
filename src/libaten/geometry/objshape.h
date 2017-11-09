#pragma once

#include <atomic>

#include "types.h"
#include "material/material.h"
#include "math/mat4.h"
#include "geometry/face.h"
#include "visualizer/GeomDataBuffer.h"

namespace AT_NAME
{
	class objshape : public aten::geombase {
		friend class object;

	public:
		objshape() : param(aten::GeometryType::Polygon) {}
		virtual ~objshape() {}

		void build();

		void setMaterial(material* mtrl)
		{
			param.mtrl.ptr = mtrl;
			m_mtrl = mtrl;
		}

		const material* getMaterial() const
		{
			return m_mtrl;
		}

		void addFace(face* f);

		void draw(
			aten::hitable::FuncPreDraw func,
			const aten::mat4& mtxL2W,
			int parentId);

		aten::GeomParameter param;
		aten::aabb m_aabb;

	private:
		material* m_mtrl{ nullptr };
		std::vector<face*> faces;

		aten::GeomIndexBuffer m_ib;

		int m_baseIdx{ INT32_MAX };
		int m_baseTriIdx{ INT32_MAX };
	};
}
