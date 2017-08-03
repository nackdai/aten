#pragma once

#include "types.h"
#include "shape/tranformable.h"
#include "accelerator/bvh.h"
#include "math/mat4.h"

#include <atomic>

namespace aten
{
	class meshbase {
	protected:
		static std::atomic<int> g_id;

		meshbase();
		virtual ~meshbase() {}

		int meshid() const
		{
			return m_meshid;
		}

	protected:
		int m_meshid{ -1 };
	};

	template <typename INHERIT>
	class mesh : public INHERIT, public meshbase {
	protected:
		mesh() {}
		virtual ~mesh() {}

		virtual bool setBVHNodeParam(
			BVHNode& param,
			const bvhnode* parent,
			const int idx,
			std::vector<std::vector<BVHNode>>& nodes,
			const transformable* instanceParent,
			const aten::mat4& mtxL2W) override final
		{
			bool ret = INHERIT::setBVHNodeParam(param, parent, idx, nodes, instanceParent, mtxL2W);

			param.meshid = (float)meshid();

			return ret;
		}
	};
}
