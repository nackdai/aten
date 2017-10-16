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

	public:
		int getMeshId() const
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

	public:
		virtual int meshid() const override
		{
			return m_meshid;
		}
	};
}
