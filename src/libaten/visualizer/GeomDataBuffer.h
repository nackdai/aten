#pragma once

#include "types.h"

namespace aten {
	class GeomVertexBuffer {
		friend class GeomIndexBuffer;

	public:
		GeomVertexBuffer() {}
		virtual ~GeomVertexBuffer() {}

	public:
		void init(
			uint32_t stride,
			uint32_t vtxNum,
			uint32_t offset,
			void* data);

	protected:
		uint32_t m_vbo{ 0 };
		uint32_t m_vao{ 0 };

		uint32_t m_vtxStride{ 0 };
		uint32_t m_vtxNum{ 0 };
		uint32_t m_vtxOffset{ 0 };
	};

	class GeomIndexBuffer {
	public:
		GeomIndexBuffer() {}
		virtual ~GeomIndexBuffer() {}

	public:
		void init(
			uint32_t idxNum,
			void* data);

		void lock(void** dst);
		void unlock();

		void draw(
			GeomVertexBuffer& vb,
			uint32_t idxOffset,
			uint32_t primNum);

	protected:
		uint32_t m_ibo{ 0 };

		uint32_t m_idxNum{ 0 };

		bool m_isLockedIBO{ false };
	};
}