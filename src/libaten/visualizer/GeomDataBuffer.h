#pragma once

#include "types.h"

namespace aten {
	class shader;

	enum Primitive {
		Triangles,
		Lines,
	};

	struct VertexAttrib {
		int type;
		int num;
		int size;
		int offset;
		bool needNormalize{ false };
		const char* name{ nullptr };

		VertexAttrib() {}
		VertexAttrib(int t, int n, int s, int o)
			: type(t), num(n), size(s), offset(o)
		{}
	};

	class GeomVertexBuffer {
		friend class GeomIndexBuffer;

	public:
		GeomVertexBuffer() {}
		virtual ~GeomVertexBuffer()
		{
			clear();
		}

	public:
		void init(
			uint32_t stride,
			uint32_t vtxNum,
			uint32_t offset,
			const void* data,
			bool isDynamic = false);

		void init(
			uint32_t stride,
			uint32_t vtxNum,
			uint32_t offset,
			const VertexAttrib* attribs,
			uint32_t attribNum,
			const void* data,
			bool isDynamic = false);

		void initNoVAO(
			uint32_t stride,
			uint32_t vtxNum,
			uint32_t offset,
			const void* data);

		void createVAOByAttribName(
			const shader* shd,
			const VertexAttrib* attribs,
			uint32_t attribNum);

		void update(
			uint32_t vtxNum,
			const void* data);

		void draw(
			Primitive mode,
			uint32_t idxOffset,
			uint32_t primNum);

		void* beginRead()
		{
			return beginMap(true);
		}
		void endRead()
		{
			endMap();
		}

		void* beginWrite()
		{
			return beginMap(false);
		}
		void endWrite()
		{
			endMap();
		}

		void clear();

		uint32_t getVtxNum() const
		{
			return m_vtxNum;
		}

		uint32_t getStride() const
		{
			return m_vtxStride;
		}

		uint32_t getVBOHandle() const
		{
			return m_vbo;
		}

	private:
		void* beginMap(bool isRead);
		void endMap();

	protected:
		uint32_t m_vbo{ 0 };
		uint32_t m_vao{ 0 };

		uint32_t m_vtxStride{ 0 };
		uint32_t m_vtxNum{ 0 };
		uint32_t m_vtxOffset{ 0 };

		uint32_t m_initVtxNum{ 0 };

		bool m_isMapping{ false };
	};

	//////////////////////////////////////////////////////////

	class GeomIndexBuffer {
	public:
		GeomIndexBuffer() {}
		virtual ~GeomIndexBuffer();

	public:
		void init(
			uint32_t idxNum,
			const void* data);

		void update(
			uint32_t idxNum,
			const void* data);

		void lock(void** dst);
		void unlock();

		void draw(
			GeomVertexBuffer& vb,
			Primitive mode,
			uint32_t idxOffset,
			uint32_t primNum);

	protected:
		uint32_t m_ibo{ 0 };

		uint32_t m_idxNum{ 0 };

		bool m_isLockedIBO{ false };

		uint32_t m_initIdxNum{ 0 };
	};
}