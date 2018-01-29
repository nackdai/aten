#include "visualizer/GeomDataBuffer.h"
#include "visualizer/atengl.h"

namespace aten {
	void GeomVertexBuffer::init(
		uint32_t stride,
		uint32_t vtxNum,
		uint32_t offset,
		const void* data)
	{
		CALL_GL_API(::glGenBuffers(1, &m_vbo));

		auto size = stride * vtxNum;

		m_vtxStride = stride;
		m_vtxNum = vtxNum;
		m_vtxOffset = offset;

		m_initVtxNum = vtxNum;

		CALL_GL_API(::glBindBuffer(GL_ARRAY_BUFFER, m_vbo));

		CALL_GL_API(::glBufferData(
			GL_ARRAY_BUFFER,
			size,
			data,
			GL_STATIC_DRAW));

		// VAO
		{
			CALL_GL_API(::glGenVertexArrays(1, &m_vao));

			CALL_GL_API(::glBindVertexArray(m_vao));
			CALL_GL_API(::glBindBuffer(GL_ARRAY_BUFFER, m_vbo));

			// TODO
			// Fix vertex structure
			//  float4 pos
			//  float3 nml
			//  float3 uv

			static const struct VertexAttrib {
				GLenum type;
				int num;
				int size;
				int offset;
			} attribs[] = {
				{ GL_FLOAT, 3, sizeof(GLfloat), 0 },
				{ GL_FLOAT, 3, sizeof(GLfloat), 16 },
				{ GL_FLOAT, 2, sizeof(GLfloat), 28 },
			};

			auto offsetByte = offset * m_vtxStride;

			for (int i = 0; i < 3; i++) {
				CALL_GL_API(::glEnableVertexAttribArray(i));

				CALL_GL_API(::glVertexAttribPointer(
					i,
					attribs[i].num,
					attribs[i].type,
					GL_FALSE,
					m_vtxStride,
					(void*)(offset + attribs[i].offset)));
			}
		}
	}

	void GeomVertexBuffer::update(
		uint32_t vtxNum,
		const void* data)
	{
		AT_ASSERT(m_vbo > 0);
		AT_ASSERT(vtxNum <= m_initVtxNum);

		auto size = m_vtxStride * vtxNum;

		m_vtxNum = vtxNum;

		CALL_GL_API(::glNamedBufferSubData(
			m_vbo,
			(GLintptr)0,
			size,
			data));
	}

	static GLenum prims[] = {
		GL_TRIANGLES,
		GL_LINES,
	};

	inline uint32_t computeVtxNum(Primitive mode, uint32_t primNum)
	{
		uint32_t vtxNum = 0;

		switch (mode)
		{
		case Primitive::Triangles:
			vtxNum = primNum * 3;
			break;
		case Primitive::Lines:
			vtxNum = primNum * 2;
			break;
		}

		return vtxNum;
	}

	void GeomVertexBuffer::draw(
		Primitive mode,
		uint32_t idxOffset,
		uint32_t primNum)
	{
		CALL_GL_API(::glBindVertexArray(m_vao));

		auto vtxNum = computeVtxNum(mode, primNum);

		CALL_GL_API(::glDrawArrays(prims[mode], idxOffset, vtxNum));
	}

	//////////////////////////////////////////////////////////

	void GeomIndexBuffer::init(
		uint32_t idxNum,
		const void* data)
	{
		CALL_GL_API(::glGenBuffers(1, &m_ibo));

		auto size = sizeof(GLuint) * idxNum;

		m_idxNum = idxNum;

		m_initIdxNum = idxNum;

		CALL_GL_API(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo));

		CALL_GL_API(::glBufferData(
			GL_ELEMENT_ARRAY_BUFFER,
			size,
			data,
			GL_STATIC_DRAW));
	}

	void GeomIndexBuffer::update(
		uint32_t idxNum,
		const void* data)
	{
		AT_ASSERT(m_ibo > 0);
		AT_ASSERT(idxNum <= m_initIdxNum);

		auto size = sizeof(GLuint) * idxNum;

		m_idxNum = idxNum;

		if (size > 0) {
			CALL_GL_API(::glNamedBufferSubData(
				m_ibo,
				(GLintptr)0,
				size,
				data));
		}
	}

	void GeomIndexBuffer::lock(void** dst)
	{
		void* tmp = nullptr;

		auto lockSize = sizeof(GLuint) * m_idxNum;

		CALL_GL_API(tmp = ::glMapNamedBufferRange(
			m_ibo,
			0,
			lockSize,
			GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));

		*dst = tmp;

		m_isLockedIBO = true;
	}

	void GeomIndexBuffer::unlock()
	{
		if (m_isLockedIBO) {
			CALL_GL_API(::glUnmapNamedBuffer(m_ibo));
		}

		m_isLockedIBO = false;
	}

	void GeomIndexBuffer::draw(
		GeomVertexBuffer& vb,
		Primitive mode,
		uint32_t idxOffset,
		uint32_t primNum)
	{
		CALL_GL_API(::glBindVertexArray(vb.m_vao));
		CALL_GL_API(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo));

		auto offsetByte = idxOffset * sizeof(GLuint);

		auto idxNum = computeVtxNum(mode, primNum);

		CALL_GL_API(::glDrawElements(
			prims[mode],
			idxNum,
			GL_UNSIGNED_INT,
			(const GLvoid*)offsetByte));
	}
}
