#include "visualizer/GeomDataBuffer.h"
#include "visualizer/shader.h"
#include "visualizer/atengl.h"

namespace aten {
    void GeomVertexBuffer::init(
        size_t stride,
        uint32_t vtxNum,
        uint32_t offset,
        const void* data,
        bool isDynamic/*= false*/)
    {
        // Fix vertex structure
        //  float4 pos
        //  float3 nml
        //  float3 uv

        static const VertexAttrib attribs[] = {
            VertexAttrib(GL_FLOAT, 3, sizeof(GLfloat), 0),
            VertexAttrib(GL_FLOAT, 3, sizeof(GLfloat), 16),
            VertexAttrib(GL_FLOAT, 2, sizeof(GLfloat), 28),
        };

        init(
            stride,
            vtxNum,
            offset,
            attribs,
            AT_COUNTOF(attribs),
            data,
            isDynamic);
    }

    void GeomVertexBuffer::init(
        size_t stride,
        uint32_t vtxNum,
        uint32_t offset,
        const VertexAttrib* attribs,
        uint32_t attribNum,
        const void* data,
        bool isDynamic/*= false*/)
    {
        AT_ASSERT(m_vbo == 0);
        AT_ASSERT(m_vao == 0);

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
            isDynamic ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW));

        // VAO
        {
            CALL_GL_API(::glGenVertexArrays(1, &m_vao));

            CALL_GL_API(::glBindVertexArray(m_vao));
            CALL_GL_API(::glBindBuffer(GL_ARRAY_BUFFER, m_vbo));

            auto offsetByte = offset * m_vtxStride;

            for (uint32_t i = 0; i < attribNum; i++) {
                CALL_GL_API(::glEnableVertexAttribArray(i));

                CALL_GL_API(::glVertexAttribPointer(
                    i,
                    attribs[i].num,
                    attribs[i].type,
                    GL_FALSE,
                    static_cast<GLsizei>(m_vtxStride),
                    reinterpret_cast<void*>(offset + attribs[i].offset)));
            }
        }
    }

    void GeomVertexBuffer::initNoVAO(
        size_t stride,
        uint32_t vtxNum,
        uint32_t offset,
        const void* data)
    {
        AT_ASSERT(m_vbo == 0);

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
    }

    void GeomVertexBuffer::createVAOByAttribName(
        const shader* shd,
        const VertexAttrib* attribs,
        uint32_t attribNum)
    {
        AT_ASSERT(m_vbo > 0);
        AT_ASSERT(m_vao == 0);

        CALL_GL_API(::glGenVertexArrays(1, &m_vao));

        CALL_GL_API(::glBindVertexArray(m_vao));
        CALL_GL_API(::glBindBuffer(GL_ARRAY_BUFFER, m_vbo));

        auto offsetByte = m_vtxOffset * m_vtxStride;

        auto program = shd->getProgramHandle();

        for (uint32_t i = 0; i < attribNum; i++) {
#if 0
            CALL_GL_API(::glEnableVertexAttribArray(i));

            CALL_GL_API(::glVertexAttribPointer(
                i,
                attribs[i].num,
                attribs[i].type,
                GL_FALSE,
                m_vtxStride,
                reinterpret_cast<void*>(m_vtxOffset + attribs[i].offset)));
#else
            if (attribs[i].name) {
                GLint loc = -1;
                CALL_GL_API(loc = ::glGetAttribLocation(program, attribs[i].name));

                if (loc >= 0) {
                    CALL_GL_API(::glEnableVertexAttribArray(loc));

                    CALL_GL_API(::glVertexAttribPointer(
                        loc,
                        attribs[i].num,
                        attribs[i].type,
                        attribs[i].needNormalize ? GL_TRUE : GL_FALSE,
                        static_cast<GLsizei>(m_vtxStride),
                        reinterpret_cast<void*>(m_vtxOffset + attribs[i].offset)));
                }
            }
#endif


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

    static constexpr GLenum prims[] = {
        GL_TRIANGLES,
        GL_LINES,
        GL_POINTS,
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
        case Primitive::Points:
            vtxNum = primNum;
            break;
        default:
            AT_ASSERT(false);
            break;
        }

        return vtxNum;
    }

    void GeomVertexBuffer::draw(
        Primitive mode,
        uint32_t idxOffset,
        uint32_t primNum)
    {
        AT_ASSERT(m_vao > 0);

        CALL_GL_API(::glBindVertexArray(m_vao));

        auto vtxNum = computeVtxNum(mode, primNum);

        const int prim_idx = static_cast<int>(mode);

        CALL_GL_API(::glDrawArrays(prims[prim_idx], idxOffset, vtxNum));
    }

    void* GeomVertexBuffer::beginMap(bool isRead)
    {
        AT_ASSERT(m_vbo > 0);
        AT_ASSERT(!m_isMapping);

        void* ret = nullptr;

        if (!m_isMapping) {
            CALL_GL_API(ret = ::glMapNamedBuffer(m_vbo, isRead ? GL_READ_ONLY : GL_WRITE_ONLY));
            m_isMapping = true;
        }

        return ret;
    }

    void GeomVertexBuffer::endMap()
    {
        AT_ASSERT(m_vbo > 0);
        AT_ASSERT(m_isMapping);

        if (m_isMapping) {
            CALL_GL_API(::glUnmapNamedBuffer(m_vbo));
            m_isMapping = false;
        }
    }

    void GeomVertexBuffer::clear()
    {
        if (m_vbo > 0) {
            CALL_GL_API(::glDeleteBuffers(1, &m_vbo));
        }
        if (m_vao > 0) {
            CALL_GL_API(::glDeleteVertexArrays(1, &m_vao));
        }

        m_vbo = 0;
        m_vao = 0;
    }

    //////////////////////////////////////////////////////////

    static inline uint32_t getElementSize(const VertexAttrib& attrib)
    {
        uint32_t ret = 0;

        switch (attrib.type)
        {
        case GL_FLOAT:
            ret = sizeof(float) * attrib.num;
            break;
        case GL_BYTE:
            ret = sizeof(GLbyte) * attrib.num;
            break;
        default:
            AT_ASSERT(false);
            break;
        }

        return ret;
    }

    void GeomMultiVertexBuffer::init(
        uint32_t vtxNum,
        const VertexAttrib* attribs,
        uint32_t attribNum,
        const void* data[],
        bool isDynamic/*= false*/)
    {
        AT_ASSERT(m_vbos.empty());
        AT_ASSERT(m_vao == 0);

        CALL_GL_API(::glGenVertexArrays(1, &m_vao));
        CALL_GL_API(::glBindVertexArray(m_vao));

        for (uint32_t i = 0; i < attribNum; i++) {
            GLuint vbo = 0;
            CALL_GL_API(::glGenBuffers(1, &vbo));

            AT_ASSERT(vbo > 0);

            m_vbos.push_back(vbo);

            CALL_GL_API(::glBindBuffer(GL_ARRAY_BUFFER, vbo));

            auto elementSize = getElementSize(attribs[i]);
            auto size = elementSize * vtxNum;

            CALL_GL_API(::glBufferData(
                GL_ARRAY_BUFFER,
                size,
                data ? data[i] : nullptr,
                isDynamic ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW));

            CALL_GL_API(::glEnableVertexAttribArray(i));

            CALL_GL_API(::glVertexAttribPointer(
                i,
                attribs[i].num,
                attribs[i].type,
                GL_FALSE,
                elementSize,
                reinterpret_cast<void*>(attribs[i].offset)));
        }

        m_vtxNum = vtxNum;
    }

    void* GeomMultiVertexBuffer::beginMap(bool isRead, uint32_t idx)
    {
        AT_ASSERT(!m_vbos.empty());
        AT_ASSERT(!m_isMapping);

        void* ret = nullptr;

        if (!m_isMapping) {
            CALL_GL_API(ret = ::glMapNamedBuffer(m_vbos[idx], isRead ? GL_READ_ONLY : GL_WRITE_ONLY));
            m_isMapping = true;
        }

        return ret;
    }

    void GeomMultiVertexBuffer::endMap(uint32_t idx)
    {
        AT_ASSERT(!m_vbos.empty());
        AT_ASSERT(m_isMapping);

        if (m_isMapping) {
            CALL_GL_API(::glUnmapNamedBuffer(m_vbos[idx]));
            m_isMapping = false;
        }
    }

    //////////////////////////////////////////////////////////

    GeomIndexBuffer::~GeomIndexBuffer()
    {
        if (m_ibo > 0) {
            CALL_GL_API(::glDeleteBuffers(1, &m_ibo));
        }
    }

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
        const GeomVertexBuffer& vb,
        Primitive mode,
        uint32_t idxOffset,
        uint32_t primNum) const
    {
        draw(vb.m_vao, mode, idxOffset, primNum);
    }

    void GeomIndexBuffer::draw(
        const GeomMultiVertexBuffer& vb,
        Primitive mode,
        uint32_t idxOffset,
        uint32_t primNum) const
    {
        draw(vb.m_vao, mode, idxOffset, primNum);
    }

    void GeomIndexBuffer::draw(
        uint32_t vao,
        Primitive mode,
        uint32_t idxOffset,
        uint32_t primNum) const
    {
        AT_ASSERT(m_ibo > 0);
        AT_ASSERT(vao > 0);

        CALL_GL_API(::glBindVertexArray(vao));
        CALL_GL_API(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo));

        auto offsetByte = idxOffset * sizeof(GLuint);

        auto idxNum = computeVtxNum(mode, primNum);

        const int prim_idx = static_cast<int>(mode);

        CALL_GL_API(::glDrawElements(
            prims[prim_idx],
            idxNum,
            GL_UNSIGNED_INT,
            (const GLvoid*)offsetByte));
    }
}
