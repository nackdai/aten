#pragma once

#include <vector>

#include "types.h"
#include "defs.h"

namespace aten {
    class shader;

    enum class Primitive {
        Triangles,
        Lines,
        Points,
    };

    struct VertexAttrib {
        int32_t type{ 0 };
        int32_t num{ 0 };
        int32_t size{ 0 };
        int32_t offset{ 0 };
        bool needNormalize{ false };
        const char* name{ nullptr };

        VertexAttrib() = default;
        VertexAttrib(int32_t t, int32_t n, int32_t s, int32_t o)
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
            size_t stride,
            size_t vtxNum,
            size_t offset,
            const void* data,
            bool isDynamic = false);

        void init(
            size_t stride,
            size_t vtxNum,
            size_t offset,
            const VertexAttrib* attribs,
            size_t attribNum,
            const void* data,
            bool isDynamic = false);

        void initNoVAO(
            size_t stride,
            size_t vtxNum,
            int32_t offset,
            const void* data);

        void createVAOByAttribName(
            const shader* shd,
            const VertexAttrib* attribs,
            size_t attribNum);

        void update(
            size_t vtxNum,
            const void* data);

        void draw(
            Primitive mode,
            int32_t idxOffset,
            size_t primNum);

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

        size_t getVtxNum() const
        {
            return m_vtxNum;
        }

        virtual uint32_t getVBOHandle() const
        {
            return m_vbo;
        }

        virtual bool isInitialized() const
        {
            return m_vbo > 0;
        }

    private:
        void* beginMap(bool isRead);
        void endMap();

    protected:
        uint32_t m_vbo{ 0 };
        uint32_t m_vao{ 0 };

        size_t m_vtxStride{ 0 };
        size_t m_vtxNum{ 0 };
        size_t m_vtxOffset{ 0 };

        size_t m_initVtxNum{ 0 };

        bool m_isMapping{ false };
    };

    //////////////////////////////////////////////////////////

    class GeomMultiVertexBuffer : public GeomVertexBuffer {
        friend class GeomIndexBuffer;

    public:
        GeomMultiVertexBuffer() {}
        virtual ~GeomMultiVertexBuffer() {}

    public:
        void init(
            uint32_t vtxNum,
            const VertexAttrib* attribs,
            uint32_t attribNum,
            const void* data[],
            bool isDynamic = false);

        const std::vector<uint32_t>& getVBOHandles() const
        {
            return m_vbos;
        }

        uint32_t getHandleNum() const
        {
            return (uint32_t)m_vbos.size();
        }

        void* beginRead(uint32_t idx)
        {
            return beginMap(true, idx);
        }
        void endRead(uint32_t idx)
        {
            endMap(idx);
        }

        virtual bool isInitialized() const final
        {
            return getHandleNum() > 0;
        }

    private:
        void* beginMap(bool isRead, uint32_t idx);
        void endMap(uint32_t idx);

        // Hide interface.
        virtual uint32_t getVBOHandle() const override final
        {
            AT_ASSERT(false);
            return m_vbo;
        }

    protected:
        std::vector<uint32_t> m_vbos;
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
            const GeomVertexBuffer& vb,
            Primitive mode,
            uint32_t idxOffset,
            uint32_t primNum) const;

        void draw(
            const GeomMultiVertexBuffer& vb,
            Primitive mode,
            uint32_t idxOffset,
            uint32_t primNum) const;

    private:
        void draw(
            uint32_t vao,
            Primitive mode,
            uint32_t idxOffset,
            uint32_t primNum) const;

    protected:
        uint32_t m_ibo{ 0 };

        uint32_t m_idxNum{ 0 };

        bool m_isLockedIBO{ false };

        uint32_t m_initIdxNum{ 0 };
    };
}
