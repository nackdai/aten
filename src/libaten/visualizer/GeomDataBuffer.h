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
            int32_t idx_offset,
            size_t primitive_num);

        void* BeginRead()
        {
            return BeginMap(true);
        }
        void EndRead()
        {
            EndMap();
        }

        void* beginWrite()
        {
            return BeginMap(false);
        }
        void endWrite()
        {
            EndMap();
        }

        void clear();

        size_t getVtxNum() const
        {
            return m_vtxNum;
        }

        virtual uint32_t GetVBOHandle() const
        {
            return m_vbo;
        }

        virtual bool IsInitialized() const
        {
            return m_vbo > 0;
        }

    private:
        void* BeginMap(bool isRead);
        void EndMap();

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

        const std::vector<uint32_t>& GetVBOHandles() const
        {
            return vbos_;
        }

        uint32_t GetHandleNum() const
        {
            return (uint32_t)vbos_.size();
        }

        void* BeginRead(uint32_t idx)
        {
            return BeginMap(true, idx);
        }
        void EndRead(uint32_t idx)
        {
            EndMap(idx);
        }

        bool IsInitialized() const final
        {
            return GetHandleNum() > 0;
        }

    private:
        void* BeginMap(bool isRead, uint32_t idx);
        void EndMap(uint32_t idx);

        // Hide interface.
        uint32_t GetVBOHandle() const override final
        {
            AT_ASSERT(false);
            return m_vbo;
        }

    protected:
        std::vector<uint32_t> vbos_;
    };

    //////////////////////////////////////////////////////////

    class GeomIndexBuffer {
    public:
        GeomIndexBuffer() {}
        virtual ~GeomIndexBuffer();

    public:
        void init(
            uint32_t idx_num,
            const void* data);

        void update(
            uint32_t idx_num,
            const void* data);

        void lock(void** dst);
        void unlock();

        void draw(
            const GeomVertexBuffer& vb,
            Primitive mode,
            uint32_t idx_offset,
            uint32_t primitive_num) const;

        void draw(
            const GeomMultiVertexBuffer& vb,
            Primitive mode,
            uint32_t idx_offset,
            uint32_t primitive_num) const;

    private:
        void draw(
            uint32_t vao,
            Primitive mode,
            uint32_t idx_offset,
            uint32_t primitive_num) const;

    protected:
        uint32_t ibo_{ 0 };

        uint32_t idx_num_{ 0 };

        bool is_locked_ibo_{ false };

        uint32_t init_idx_num_{ 0 };
    };
}
