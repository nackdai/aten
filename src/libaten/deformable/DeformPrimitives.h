#pragma

#include <vector>
#include <functional>

#include "defs.h"
#include "deformable/MSHFormat.h"
#include "deformable/Skeleton.h"
#include "misc/stream.h"
#include "visualizer/GeomDataBuffer.h"
#include "math/mat4.h"
#include "material/material.h"
#include "scene/host_scene_context.h"

namespace aten
{
    class IDeformMeshRenderHelper {
    protected:
        IDeformMeshRenderHelper() {}
        virtual ~IDeformMeshRenderHelper() {}

    public:
        virtual void applyMatrix(uint32_t idx, const mat4& mtx) = 0;
        virtual void applyMaterial(const context& ctxt, const MeshMaterial& mtrlDesc) = 0;
        virtual void commitChanges(bool isGPUSkinning, uint32_t triOffset) = 0;
    };

    /**
     * @brief プリミティブデータ.
     * メッシュデータの最小単位.
     */
    class DeformPrimitives {
        friend class DeformMeshSet;
        friend class DeformMeshGroup;

    public:
        DeformPrimitives() {}
        ~DeformPrimitives() {}

    private:
        bool read(
            FileInputStream* stream,
            bool isGPUSkinning);

        void render(
            const SkeletonController& skeleton,
            IDeformMeshRenderHelper* helper,
            bool isGPUSkinning);

        const PrimitiveSet& getDesc() const
        {
            return m_desc;
        }

        void setVB(GeomVertexBuffer* vb)
        {
            vertex_buffer_ = vb;
        }

        void setVB(GeomMultiVertexBuffer* vb)
        {
            vertex_buffer_ = vb;
        }

        void getIndices(std::vector<uint32_t>& indices) const;

    private:
        void setTriOffset(uint32_t offset)
        {
            m_triOffset = offset;
        }

    private:
        PrimitiveSet m_desc;

        // ジョイントインデックス.
        std::vector<int16_t> m_joints;

        std::vector<uint32_t> m_indices;

        GeomVertexBuffer* vertex_buffer_{ nullptr };
        GeomIndexBuffer m_ib;

        uint32_t m_triOffset{ 0 };
    };
}
