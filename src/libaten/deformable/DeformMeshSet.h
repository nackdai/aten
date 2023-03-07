#pragma once

#include "deformable/DeformPrimitives.h"
#include "deformable/Skeleton.h"
#include "misc/stream.h"
#include "geometry/NoHitableMesh.h"
#include "scene/context.h"

namespace aten
{
    /**
     * @brief メッシュセット.
     * マテリアルごとのプリミティブセットの集まり
     */
    class DeformMeshSet : public NoHitableMesh {
        friend class DeformMeshGroup;

    public:
        DeformMeshSet() {}
        ~DeformMeshSet() {}

    private:
        bool read(
            FileInputStream* stream,
            bool isGPUSkinning);

        void initGLResources(
            shader* shd,
            bool isGPUSkinning,
            std::vector<GeomVertexBuffer>& vbs);

        void setExternalVertexBuffer(GeomMultiVertexBuffer& vb);

        void render(
            const context& ctxt,
            const SkeletonController& skeleton,
            IDeformMeshRenderHelper* helper,
            bool isGPUSkinning);

        const std::vector<DeformPrimitives>& getPrimitives() const
        {
            return m_prims;
        }

        const MeshSet& getDesc() const
        {
            return m_desc;
        }

    private:
        MeshSet m_desc;

        std::vector<DeformPrimitives> m_prims;
    };
}
