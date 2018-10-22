#pragma once

#include "deformable/DeformMeshGroup.h"
#include "deformable/SkinningVertex.h"
#include "deformable/Skeleton.h"
#include "misc/stream.h"
#include "scene/context.h"

namespace aten
{
    /**
     * @brief メッシュデータ.
     */
    class DeformMesh {
        friend class deformable;

    private:
        DeformMesh() {}
        ~DeformMesh() {}

    private:
        bool read(FileInputStream* stream);

        void initGLResources(shader* shd);

        void render(
            const context& ctxt,
            const SkeletonController& skeleton,
            IDeformMeshRenderHelper* helper);

        void release()
        {
            m_groups.clear();
        }

        void getGeometryData(
            const context& ctxt,
            std::vector<SkinningVertex>& vtx,
            std::vector<uint32_t>& idx,
            std::vector<aten::PrimitiveParamter>& tris) const;

        const MeshHeader& getDesc() const
        {
            return m_header;
        }

        GeomMultiVertexBuffer& getVBForGPUSkinning()
        {
            // TODO
            return m_groups[0].getVBForGPUSkinning();
        }

        uint32_t getTriangleCount() const
        {
            // TODO
            return m_groups[0].getTriangleCount();
        }

    private:
        MeshHeader m_header;

        std::vector<DeformMeshGroup> m_groups;
    };
}
