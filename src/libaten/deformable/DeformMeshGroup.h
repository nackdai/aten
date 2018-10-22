#pragma once

#include "deformable/DeformMeshSet.h"
#include "deformable/SkinningVertex.h"
#include "deformable/Skeleton.h"
#include "misc/stream.h"
#include "geometry/geomparam.h"
#include "scene/context.h"

namespace aten
{
    /**
     * @brief メッシュグループ.
     * LODのレベルごとのメッシュセットの集まり
     */
    class DeformMeshGroup {
        friend class DeformMesh;

    public:
        DeformMeshGroup() {}
        ~DeformMeshGroup() {}

    private:
        bool read(
            FileInputStream* stream,
            bool isGPUSkinning);

        void initGLResources(
            shader* shd,
            bool isGPUSkinning);

        void render(
            const context& ctxt,
            const SkeletonController& skeleton,
            IDeformMeshRenderHelper* helper,
            bool isGPUSkinning);

        void getGeometryData(
            const context& ctxt,
            std::vector<SkinningVertex>& vtx,
            std::vector<uint32_t>& idx,
            std::vector<aten::PrimitiveParamter>& tris) const;

        GeomMultiVertexBuffer& getVBForGPUSkinning()
        {
            return m_vbForGPUSkinning;
        }

        uint32_t getTriangleCount() const
        {
            return m_triangles;
        }

    private:
        MeshGroup m_desc;

        uint32_t m_vtxTotalNum{ 0 };
        std::vector<uint8_t> m_vertices;

        std::vector<DeformMeshSet> m_meshs;
        std::vector<GeomVertexBuffer> m_vbs;

        uint32_t m_triangles{ 0 };

        GeomMultiVertexBuffer m_vbForGPUSkinning;
    };
}
