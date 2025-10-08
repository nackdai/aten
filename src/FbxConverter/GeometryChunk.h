#pragma once

#include "FbxImporter.h"
#include "GeometryCommon.h"
#include "FileOutputStream.h"

#include <vector>

class GeometryChunkExporter
{
    static GeometryChunkExporter s_cInstance;

public:
    static GeometryChunkExporter& getInstance() { return s_cInstance; }

protected:
    GeometryChunkExporter()
    {
        m_ExportTriList = false;
    }
    ~GeometryChunkExporter() {}

public:
    bool exportGeometry(
        uint32_t maxJointMtxNum,
        FileOutputStream* pOut,
        aten::FbxImporter* pImporter,
        bool isExportForGPUSkinning);

    void Clear();

    /** トライアングルリストで出力するかどうかを設定.
     */
    void setIsExportTriList(bool flag)
    {
        m_ExportTriList = flag;
    }

protected:
    bool exportGroup(
        FileOutputStream* pOut,
        aten::FbxImporter* pImporter);

    void bindJointToTriangle(
        aten::FbxImporter* pImporter,
        MeshInfo& sMesh);

    void classifyTriangleByJoint(MeshInfo& sMesh);

    void getMeshInfo(
        aten::FbxImporter* pImporter,
        MeshInfo& sMesh);

    bool computeVtxNormal(
        aten::FbxImporter* pImporter,
        const TriangleParam& sTri);

    bool computeVtxTangent(
        aten::FbxImporter* pImporter,
        const TriangleParam& sTri);

    void computeVtxParemters(aten::FbxImporter* pImporter);

    uint32_t exportVertices(
        FileOutputStream* pOut,
        aten::FbxImporter* pImporter);

    bool exportVertices(
        FileOutputStream* pOut,
        aten::FbxImporter* pImporter,
        const MeshInfo& sMesh,
        PrimitiveSetParam& sPrimSet);

    bool exportMesh(
        FileOutputStream* pOut,
        aten::FbxImporter* pImporter);

    void getMinMaxPos(
        aten::FbxImporter* pImporter,
        aten::vec4& v_min,
        aten::vec4& v_max,
        const PrimitiveSetParam& sPrimSet);

    bool exportPrimitiveSet(
        FileOutputStream* pOut,
        aten::FbxImporter* pImporter,
        const PrimitiveSetParam& sPrimSet);

    uint32_t exportIndices(
        FileOutputStream* pOut,
        aten::FbxImporter* pImporter,
        const PrimitiveSetParam& sPrimSet);

protected:
    std::vector<MeshInfo> m_MeshList;
    std::vector<TriangleParam> m_TriList;
    std::vector<SkinParam> m_SkinList;
    std::vector<VtxAdditional> m_VtxList;

    std::vector<uint32_t> m_ExportedVtx;

    aten::vec4 m_vMin;
    aten::vec4 m_vMax;

    aten::MeshHeader m_Header;

    // メッシュが影響を受けるマトリクスの最大数.
    // シェーダに設定するマトリクス数.
    uint32_t m_MaxJointMtxNum;

    // GPUスキニング向けの出力をするかどうか.
    bool m_isExportForGPUSkinning{ false };

    // トライアングルリストで出力するかどうか
    bool m_ExportTriList{ true };
};
