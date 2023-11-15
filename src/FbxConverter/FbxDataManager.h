#pragma once

#include "fbxsdk.h"
#include "types.h"
#include "defs.h"

#include <vector>
#include <map>
#include <set>

struct Face {
    uint32_t vtx[3];
};

struct UVData {
    uint32_t idxInMesh;
    FbxVector2 uv;

    FbxMesh* fbxMesh{ nullptr };
    FbxSurfaceMaterial* mtrl{ nullptr };
};

struct PosData {
    uint32_t idxInMesh;
    FbxVector4 pos;

    FbxMesh* fbxMesh{ nullptr };
    FbxSurfaceMaterial* mtrl{ nullptr };
};

struct NormalData {
    uint32_t idxInMesh;
    FbxVector4 nml;

    FbxMesh* fbxMesh{ nullptr };
    FbxSurfaceMaterial* mtrl{ nullptr };
};

struct ColorData {
    uint32_t idxInMesh;
    FbxColor clr;

    FbxMesh* fbxMesh{ nullptr };
    FbxSurfaceMaterial* mtrl{ nullptr };
};

struct VertexData {
    uint32_t idxInMesh;

    FbxVector2 uv;      // UV.
    FbxVector4 pos;     // 位置.
    FbxVector4 nml;     // 法線.
    FbxColor clr;       // 頂点カラー.

    std::vector<float> weight;
    std::vector<uint32_t> joint;

    FbxMesh* fbxMesh{ nullptr };
    FbxSurfaceMaterial* mtrl{ nullptr };

    bool operator==(const VertexData& rhs)
    {
        bool isPos = (this->pos.mData[0] == rhs.pos.mData[0])
                        && (this->pos.mData[1] == rhs.pos.mData[1])
                        && (this->pos.mData[2] == rhs.pos.mData[2]);

        bool isUV = (this->uv.mData[0] == rhs.uv.mData[0])
                        && (this->uv.mData[1] == rhs.uv.mData[1]);

        bool isNml = (this->nml.mData[0] == rhs.uv.mData[0])
                        && (this->nml.mData[1] == rhs.nml.mData[1])
                        && (this->nml.mData[2] == rhs.nml.mData[2]);

        bool isMesh = (this->fbxMesh == rhs.fbxMesh);
        bool isMtrl = (this->mtrl == rhs.mtrl);

        return (isPos && isUV && isNml && isMesh && isMtrl);
    }
};

struct IndexData
{
    uint32_t idxInMesh;

    uint32_t polygonIdxInMesh;

    FbxMesh* fbxMesh{ nullptr };
    FbxSurfaceMaterial* mtrl{ nullptr };

    IndexData(uint32_t idx, uint32_t polyIdx, FbxMesh* mesh, FbxSurfaceMaterial* _mtrl)
        : idxInMesh(idx), polygonIdxInMesh(polyIdx), fbxMesh(mesh), mtrl(_mtrl)
    {}
};

struct SkinData
{
    uint32_t idxInMesh;

    FbxMesh* fbxMesh{ nullptr };

    std::vector<float> weight;
    std::vector<uint32_t> joint;

    SkinData() {}

    SkinData(uint32_t idx, FbxMesh* mesh)
        : idxInMesh(idx), fbxMesh(mesh)
    {}

    bool isEmpty() const
    {
        return (fbxMesh == nullptr);
    }

    bool operator==(const SkinData& rhs)
    {
        return (idxInMesh == rhs.idxInMesh
            && fbxMesh == rhs.fbxMesh);
    }
};

struct MeshSubset {
    std::vector<Face> faces;

    FbxMesh* fbxMesh{ nullptr };
    FbxSurfaceMaterial* mtrl{ nullptr };

    uint32_t vtxNum = 0;

    MeshSubset() {}

    MeshSubset(FbxMesh* _mesh, FbxSurfaceMaterial* _mtrl)
        : fbxMesh(_mesh), mtrl(_mtrl)
    {}

    bool operator==(const MeshSubset& rhs)
    {
        return (fbxMesh == rhs.fbxMesh && mtrl == rhs.mtrl);
    }
};

struct Node {
    fbxsdk::FbxNode* fbxNode{ nullptr };
    int32_t targetIdx{ -1 };

    Node() {}

    Node(fbxsdk::FbxNode* node)
        : fbxNode(node)
    {}

    Node(fbxsdk::FbxNode* node, int32_t idx)
        : fbxNode(node), targetIdx(idx)
    {}
};

class FbxDataManager {
public:
    FbxDataManager() = default;
    ~FbxDataManager() = default;

public:
    bool IsValid() const;

    bool open(std::string_view path);
    bool openForAnm(std::string_view path, bool nodeOnly = false);

    void close();

    void loadMesh();
    void loadMaterial();

    uint32_t getFbxMeshNum() const;

    FbxMesh* getFbxMesh(uint32_t idx);

    uint32_t getMeshNum() const;

    MeshSubset& getMesh(uint32_t idx);
    const MeshSubset& getMesh(uint32_t idx) const;

    uint32_t getVtxNum() const;

    const VertexData& GetVertex(uint32_t idx) const;

    uint32_t getNodeNum() const;

    FbxNode* getFbxNode(uint32_t idx);
    const Node& getNode(uint32_t idx);

    uint32_t getNodeIndex(const FbxNode* node) const;

    FbxCluster* getClusterByNode(const FbxNode* node);

    void getSkinData(
        uint32_t idx,
        std::vector<float>& weight,
        std::vector<uint32_t>& joint) const;

    uint32_t getMaterialNum() const;
    FbxSurfaceMaterial* GetMaterial(uint32_t idx);

    int32_t getAnmStartFrame() const { return m_AnmStartFrame; }
    int32_t getAnmStopFrame() const { return m_AnmStopFrame; }

    // ベースモデルデータに基づいてノードの再調整.
    uint32_t reArrangeNodeByTargetBaseModel(FbxDataManager* target);

private:
    void loadAnimation(FbxImporter* importer);

    // ノードを集める.
    void gatherNodes(FbxNode* node);

    void gatherMeshes();

    void gatherClusters();

    void gatherFaces();

    void gatherVertices();

    void gatherPos(std::map<FbxMesh*, std::vector<PosData>>& posList);
    void gatherUV(std::map<FbxMesh*, std::vector<UVData>>& uvList);

    void gatherNormal(std::map<FbxMesh*, std::vector<NormalData>>& nmlList);
    void gatherColor(std::map<FbxMesh*, std::vector<ColorData>>& clrList);

    void gatherSkin(std::vector<SkinData>& skinList);

    fbxsdk::FbxSurfaceMaterial* GetMaterial(FbxMesh* fbxMesh, uint32_t index);

private:
    FbxManager* m_manager{ nullptr };
    FbxScene* m_scene{ nullptr };

    std::vector<Node> m_nodes;
    std::vector<FbxMesh*> m_fbxMeshes;

    std::vector<FbxCluster*> m_clusters;

    std::vector<MeshSubset> m_meshes;

    std::map<FbxMesh*, std::vector<IndexData>> m_indices;
    std::vector<VertexData> vertices_;

    std::vector<fbxsdk::FbxSurfaceMaterial*> materials_;

    int32_t m_AnmStartFrame;
    int32_t m_AnmStopFrame;
};
