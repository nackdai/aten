#include <algorithm>
#include <iterator>
#include "FbxDataManager.h"

bool FbxDataManager::IsValid() const
{
    return (m_manager != nullptr && m_scene != nullptr);
}

bool FbxDataManager::open(std::string_view path)
{
    m_manager = FbxManager::Create();

    FbxIOSettings* ios = FbxIOSettings::Create(m_manager, IOSROOT);
    m_manager->SetIOSettings(ios);

    m_scene = FbxScene::Create(m_manager, "");

    auto importer = FbxImporter::Create(m_manager, "");

    int32_t fileFormat = -1;

    if (m_manager->GetIOPluginRegistry()->DetectReaderFileFormat(path.data(), fileFormat)) {
        if (importer->Initialize(path.data(), fileFormat)) {
            if (importer->Import(m_scene)) {
                importer->Destroy(true);

#if 0
                // Convert Axis System to what is used in this example, if needed
                FbxAxisSystem SceneAxisSystem = m_scene->GetGlobalSettings().GetAxisSystem();
                FbxAxisSystem axisSystem(
                    FbxAxisSystem::eYAxis,
                    FbxAxisSystem::eParityOdd,
                    FbxAxisSystem::eLeftHanded);

                if (SceneAxisSystem != axisSystem)
                {
                    axisSystem.ConvertScene(m_scene);
                }
#endif

                // Convert Unit System to what is used in this example, if needed
                FbxSystemUnit SceneSystemUnit = m_scene->GetGlobalSettings().GetSystemUnit();
                if (SceneSystemUnit.GetScaleFactor() != 1.0)
                {
                    //The unit in this tool is centimeter.
                    FbxSystemUnit::cm.ConvertScene(m_scene);
                }

                // Convert mesh, NURBS and patch into triangle mesh
                FbxGeometryConverter geomConverter(m_manager);
                geomConverter.Triangulate(m_scene, true);

                gatherNodes(m_scene->GetRootNode());
                gatherMeshes();
                gatherClusters();

                return true;
            }
        }
    }

    AT_ASSERT(false);
    return false;
}

bool FbxDataManager::openForAnm(std::string_view path, bool nodeOnly/*= false*/)
{
    m_manager = FbxManager::Create();

    FbxIOSettings* ios = FbxIOSettings::Create(m_manager, IOSROOT);
    m_manager->SetIOSettings(ios);

    m_scene = FbxScene::Create(m_manager, "animationScene");

    FbxImporter* importer = FbxImporter::Create(m_manager, "");

    int32_t fileFormat = -1;

    if (m_manager->GetIOPluginRegistry()->DetectReaderFileFormat(path.data(), fileFormat)) {
        if (importer->Initialize(path.data(), fileFormat)) {
            if (importer->Import(m_scene)) {
#if 0
                // Convert Axis System to what is used in this example, if needed
                FbxAxisSystem SceneAxisSystem = m_scene->GetGlobalSettings().GetAxisSystem();
                FbxAxisSystem axisSystem(
                    FbxAxisSystem::eYAxis,
                    FbxAxisSystem::eParityOdd,
                    FbxAxisSystem::eLeftHanded);

                if (SceneAxisSystem != axisSystem)
                {
                    axisSystem.ConvertScene(m_scene);
                }
#endif

                // Convert Unit System to what is used in this example, if needed
                FbxSystemUnit SceneSystemUnit = m_scene->GetGlobalSettings().GetSystemUnit();
                if (SceneSystemUnit.GetScaleFactor() != 1.0)
                {
                    //The unit in this tool is centimeter.
                    FbxSystemUnit::cm.ConvertScene(m_scene);
                }

                // �m�[�h���W�߂�.
                gatherNodes(nullptr);

                if (!nodeOnly) {
                    loadAnimation(importer);
                }

                importer->Destroy();

                return true;
            }
        }
    }

    AT_ASSERT(false);
    return false;
}

void FbxDataManager::close()
{
    if (m_scene) {
        m_scene->Destroy(true);
    }

    if (m_manager) {
        // Delete the FBX SDK manager. All the objects that have been allocated
        // using the FBX SDK manager and that haven't been explicitly destroyed
        // are automatically destroyed at the same time.
        m_manager->Destroy();
    }
}

void FbxDataManager::loadMesh()
{
    if (IsValid() && vertices_.size() == 0) {
        gatherFaces();

        gatherVertices();
    }
}

void FbxDataManager::loadMaterial()
{
    if (IsValid() && materials_.size() == 0) {
        // �V�[���Ɋ܂܂�郁�b�V���̉��
        auto meshCount = m_scene->GetMemberCount<FbxMesh>();

        for (int32_t i = 0; i < meshCount; ++i)
        {
            FbxMesh* fbxMesh = m_scene->GetMember<FbxMesh>(i);

            if (fbxMesh->GetElementMaterial() == NULL) {
                continue;
            }

            // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j�̐�.
            const int32_t polygonCount = fbxMesh->GetPolygonCount();

            auto& materialIndices = fbxMesh->GetElementMaterial()->GetIndexArray();

            auto mtrlCnt = materialIndices.GetCount();

            if (mtrlCnt == polygonCount)
            {
                for (int32_t i = 0; i < polygonCount; i++)
                {
                    // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j���������Ă���}�e���A���ւ̃C���f�b�N�X.
                    const int32_t materialIdx = materialIndices.GetAt(i);

                    // �}�e���A���{�̂��擾.
                    auto material = m_scene->GetMaterial(materialIdx);

                    auto itMtrl = std::find(materials_.begin(), materials_.end(), material);
                    if (itMtrl == materials_.end())
                    {
                        materials_.push_back(material);
                    }
                }
            }
            else {
                AT_ASSERT(mtrlCnt == 1);

                auto material = fbxMesh->GetNode()->GetMaterial(0);

                auto itMtrl = std::find(materials_.begin(), materials_.end(), material);
                if (itMtrl == materials_.end())
                {
                    materials_.push_back(material);
                }
            }
        }
    }
}

void FbxDataManager::loadAnimation(FbxImporter* importer)
{
    // �܂܂��A�j���[�V������.
    // �P���������Ȃ�.
    auto animStackCount = importer->GetAnimStackCount();
    AT_ASSERT(animStackCount == 1);

    auto takeInfo = importer->GetTakeInfo(0);

    auto importOffset = takeInfo->mImportOffset;
    auto startTime = takeInfo->mLocalTimeSpan.GetStart();
    auto stopTime = takeInfo->mLocalTimeSpan.GetStop();

    // TODO
    auto oneFrame = fbxsdk::FbxTime::GetOneFrameValue(fbxsdk::FbxTime::eFrames60);

    // �t���[�����v�Z.
    m_AnmStartFrame = (importOffset.Get() + startTime.Get()) / oneFrame;

    if (m_AnmStartFrame < 0) {
        m_AnmStartFrame = 0;
    }

    m_AnmStopFrame = (importOffset.Get() + stopTime.Get()) / oneFrame;
    AT_ASSERT(m_AnmStartFrame < m_AnmStopFrame);
}

// �x�[�X���f���f�[�^�Ɋ�Â��ăm�[�h�̍Ē���.
uint32_t FbxDataManager::reArrangeNodeByTargetBaseModel(FbxDataManager* target)
{
    AT_ASSERT(target);

    std::vector<Node> nodes;

    uint32_t nodeNum = target->getNodeNum();

    for (uint32_t i = 0; i < nodeNum; i++)
    {
        auto targetNode = target->getFbxNode(i);

        // �^�[�Q�b�g�ɑ�������m�[�h�����邩�T��.
        auto found = std::find_if(
            m_nodes.begin(),
            m_nodes.end(),
            [&](const Node& n) {
            std::string name(n.fbxNode->GetName());
            std::string targetName(targetNode->GetName());
            return name == targetName;
        });

        // �������̂œo�^.
        if (found != m_nodes.end()) {
            nodes.push_back(Node(found->fbxNode, (int32_t)i));
        }
    }

    if (nodes.size() != m_nodes.size()) {
        m_nodes.clear();

        std::copy(nodes.begin(), nodes.end(), std::back_inserter(m_nodes));
    }

    uint32_t ret = (uint32_t)m_nodes.size();
    return ret;
}

uint32_t FbxDataManager::getFbxMeshNum() const
{
    uint32_t ret = (uint32_t)m_fbxMeshes.size();
    return ret;
}

FbxMesh* FbxDataManager::getFbxMesh(uint32_t idx)
{
    AT_ASSERT(idx <  (uint32_t)m_fbxMeshes.size());

    FbxMesh* ret = m_fbxMeshes[idx];
    return ret;
}

uint32_t FbxDataManager::getMeshNum() const
{
    uint32_t ret = (uint32_t)m_meshes.size();
    return ret;
}

MeshSubset& FbxDataManager::getMesh(uint32_t idx)
{
    AT_ASSERT(idx <  (uint32_t)m_meshes.size());
    return m_meshes[idx];
}


const MeshSubset& FbxDataManager::getMesh(uint32_t idx) const
{
    AT_ASSERT(idx <  (uint32_t)m_meshes.size());
    return m_meshes[idx];
}

uint32_t FbxDataManager::getVtxNum() const
{
    uint32_t ret = (uint32_t)vertices_.size();
    return ret;
}

const VertexData& FbxDataManager::GetVertex(uint32_t idx) const
{
    return vertices_[idx];
}

uint32_t FbxDataManager::getNodeNum() const
{
    return  (uint32_t)m_nodes.size();
}

FbxNode* FbxDataManager::getFbxNode(uint32_t idx)
{
    return m_nodes[idx].fbxNode;
}

const Node& FbxDataManager::getNode(uint32_t idx)
{
    return m_nodes[idx];
}

uint32_t FbxDataManager::getNodeIndex(const FbxNode* node) const
{
    auto it = std::find_if(
        m_nodes.begin(),
        m_nodes.end(),
        [&](const Node& n) {
        return n.fbxNode == node;
    });

    if (it == m_nodes.end()) {
        return -1;
    }

    auto dist = std::distance(m_nodes.begin(), it);

    return  (uint32_t)dist;
}

FbxCluster* FbxDataManager::getClusterByNode(const FbxNode* node)
{
    for each (auto cluster in m_clusters)
    {
        FbxNode* targetNode = cluster->GetLink();

        if (node == targetNode) {
            return cluster;
        }
    }

    return nullptr;
}

void FbxDataManager::getSkinData(
    uint32_t idx,
    std::vector<float>& weight,
    std::vector<uint32_t>& joint) const
{
    const auto& vtx = vertices_[idx];

    std::copy(vtx.weight.begin(), vtx.weight.end(), std::back_inserter(weight));
    std::copy(vtx.joint.begin(), vtx.joint.end(), std::back_inserter(joint));
}

uint32_t FbxDataManager::getMaterialNum() const
{
    return  (uint32_t)materials_.size();
}

FbxSurfaceMaterial* FbxDataManager::GetMaterial(uint32_t idx)
{
    AT_ASSERT(idx <  (uint32_t)materials_.size());
    return materials_[idx];
}

// �m�[�h���W�߂�.
void FbxDataManager::gatherNodes(FbxNode* node)
{
#if 0
    // NOTE
    // RootNode�͏��O����.
    if (node != m_scene->GetRootNode()) {
        m_nodes.push_back(node);
    }

    for (uint32_t i = 0; i < node->GetChildCount(); i++) {
        FbxNode* child = node->GetChild(i);
        gatherNodes(child);
    }
#else
    // �m�[�h���W�߂�.
    auto nodeCount = m_scene->GetNodeCount();
    for (int32_t i = 0; i < nodeCount; ++i)
    {
        auto fbxNode = m_scene->GetNode(i);
        m_nodes.push_back(Node(fbxNode));
    }
#endif
}

// ���b�V�����W�߂�
void FbxDataManager::gatherMeshes()
{
    // �V�[���Ɋ܂܂�郁�b�V���̉��
    auto meshCount = m_scene->GetMemberCount<FbxMesh>();

    for (int32_t i = 0; i < meshCount; ++i)
    {
        FbxMesh* mesh = m_scene->GetMember<FbxMesh>(i);
        m_fbxMeshes.push_back(mesh);
    }
}

void FbxDataManager::gatherClusters()
{
    for each (FbxMesh* mesh in m_fbxMeshes)
    {
        // ���b�V���Ɋ܂܂��X�L�j���O���.
        int32_t skinCount = mesh->GetDeformerCount(FbxDeformer::EDeformerType::eSkin);

        for (int32_t n = 0; n < skinCount; n++) {
            // �X�L�j���O�����擾.
            FbxDeformer* deformer = mesh->GetDeformer(n, FbxDeformer::EDeformerType::eSkin);

            FbxSkin* skin = (FbxSkin*)deformer;

            // �X�L�j���O�ɉe����^����{�[���̐�.
            int32_t boneCount = skin->GetClusterCount();

            for (int32_t b = 0; b < boneCount; b++) {
                FbxCluster* cluster = skin->GetCluster(b);

                auto it = std::find(m_clusters.begin(), m_clusters.end(), cluster);
                if (it == m_clusters.end()) {
                    m_clusters.push_back(cluster);
                }
            }
        }
    }
}

fbxsdk::FbxSurfaceMaterial* FbxDataManager::GetMaterial(FbxMesh* fbxMesh, uint32_t index)
{
    fbxsdk::FbxSurfaceMaterial* material = nullptr;

    // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j�̐�.
    const int32_t polygonCount = fbxMesh->GetPolygonCount();

    auto& materialIndices = fbxMesh->GetElementMaterial()->GetIndexArray();

    const auto cntMtrl = materialIndices.GetCount();

    // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j���������Ă���}�e���A���ւ̃C���f�b�N�X.
    if (cntMtrl == polygonCount) {
        auto materialIdx = materialIndices.GetAt(index);

        // �}�e���A���{�̂��擾.
        material = m_scene->GetMaterial(materialIdx);
    }
    else {
        AT_ASSERT(cntMtrl == 1);

        material = fbxMesh->GetNode()->GetMaterial(0);
    }

    return material;
}

void FbxDataManager::gatherFaces()
{
    // �V�[���Ɋ܂܂�郁�b�V���̉��
    auto meshCount = m_scene->GetMemberCount<FbxMesh>();

    for (int32_t i = 0; i < meshCount; ++i)
    {
        FbxMesh* fbxMesh = m_scene->GetMember<FbxMesh>(i);

        if (fbxMesh->GetElementMaterial() == NULL) {
            continue;
        }

        m_indices.insert(std::make_pair(fbxMesh, std::vector<IndexData>()));
        auto& indices = m_indices[fbxMesh];

        // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j�̐�.
        const int32_t polygonCount = fbxMesh->GetPolygonCount();

        auto& materialIndices = fbxMesh->GetElementMaterial()->GetIndexArray();

        const auto cntMtrl = materialIndices.GetCount();

        // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j���Ƃɂǂ̃}�e���A���ɏ������Ă���̂��𒲂ׂ�.
        for (int32_t i = 0; i < polygonCount; i++)
        {
#if 0
            int32_t materialIdx = -1;

            // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j���������Ă���}�e���A���ւ̃C���f�b�N�X.
            if (cntMtrl == polygonCount) {
                materialIdx = materialIndices.GetAt(i);
            }
            else {
                AT_ASSERT(cntMtrl == 1);
                materialIdx = materialIndices.GetAt(0);
            }

            // �}�e���A���{�̂��擾.
            auto material = m_scene->GetMaterial(materialIdx);
#else
            auto material = GetMaterial(fbxMesh, i);
#endif

            auto itMtrl = std::find(materials_.begin(), materials_.end(), material);
            if (itMtrl == materials_.end())
            {
                materials_.push_back(material);
            }

            // �o�^�ς݃��b�V����T��.
            auto it = std::find(m_meshes.begin(), m_meshes.end(), MeshSubset(fbxMesh, material));

            // ���o�^������.
            if (it == m_meshes.end())
            {
                m_meshes.push_back(MeshSubset(fbxMesh, material));
            }

            indices.push_back(IndexData(fbxMesh->GetPolygonVertex(i, 0), i, fbxMesh, material));
            indices.push_back(IndexData(fbxMesh->GetPolygonVertex(i, 1), i, fbxMesh, material));
            indices.push_back(IndexData(fbxMesh->GetPolygonVertex(i, 2), i, fbxMesh, material));
        }
    }
}

void FbxDataManager::gatherVertices()
{
    // �ʒu.
    std::map<FbxMesh*, std::vector<PosData>> posList;
    gatherPos(posList);

    // UV.
    std::map<FbxMesh*, std::vector<UVData>> uvList;
    gatherUV(uvList);

    // �@��.
    std::map<FbxMesh*, std::vector<NormalData>> nmlList;
    gatherNormal(nmlList);

    // ���_�J���[.
    std::map<FbxMesh*, std::vector<ColorData>> clrList;
    gatherColor(clrList);

    // �X�L��.
    std::vector<SkinData> skinList;
    gatherSkin(skinList);

    AT_ASSERT(posList.size() == uvList.size());
    AT_ASSERT(uvList.size() == nmlList.size());

    std::vector<IndexData> indices;

    // ���_�f�[�^�𓝍����Đ�������.

    for (auto& m : m_meshes)
    {
        auto& idxList = m_indices[m.fbxMesh];

        for (uint32_t i = 0; i < idxList.size(); i++)
        {
            auto index = idxList[i];

            auto itMesh = std::find(m_meshes.begin(), m_meshes.end(), MeshSubset(index.fbxMesh, index.mtrl));
            AT_ASSERT(itMesh != m_meshes.end());

            MeshSubset& mesh = *itMesh;
            AT_ASSERT(mesh.fbxMesh == m.fbxMesh);

            auto& pos = posList[mesh.fbxMesh];
            auto& uv = uvList[mesh.fbxMesh];
            auto& nml = nmlList[mesh.fbxMesh];

            const auto& posData = pos[i];
            const auto& uvData = uv[i];
            const auto& nmlData = nml[i];

            AT_ASSERT(posData.idxInMesh == index.idxInMesh);
            AT_ASSERT(uvData.idxInMesh == index.idxInMesh);
            AT_ASSERT(nmlData.idxInMesh == index.idxInMesh);

            AT_ASSERT(posData.fbxMesh == index.fbxMesh);
            AT_ASSERT(uvData.fbxMesh == index.fbxMesh);
            AT_ASSERT(nmlData.fbxMesh == index.fbxMesh);

            AT_ASSERT(posData.mtrl == uvData.mtrl);
            AT_ASSERT(uvData.mtrl == nmlData.mtrl);

            VertexData vtx;
            vtx.idxInMesh = index.idxInMesh;
            vtx.pos = posData.pos;
            vtx.uv = uvData.uv;
            vtx.nml = nmlData.nml;
            vtx.fbxMesh = index.fbxMesh;
            vtx.mtrl = posData.mtrl;

            if (clrList.size() > 0) {
                auto& clr = clrList[mesh.fbxMesh];
                const auto& clrData = clr[i];

                AT_ASSERT(clrData.idxInMesh == index.idxInMesh);
                AT_ASSERT(clrData.fbxMesh == index.fbxMesh);
                AT_ASSERT(clrData.mtrl == posData.mtrl);

                vtx.clr = clrData.clr;
            }
            else {
                vtx.clr.Set(1.0, 1.0, 1.0);
            }

            // �X�L��.
            auto itSkin = std::find(skinList.begin(), skinList.end(), SkinData(index.idxInMesh, mesh.fbxMesh));
            if (itSkin != skinList.end())
            {
                auto& skin = *itSkin;
                std::copy(skin.weight.begin(), skin.weight.end(), std::back_inserter(vtx.weight));
                std::copy(skin.joint.begin(), skin.joint.end(), std::back_inserter(vtx.joint));
            }
            else {
                // �X�L����񂪖����ꍇ������̂ŁA���̂Ƃ��͏����m�[�h��100%.
                auto node = mesh.fbxMesh->GetNode();
                auto nodeIdx = this->getNodeIndex(node);
                vtx.joint.push_back(nodeIdx);
                vtx.weight.push_back(1.0f);
            }

            // �����f�[�^�̒��_�̗L�����m�F.
            auto it = std::find(vertices_.begin(), vertices_.end(), vtx);

            if (it == vertices_.end())
            {
                // ���o�^.

                IndexData newIdx(
                    (uint32_t)vertices_.size(),
                    0,  // �����g��Ȃ�.
                    mesh.fbxMesh,
                    mesh.mtrl);

                indices.push_back(newIdx);
                vertices_.push_back(vtx);

                mesh.vtxNum++;
            }
            else
            {
                // ���łɂ������̂ŁA�ǂ̒��_�C���f�b�N�X���擾.
                IndexData newIdx(
                    (uint32_t)std::distance(vertices_.begin(), it),
                    0,  // �����g��Ȃ�.
                    mesh.fbxMesh,
                    mesh.mtrl);

                indices.push_back(newIdx);
            }
        }
    }

    // �V�������ꂽ�C���f�b�N�X���X�g����ʃ��X�g�𐶐�.
    for (uint32_t i = 0; i < indices.size(); i += 3)
    {
        auto index = indices[i];

        AT_ASSERT(indices[i + 0].fbxMesh == indices[i + 1].fbxMesh);
        AT_ASSERT(indices[i + 0].fbxMesh == indices[i + 2].fbxMesh);

        AT_ASSERT(indices[i + 0].mtrl == indices[i + 1].mtrl);
        AT_ASSERT(indices[i + 0].mtrl == indices[i + 2].mtrl);

        auto itMesh = std::find(m_meshes.begin(), m_meshes.end(), MeshSubset(index.fbxMesh, index.mtrl));
        AT_ASSERT(itMesh != m_meshes.end());

        MeshSubset& mesh = *itMesh;

        Face face;
        face.vtx[0] = indices[i + 0].idxInMesh;
        face.vtx[1] = indices[i + 1].idxInMesh;
        face.vtx[2] = indices[i + 2].idxInMesh;

        mesh.faces.push_back(face);
    }

    // ��������Ȃ�
    m_indices.clear();
}

void FbxDataManager::gatherPos(std::map<FbxMesh*, std::vector<PosData>>& posList)
{
    for (uint32_t m = 0; m < m_fbxMeshes.size(); m++)
    {
        FbxMesh* fbxMesh = m_fbxMeshes[m];

        posList.insert(std::make_pair(fbxMesh, std::vector<PosData>()));
        auto it = posList.find(fbxMesh);

        auto& list = it->second;

        auto polygonCnt = fbxMesh->GetPolygonCount();

        for (int32_t p = 0; p < polygonCnt; p++)
        {
            for (uint32_t i = 0; i < 3; i++)
            {
#if 0
                int32_t materialIdx = -1;

                // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j���������Ă���}�e���A���ւ̃C���f�b�N�X.
                if (cntMtrl == polygonCnt) {
                    materialIdx = materialIndices.GetAt(p);
                }
                else {
                    AT_ASSERT(cntMtrl == 1);
                    materialIdx = materialIndices.GetAt(0);
                }

                // �}�e���A���{�̂��擾.
                auto material = m_scene->GetMaterial(materialIdx);
#else
                auto material = GetMaterial(fbxMesh, p);
#endif

                uint32_t idx = fbxMesh->GetPolygonVertex(p, i);

                auto position = fbxMesh->GetControlPointAt(idx);

                PosData pos;
                {
                    pos.idxInMesh = idx;
                    pos.pos = position;

                    pos.fbxMesh = fbxMesh;
                    pos.mtrl = material;
                }

                list.push_back(pos);
            }
        }
    }
}

void FbxDataManager::gatherUV(std::map<FbxMesh*, std::vector<UVData>>& uvList)
{
    for (uint32_t m = 0; m < m_fbxMeshes.size(); m++)
    {
        FbxMesh* fbxMesh = m_fbxMeshes[m];

        auto elemUV = fbxMesh->GetElementUV();
        auto mappingMode = elemUV->GetMappingMode();
        auto referenceMode = elemUV->GetReferenceMode();

        AT_ASSERT(mappingMode == FbxGeometryElement::eByPolygonVertex);
        AT_ASSERT(referenceMode == FbxGeometryElement::eIndexToDirect);

        uvList.insert(std::make_pair(fbxMesh, std::vector<UVData>()));
        auto it = uvList.find(fbxMesh);

        auto& list = it->second;

        uint32_t polygonCnt = fbxMesh->GetPolygonCount();
        uint32_t vtxNum = fbxMesh->GetControlPointsCount();

        FbxLayerElementUV* layerUV = fbxMesh->GetLayer(0)->GetUVs();
        uint32_t UVIndex = 0;

        for (uint32_t p = 0; p < polygonCnt; p++)
        {
            for (uint32_t n = 0; n < 3; n++)
            {
#if 0
                int32_t materialIdx = -1;

                // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j���������Ă���}�e���A���ւ̃C���f�b�N�X.
                if (cntMtrl == polygonCnt) {
                    materialIdx = materialIndices.GetAt(p);
                }
                else {
                    AT_ASSERT(cntMtrl == 1);
                    materialIdx = materialIndices.GetAt(0);
                }

                // �}�e���A���{�̂��擾.
                auto material = m_scene->GetMaterial(materialIdx);
#else
                auto material = GetMaterial(fbxMesh, p);
#endif

                int32_t lUVIndex = layerUV->GetIndexArray().GetAt(UVIndex);

                // �擾�����C���f�b�N�X���� UV ���擾����
                FbxVector2 lVec2 = layerUV->GetDirectArray().GetAt(lUVIndex);

                uint32_t idxInMesh = fbxMesh->GetPolygonVertex(p, n);

                UVData uv;
                {
                    uv.idxInMesh = idxInMesh;
                    uv.uv = lVec2;

                    uv.fbxMesh = fbxMesh;
                    uv.mtrl = material;
                }

                list.push_back(uv);

                UVIndex++;
            }
        }
    }
}

void FbxDataManager::gatherNormal(std::map<FbxMesh*, std::vector<NormalData>>& nmlList)
{
    for (uint32_t m = 0; m < m_fbxMeshes.size(); m++)
    {
        FbxMesh* fbxMesh = m_fbxMeshes[m];

        auto elemNml = fbxMesh->GetElementNormal();
        auto mappingMode = elemNml->GetMappingMode();
        auto referenceMode = elemNml->GetReferenceMode();

        nmlList.insert(std::make_pair(fbxMesh, std::vector<NormalData>()));
        auto it = nmlList.find(fbxMesh);

        auto& list = it->second;

        uint32_t polygonCnt = fbxMesh->GetPolygonCount();
        uint32_t vtxNum = fbxMesh->GetControlPointsCount();

        FbxLayerElementNormal* layerNml = fbxMesh->GetLayer(0)->GetNormals();
        uint32_t idxNml = 0;

        if (mappingMode == FbxGeometryElement::eByPolygonVertex)
        {
            for (uint32_t p = 0; p < polygonCnt; p++)
            {
                for (uint32_t n = 0; n < 3; n++)
                {
#if 0
                    int32_t materialIdx = -1;

                    // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j���������Ă���}�e���A���ւ̃C���f�b�N�X.
                    if (cntMtrl == polygonCnt) {
                        materialIdx = materialIndices.GetAt(p);
                    }
                    else {
                        AT_ASSERT(cntMtrl == 1);
                        materialIdx = materialIndices.GetAt(0);
                    }

                    // �}�e���A���{�̂��擾.
                    auto material = m_scene->GetMaterial(materialIdx);
#else
                    auto material = GetMaterial(fbxMesh, p);
#endif

                    int32_t lNmlIndex = (referenceMode == FbxGeometryElement::eIndexToDirect
                        ? layerNml->GetIndexArray().GetAt(idxNml)
                        : idxNml);

                    // �擾�����C���f�b�N�X����@�����擾����
                    FbxVector4 lVec4 = layerNml->GetDirectArray().GetAt(lNmlIndex);

                    uint32_t idxInMesh = fbxMesh->GetPolygonVertex(p, n);

                    NormalData nml;
                    {
                        nml.idxInMesh = idxInMesh;
                        nml.nml = lVec4;

                        nml.fbxMesh = fbxMesh;
                        nml.mtrl = material;
                    }

                    list.push_back(nml);

                    idxNml++;
                }
            }
        }
        else {
            auto& indices = m_indices[fbxMesh];

            for (const auto& idx : indices)
            {
                auto material = GetMaterial(fbxMesh, idx.polygonIdxInMesh);

                int32_t lNmlIndex = (referenceMode == FbxGeometryElement::eIndexToDirect
                    ? layerNml->GetIndexArray().GetAt(idx.idxInMesh)
                    : idx.idxInMesh);

                // �擾�����C���f�b�N�X����@�����擾����
                FbxVector4 lVec4 = layerNml->GetDirectArray().GetAt(lNmlIndex);

                NormalData nml;
                {
                    nml.idxInMesh = idx.idxInMesh;
                    nml.nml = lVec4;

                    nml.fbxMesh = fbxMesh;
                    nml.mtrl = material;
                }

                list.push_back(nml);
            }
        }
    }
}

void FbxDataManager::gatherColor(std::map<FbxMesh*, std::vector<ColorData>>& clrList)
{
    for (uint32_t m = 0; m < m_fbxMeshes.size(); m++)
    {
        FbxMesh* fbxMesh = m_fbxMeshes[m];

        if (fbxMesh->GetElementVertexColorCount() == 0) {
            continue;
        }

        auto elemClr = fbxMesh->GetElementVertexColor();
        auto mappingMode = elemClr->GetMappingMode();
        auto referenceMode = elemClr->GetReferenceMode();

        AT_ASSERT(mappingMode == FbxGeometryElement::eByPolygonVertex);
        AT_ASSERT(referenceMode == FbxGeometryElement::eIndexToDirect);

        auto& materialIndices = fbxMesh->GetElementMaterial()->GetIndexArray();

        const auto cntMtrl = materialIndices.GetCount();

        clrList.insert(std::make_pair(fbxMesh, std::vector<ColorData>()));
        auto it = clrList.find(fbxMesh);

        auto& list = it->second;

        uint32_t polygonCnt = fbxMesh->GetPolygonCount();
        uint32_t vtxNum = fbxMesh->GetControlPointsCount();

        FbxLayerElementVertexColor* layerClr = fbxMesh->GetLayer(0)->GetVertexColors();
        uint32_t idxClr = 0;

        for (uint32_t p = 0; p < polygonCnt; p++)
        {
            for (uint32_t n = 0; n < 3; n++)
            {
#if 0
                int32_t materialIdx = -1;

                // ���b�V���Ɋ܂܂��|���S���i�O�p�`�j���������Ă���}�e���A���ւ̃C���f�b�N�X.
                if (cntMtrl == polygonCnt) {
                    materialIdx = materialIndices.GetAt(p);
                }
                else {
                    AT_ASSERT(cntMtrl == 1);
                    materialIdx = materialIndices.GetAt(0);
                }

                // �}�e���A���{�̂��擾.
                auto material = m_scene->GetMaterial(materialIdx);
#else
                auto material = GetMaterial(fbxMesh, p);
#endif

                int32_t lNmlIndex = layerClr->GetIndexArray().GetAt(idxClr);

                // �擾�����C���f�b�N�X����@�����擾����
                FbxColor color = layerClr->GetDirectArray().GetAt(lNmlIndex);

                uint32_t idxInMesh = fbxMesh->GetPolygonVertex(p, n);

                ColorData clr;
                {
                    clr.idxInMesh = idxInMesh;
                    clr.clr = color;

                    clr.fbxMesh = fbxMesh;
                    clr.mtrl = material;
                }

                list.push_back(clr);

                idxClr++;
            }
        }
    }
}

void FbxDataManager::gatherSkin(std::vector<SkinData>& skinList)
{
#if 0
    // NOTE
    // http://qiita.com/makanai/items/b9a4d82d245475a3e143
    // SD Unity�����̖��
    // ��̃��f���ł���_face�m�[�h��Character1_Head�̎q�ɂȂ��Ă��܂����A�X�L�j���O����Ă��Ȃ����߁A���_�ʒu�ɕ\������܂�.
    // -> �X�L�j���O����ĂȂ��ꍇ�A�e�m�[�h�֋����I�ɃX�L�j���O����.

    for (uint32_t m = 0; m < m_fbxMeshes.size(); m++)
    {
        fbxsdk::FbxMesh* fbxMesh = m_fbxMeshes[m];

        // ���b�V���Ɋ܂܂��X�L�j���O���.
        int32_t skinCount = fbxMesh->GetDeformerCount(FbxDeformer::EDeformerType::eSkin);

        // �X�L�j���O����ĂȂ��ꍇ�A�e�m�[�h�֋����I�ɃX�L�j���O����.
        if (skinCount == 0) {
            std::function<fbxsdk::FbxNode*(fbxsdk::FbxNode* node, fbxsdk::FbxMesh* mesh)> findNodeByMesh = [&](fbxsdk::FbxNode* node, fbxsdk::FbxMesh* mesh)->fbxsdk::FbxNode*
            {
                if (node->getMesh() == mesh) {
                    return node;
                }
                for (int32_t i = 0; i < node->GetChildCount(); i++) {
                    auto n = findNodeByMesh(node->GetChild(i), mesh);
                    if (n) {
                        return n;
                    }
                }
                return nullptr;
            };

            auto child = findNodeByMesh(m_scene->GetRootNode(), fbxMesh);
            auto parent = child->GetParent();
            parent->RemoveChild(child);
            m_scene->GetRootNode()->AddChild(child);

            FbxAnimEvaluator* sceneEvaluator = m_scene->GetAnimationEvaluator();
            auto mtxBase = sceneEvaluator->GetNodeGlobalTransform(parent);
            auto mtxInv = mtxBase.Inverse();

            FbxCluster *clusterToRoot = FbxCluster::Create(m_scene, "");
            clusterToRoot->SetLink(parent);
            clusterToRoot->SetTransformLinkMatrix(mtxBase);
            clusterToRoot->SetLinkMode(FbxCluster::eTotalOne);

            for (int32_t p = 0; p < fbxMesh->GetControlPointsCount(); p++) {
                clusterToRoot->AddControlPointIndex(p, 1.0f);
            }

            FbxSkin* skin = FbxSkin::Create(m_scene, "");
            skin->AddCluster(clusterToRoot);
            fbxMesh->AddDeformer(skin);
        }
    }
#endif

    for (uint32_t m = 0; m < m_fbxMeshes.size(); m++)
    {
        fbxsdk::FbxMesh* fbxMesh = m_fbxMeshes[m];

        AT_PRINTF("Mesh Node (%s)\n", fbxMesh->GetNode()->GetName());

        // ���b�V���Ɋ܂܂��X�L�j���O���.
        int32_t skinCount = fbxMesh->GetDeformerCount(FbxDeformer::EDeformerType::eSkin);

        for (int32_t n = 0; n < skinCount; n++) {
            // �X�L�j���O�����擾.
            FbxDeformer* deformer = fbxMesh->GetDeformer(n, FbxDeformer::EDeformerType::eSkin);

            FbxSkin* skin = (FbxSkin*)deformer;

            // �X�L�j���O�ɉe����^����{�[���̐�.
            int32_t boneCount = skin->GetClusterCount();

            for (int32_t b = 0; b < boneCount; b++) {
                FbxCluster* cluster = skin->GetCluster(b);

                // �{�[�����e����^���钸�_��.
                int32_t influencedVtxNum = cluster->GetControlPointIndicesCount();

                // �{�[���Ɗ֘A�Â��Ă���m�[�h.
                FbxNode* targetNode = cluster->GetLink();

                // �m�[�h�̃C���f�b�N�X.
                int32_t nodeIdx = getNodeIndex(targetNode);
                AT_ASSERT(nodeIdx >= 0);

                AT_PRINTF("    skin[%d] : bone [%d] ([%d]%s)\n", n, b, nodeIdx, targetNode->GetName());

                for (int32_t v = 0; v < influencedVtxNum; v++) {
                    int32_t vtxIdxInMesh = cluster->GetControlPointIndices()[v];
                    float weight = (float)cluster->GetControlPointWeights()[v];

                    auto it = std::find(skinList.begin(), skinList.end(), SkinData(vtxIdxInMesh, fbxMesh));

                    if (it == skinList.end()) {
                        SkinData skin(vtxIdxInMesh, fbxMesh);
                        skin.weight.push_back(weight);
                        skin.joint.push_back(nodeIdx);

                        skinList.push_back(skin);
                    }
                    else {
                        SkinData& skin = *it;

                        skin.weight.push_back(weight);
                        skin.joint.push_back(nodeIdx);
                    }
                }
            }
        }
    }
}
