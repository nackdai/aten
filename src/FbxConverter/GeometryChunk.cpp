#include "GeometryChunk.h"
#include "NvTriStrip.h"

#include "misc/color.h"

#include <algorithm>

//#pragma optimize( "", off)

GeometryChunkExporter GeometryChunkExporter::s_cInstance;

// ジオメトリチャンク
// +------------------------+
// |     チャンクヘッダ     |
// +------------------------+
// |   ジオメトリチャンク   |
// |         ヘッダ         |
// +------------------------+
// |   頂点データテーブル   |
// | +--------------------+ |
// | |      ヘッダ        | |
// | +--------------------+ |
// | |     頂点データ     | |
// | +--------------------+ |
// |         ・・・         |
// |                        |
// |    メッシュテーブル    |
// | +--------------------+ |
// | |      メッシュ      | |
// | |+------------------+| |
// | ||     ヘッダ       || |
// | |+------------------+| |
// | |                    | |
// | |     サブセット     | |
// | |+------------------+| |
// | ||     ヘッダ       || |
// | |+------------------+| |
// | ||インデックスデータ|| |
// | |+------------------+| |
// | |      ・・・        | |
// | +--------------------+ |
// |        ・・・          |
// +------------------------+

bool GeometryChunkExporter::exportGeometry(
    uint32_t maxJointMtxNum,
    FileOutputStream* pOut,
    aten::FbxImporter* pImporter)
{
    // メッシュが影響を受けるマトリクスの最大数
    m_MaxJointMtxNum = std::max<uint32_t>(
                        maxJointMtxNum,
                        aten::MaxJointMtxNum);
    m_MaxJointMtxNum = ((m_MaxJointMtxNum & 0x01) == 1
                        ? m_MaxJointMtxNum + 1
                        : m_MaxJointMtxNum);

    
    {
        // TODO
        // version, magic number...

        m_Header.sizeHeader = sizeof(m_Header);
    }

    // Blank for S_MSH_HEADER.
    IoStreamSeekHelper seekHelper(pOut);
    AT_VRETURN_FALSE(seekHelper.skip(sizeof(m_Header)));

    // TODO
    // Export mesh groups.
	AT_VRETURN_FALSE(exportGroup(pOut, pImporter));

    m_Header.sizeFile = pOut->getCurPos();

    m_Header.minVtx[0] = m_vMin.x;
    m_Header.minVtx[1] = m_vMin.y;
    m_Header.minVtx[2] = m_vMin.z;

    m_Header.maxVtx[0] = m_vMin.x;
    m_Header.maxVtx[1] = m_vMin.y;
    m_Header.maxVtx[2] = m_vMin.z;

    // TODO
    m_Header.numMeshGroup = 1;

    // Export S_MSH_HEADER.
    {
        // Rmenber end of geometry chunk.
		AT_VRETURN_FALSE(seekHelper.returnWithAnchor());

        OUTPUT_WRITE_VRETURN(pOut, &m_Header, 0, sizeof(m_Header));
        seekHelper.step(sizeof(m_Header));

        // returnTo end of geometry chunk.
		AT_VRETURN_FALSE(seekHelper.returnToAnchor());
    }

    pImporter->exportGeometryCompleted();

    return true;
}

void GeometryChunkExporter::Clear()
{
    m_MeshList.clear();
    m_TriList.clear();
    m_SkinList.clear();
    m_VtxList.clear();
}

// メッシュグループ出力
bool GeometryChunkExporter::exportGroup(
    FileOutputStream* pOut,
    aten::FbxImporter* pImporter)
{
    aten::MeshGroup sGroupInfo;
    {
        // メッシュグループに所属するメッシュセット数
        sGroupInfo.numMeshSet = pImporter->getMeshNum();
    }

    uint32_t nChunkStartPos = pOut->getCurPos();

    // Blank for MeshGroup.
    IoStreamSeekHelper seekHelper(pOut);
    AT_VRETURN_FALSE(seekHelper.skip(sizeof(sGroupInfo)));

    // メッシュセットリスト
    m_MeshList.resize(sGroupInfo.numMeshSet);

    uint32_t nVtxNum = 0;

#if 0
    // 三角形リストの保持先を覚えさせておく
    PrimitiveSetParam::SetTriList(&m_TriList);
#endif

    for (uint32_t nMeshIdx = 0; nMeshIdx < sGroupInfo.numMeshSet; nMeshIdx++) {
        MeshInfo& sMesh = m_MeshList[nMeshIdx];

        pImporter->beginMesh(nMeshIdx);

        // メッシュセットに含まれる三角形の開始位置
        sMesh.startTri = static_cast<uint32_t>(m_TriList.size());

        // Get triangels.
        nVtxNum += pImporter->getTriangles(m_TriList);

        // メッシュセットに含まれる三角形の終了位置
        sMesh.endTri = static_cast<uint32_t>(m_TriList.size());

        // Get skin data.
        pImporter->getSkinList(m_SkinList);

        // Bind joint to triangle.
        bindJointToTriangle(pImporter, sMesh);

        // Classify triangles by joint
        classifyTriangleByJoint(sMesh);

        getMeshInfo(pImporter, sMesh);

        pImporter->endMesh();
    }

    m_VtxList.resize(nVtxNum);

    // Compute Normal, Tangent, etc...
    computeVtxParemters(pImporter);

    // Export vertices.
    sGroupInfo.numVB = exportVertices(
                        pOut,
                        pImporter);
	AT_VRETURN_FALSE(sGroupInfo.numVB > 0);

    // Export meshes.
    AT_VRETURN_FALSE(
        exportMesh(
            pOut,
            pImporter));

    m_Header.numVB += sGroupInfo.numVB;
    m_Header.numMeshSet += sGroupInfo.numMeshSet;

    // Export MeshGroup.
    {
        // Rmenber end of geometry chunk.
		AT_VRETURN_FALSE(seekHelper.returnWithAnchor());

        OUTPUT_WRITE_VRETURN(pOut, &sGroupInfo, 0, sizeof(sGroupInfo));
        seekHelper.step(sizeof(sGroupInfo));

        // returnTo end of geometry chunk.
		AT_VRETURN_FALSE(seekHelper.returnToAnchor());
    }

    pImporter->exportGeometryCompleted();

    return true;
}

// 関節情報
struct JointInfo {
    // 関節インデックス
    uint32_t idx;

    // 指定された関節におけるスキニングウエイト最大値
	float maxWeight{ AT_MATH_INF };

    JointInfo()
    {
    }

    void SetWeight(float weight)
    {
        maxWeight = (maxWeight < weight ? weight : maxWeight);
    }

    bool operator==(uint32_t i)
    {
        return (idx == i);
    }

    // ソート用
    bool operator<(const JointInfo& rhs) const
    {
        return (maxWeight < rhs.maxWeight);
    }
};

// 指定されたメッシュについて三角形と関節を関連付ける
void GeometryChunkExporter::bindJointToTriangle(
    aten::FbxImporter* pImporter,
    MeshInfo& sMesh)
{
    if (m_SkinList.empty()) {
        // There is no joints, so nothing is done.
        return;
    }

    // 関節情報の一時保存用のリスト
    std::vector<JointInfo> tvJointInfo;

    for (uint32_t nTriPos = sMesh.startTri; nTriPos < sMesh.endTri; nTriPos++) {
        // Get triangle belonged to specified mesh.
        TriangleParam& sTri = m_TriList[nTriPos];

        for (uint32_t i = 0; i < 3; i++) {
            // 三角形が持つ頂点に関連したスキニング情報へのインデックスを取得
            uint32_t nSkinIdx = pImporter->getSkinIdxAffectToVtx(sTri.vtx[i]);

            // Get skin information.
            const SkinParam& sSkin = m_SkinList[nSkinIdx];

            for (size_t n = 0; n < sSkin.joint.size(); n++) {
                // 関節インデックス取得
                uint32_t idx = sSkin.joint[n];

                // // 三角形に影響を与える関節インデックスが設定されているか調べる

                if (std::find(sTri.joint.begin(), sTri.joint.end(), idx) == sTri.joint.end()) {
                    // 未設定

                    // 三角形に影響を与える関節インデックスを登録
                    sTri.joint.insert(idx);

                    // 関節情報を登録
                    tvJointInfo.push_back(JointInfo());
                    JointInfo& sJointInfo = tvJointInfo.back();
                    {
                        sJointInfo.idx = idx;
                        sJointInfo.SetWeight(sSkin.weight[n]);
                    }
                }
                else {
                    // 設定済み

                    // 三角形に影響を与える関節情報を取得
                    // triangle belongs to joint.
                    std::vector<JointInfo>::iterator itJointInfo = std::find(
                                                                        tvJointInfo.begin(),
                                                                        tvJointInfo.end(),
                                                                        idx);
                    itJointInfo->SetWeight(sSkin.weight[n]);
                }
            }
        }

        // Sort in order in which weight value is small.
        std::sort(tvJointInfo.begin(), tvJointInfo.end());

        AT_ASSERT(sTri.joint.size() > 0);
        
        // 三角形に影響を与える関節数は最大４まで
        // それを超えた場合は一番影響が小さいもの（ウエイトが小さいもの）を削除する
        // If num of skin is over 4, num of skin is limited to 4 by weight.
        if (sTri.joint.size() > 4) {
            // 先頭の関節情報を取得
            // ここでは既にウエイトが小さいものの順にソートされているので
            // 一番影響が小さい関節の情報を取得
            std::vector<JointInfo>::iterator itJointInfo = tvJointInfo.begin();

            // 三角形に影響を与える関節数が４以下になるまで削除を繰り返す
            for (size_t n = sTri.joint.size(); n > 4; itJointInfo++, n--) {
                const JointInfo& sJointInfo = *itJointInfo;

                // Delete joint which has smallest weight value.
                sTri.eraseJoint(sJointInfo.idx);

                for (uint32_t i = 0; i < 3; i++) {
                    // 指定された頂点に影響を与えるスキニング情報を取得.
                    uint32_t nSkinIdx = pImporter->getSkinIdxAffectToVtx(sTri.vtx[i]);

                    // スキニング情報からも関節を削除
                    SkinParam& sSkin = m_SkinList[nSkinIdx];
                    if (sSkin.eraseJoint(sJointInfo.idx)) {
                        // ウエイト値の合計が１になるように正規化する
                        sSkin.normalize();
                    }
                }
            }
        }

        AT_ASSERT(sTri.joint.size() <= 4);

        tvJointInfo.clear();
    }
}

// 基準となるプリミティブセットとのマージ候補を探すための関数オブジェクト
struct FuncFindIncludedJointIdx {
    // マージ候補のプリミティブセット
    // 基準となっているプリミティブセットと一致する関節数ごとに候補を登録する
    static std::vector< std::vector<const PrimitiveSetParam*> > candidateList;

    // 基準となっているプリミティブセット
    const PrimitiveSetParam& master;

    // 最大関節数
    const uint32_t maxJointMtxNum;

    FuncFindIncludedJointIdx(const PrimitiveSetParam& prim, uint32_t num)
        : master(prim),
            maxJointMtxNum(num)
    {}

    ~FuncFindIncludedJointIdx() {}

    void operator()(const PrimitiveSetParam& rhs)
    {
        if (&master == &rhs) {
            // 同じものは無視する
            return;
        }

        // 所属関節群がどれだけ一致するか調べる

        uint32_t nMatchCnt = 0;

        std::set<uint32_t>::const_iterator it = master.joint.begin();
        for (; it != master.joint.end(); it++) {
            uint32_t jointIdx = *it; 

            std::set<uint32_t>::const_iterator itRhs = rhs.joint.begin();
            for (; itRhs != rhs.joint.end(); itRhs++) {
                uint32_t rhsJointIdx = *itRhs;

                if (jointIdx == rhsJointIdx) {
                    nMatchCnt++;
                }
            }
        }

        // 一致する関節があった
        if (nMatchCnt > 0) {
            // 一致する関節を持つプリミティブセットを登録する
            // しかし、そこで関節の最大数を超えてはまずいのでチェックする

            // 一致しない関節の数が新たに増える関節数になるので
            // 一致しない関節数を計算する
            uint32_t added = (uint32_t)rhs.joint.size() - nMatchCnt;

            if (added + master.joint.size() <= maxJointMtxNum) {
                // 新たに登録しても最大数は超えないので登録する
                candidateList[nMatchCnt - 1].push_back(&rhs);
            }
        }
    }
};

std::vector< std::vector<const PrimitiveSetParam*> > FuncFindIncludedJointIdx::candidateList;

// 三角形に影響を与える関節に応じて三角形を分類する
void GeometryChunkExporter::classifyTriangleByJoint(MeshInfo& sMesh)
{
    for (uint32_t nTriPos = sMesh.startTri; nTriPos < sMesh.endTri; nTriPos++) {
        TriangleParam& sTri = m_TriList[nTriPos];

        // 所属する関節群からキー値を計算する
        uint32_t nKey = sTri.computeKey();

        // 自分の所属する可能性のあるプリミティブセットを探す
        std::vector<PrimitiveSetParam>::iterator itSubset = std::find(
                                                        sMesh.subset.begin(),
                                                        sMesh.subset.end(),
                                                        nKey);
        if (itSubset == sMesh.subset.end()) {
            // 見つからなかった
            // 見つからなかったということは初出の関節群なので新規にプリミティブセットを作る
            sMesh.subset.push_back(PrimitiveSetParam());
            PrimitiveSetParam& sPrimSet = sMesh.subset.back();
            {
                // 関節群から計算されたキー値
                sPrimSet.key = nKey;

                // 三角形を登録
                sPrimSet.tri.push_back(nTriPos);

                // 所属関節インデックスを登録
                std::set<uint32_t>::const_iterator it = sTri.joint.begin();
                for (; it != sTri.joint.end(); it++) {
                    sPrimSet.joint.insert(*it);
                }
            }
        }
        else {
            // 見つかったので三角形を登録
            PrimitiveSetParam& sPrimSet = *itSubset;
            sPrimSet.tri.push_back(nTriPos);
        }
    }

#if 0
    // Merge triangles by joint idx.
    if (sMesh.subset.size() > 1) {
        std::vector<PrimitiveSetParam>::iterator it = sMesh.subset.begin();

        for (; it != sMesh.subset.end(); ) {
            PrimitiveSetParam& sPrimSet = *it;

            bool bIsAdvanceIter = true;

            // NOTE
            // PrimitiveSetParam::tri : PrimitiveSetParamを構成する三角形情報へのインデックス
            // PrimitiveSetParamを構成する三角形 -> 同じ関節群（スキン計算を行うための関節）に所属する三角形
            // そのため、関節情報についてはどれでも共通なので、sPrimSet.tri[0] を持ってくればいい

            uint32_t nTriIdx = sPrimSet.tri[0];
            const TriangleParam& sTri = m_TriList[nTriIdx];

            if (sTri.GetJointNum() < 4) {
                // sMeshGroup の持つJointIdxを全て内包するMeshGroupを探す
                std::vector<PrimitiveSetParam>::iterator itFind = std::find(
                                                                sMesh.subset.begin(),
                                                                sMesh.subset.end(),
                                                                sPrimSet);
                if (itFind != sMesh.subset.end()) {
                    PrimitiveSetParam& sSubsetMatch = *itFind;

                    // Merge to found MeshGroup.
                    sSubsetMatch.tri.insert(
                        sSubsetMatch.tri.end(),
                        sPrimSet.tri.begin(),
                        sPrimSet.tri.end());

                    // Register joint indices.
                    std::set<uint32_t>::const_iterator itJoint = sTri.joint.begin();
                    for (; itJoint != sTri.joint.end(); itJoint++) {
                        sSubsetMatch.joint.insert(*itJoint);
                    }

                    // Delete the MeshGroup because of merging to another MeshGroup.
                    it = sMesh.subset.erase(it);
                    bIsAdvanceIter = false;
                }
            }

            if (bIsAdvanceIter) {
                it++;
            }
        }
    }
#else
    // 影響を受ける関節が同じ三角形をまとめる
    // Merge triangles by joint idx.
    if (sMesh.subset.size() > 1) {
        // 候補リストを最大関節数だけ確保
        FuncFindIncludedJointIdx::candidateList.resize(m_MaxJointMtxNum);

        std::vector<PrimitiveSetParam>::iterator it = sMesh.subset.begin();

        for (; it != sMesh.subset.end(); ) {
            PrimitiveSetParam& sPrimSet = *it;
            uint32_t masterKey = sPrimSet.key;

            bool erased = false;

            if (sPrimSet.joint.size() < m_MaxJointMtxNum) {
                // マージ候補を探す
                std::for_each(
                    sMesh.subset.begin(),
                    sMesh.subset.end(),
                    FuncFindIncludedJointIdx(sPrimSet, m_MaxJointMtxNum));

                std::vector<uint32_t> releaseList;

                // 一致した関節数ごとに処理を行う

                for (int32_t i = m_MaxJointMtxNum - 1; i >= 0; i--) {
                    std::vector<const PrimitiveSetParam*>::iterator candidate = FuncFindIncludedJointIdx::candidateList[i].begin();

                    for (; candidate != FuncFindIncludedJointIdx::candidateList[i].end(); candidate++) {
                        const PrimitiveSetParam& prim = *(*candidate);

                        // マージする
                        PrimitiveSetParam& sSubsetMatch = const_cast<PrimitiveSetParam&>(prim);

                        if (sPrimSet.joint.size() + sSubsetMatch.joint.size() >= m_MaxJointMtxNum) {
                            // もう入らない可能性が高いので終了
                            break;
                        }

                        // ベースとなるプリミティブセットにマージ
                        // プリミティブセットを構成する三角形をベースとなるプリミティブセットに登録
                        sPrimSet.tri.insert(
                            sPrimSet.tri.end(),
                            sSubsetMatch.tri.begin(),
                            sSubsetMatch.tri.end());

                        // Register joint indices.
                        std::set<uint32_t>::const_iterator itJoint = sSubsetMatch.joint.begin();
                        for (; itJoint != sSubsetMatch.joint.end(); itJoint++) {
                            sPrimSet.joint.insert(*itJoint);
                        }

                        // 削除候補を登録
                        releaseList.push_back(prim.key);
                    }

                    FuncFindIncludedJointIdx::candidateList[i].clear();
                }

                // マージしたので消す
                for (size_t n = 0; n < releaseList.size(); n++) {
                    uint32_t key = releaseList[n];

                    std::vector<PrimitiveSetParam>::iterator itFind = std::find(
                                                                sMesh.subset.begin(),
                                                                sMesh.subset.end(),
                                                                key);
                    sMesh.subset.erase(itFind);
                    erased = true;
                }
            }

            if (erased) {
                it = std::find(
                        sMesh.subset.begin(),
                        sMesh.subset.end(),
                        masterKey);
                AT_ASSERT(it != sMesh.subset.end());
                it++;
            }
            else {
                it++;
            }
        }
    }
#endif
}

// メッシュ情報を取得
void GeometryChunkExporter::getMeshInfo(
    aten::FbxImporter* pImporter,
    MeshInfo& sMesh)
{
    sMesh.fmt = pImporter->getVtxFmt();
    sMesh.sizeVtx = pImporter->getVtxSize();

    // For skin.
    if (!m_SkinList.empty()) {
        sMesh.fmt |= (1 << (uint32_t)aten::MeshVertexFormat::BlendWeight);
        sMesh.fmt |= (1 << (uint32_t)aten::MeshVertexFormat::BlendIndices);

        sMesh.sizeVtx += (uint32_t)aten::MeshVertexSize::BlendIndices;
        sMesh.sizeVtx += (uint32_t)aten::MeshVertexSize::BlendWeight;
    }
}

bool GeometryChunkExporter::computeVtxNormal(
    aten::FbxImporter* pImporter,
    const TriangleParam& sTri)
{
    aten::vec4 vecPos[3];

    for (uint32_t i = 0; i < 3; i++) {
        uint32_t nVtxIdx = sTri.vtx[i];

        bool result = pImporter->getVertex(
			nVtxIdx,
            vecPos[i],
            aten::MeshVertexFormat::Position);
		AT_VRETURN_FALSE(result);
    }

    // NOTE
    // Counter Clock Wise
    //     1
    //   /   \
    //  /     \
    // 0 ----- 2

    for (uint32_t i = 0; i < 3; i++) {
        uint32_t nVtxIdx = sTri.vtx[i];

        uint32_t nIdx_0 = i;
        uint32_t nIdx_1 = (i + 1) % 3;
        uint32_t nIdx_2 = (i + 2) % 3;

		auto vP = vecPos[nIdx_1] - vecPos[nIdx_0];
		auto vQ = vecPos[nIdx_2] - vecPos[nIdx_0];

		auto nml = aten::cross(vQ, vP);

        VtxAdditional& sVtx = m_VtxList[nVtxIdx];
        sVtx.nml.push_back(nml);
    }

    return true;
}

bool GeometryChunkExporter::computeVtxTangent(
    aten::FbxImporter* pImporter,
    const TriangleParam& sTri)
{
    aten::vec4 vecPos[3];
    aten::vec4 vecUV[3];
    aten::vec4 vecNml[3];

    for (uint32_t i = 0; i < 3; i++) {
        uint32_t nVtxIdx = sTri.vtx[i];

        // Get position.
        bool result = pImporter->getVertex(
			nVtxIdx,
            vecPos[i],
            aten::MeshVertexFormat::Position);
		AT_VRETURN_FALSE(result);

        // Get texture coordinate.
        result = pImporter->getVertex(
			nVtxIdx,
            vecUV[i],
            aten::MeshVertexFormat::UV);
		AT_VRETURN_FALSE(result);

        // Get normal.
        result = pImporter->getVertex(
			nVtxIdx,
            vecNml[i],
            aten::MeshVertexFormat::Normal);
        if (!result) {
            // If mesh don't have normal, get normal from computed normal.
			AT_VRETURN_FALSE(m_VtxList.size() > nVtxIdx);

            const VtxAdditional& sVtx = m_VtxList[nVtxIdx];
            sVtx.getNormal((aten::vec3)vecNml[i]);
        }
    }

#if 1
    // NOTE
    // Counter Clock Wise
    //     0
    //   /   \
    //  /     \
    // 1 ----- 2

    float fCoeff[4];

    for (uint32_t i = 0; i < 3; i++) {
        uint32_t nVtxIdx = sTri.vtx[i];

        uint32_t nIdx_0 = i;
        uint32_t nIdx_1 = (i + 1) % 3;
        uint32_t nIdx_2 = (i + 2) % 3;

		auto vP = vecPos[nIdx_1] - vecPos[nIdx_0];
		auto vQ = vecPos[nIdx_2] - vecPos[nIdx_0];

        fCoeff[0] = vecUV[nIdx_2].v[1] - vecUV[nIdx_0].v[1];
        fCoeff[1] = -(vecUV[nIdx_1].v[1] - vecUV[nIdx_0].v[1]);
        fCoeff[2] = -(vecUV[nIdx_2].v[0] - vecUV[nIdx_0].v[0]);
        fCoeff[3] = vecUV[nIdx_1].v[0] - vecUV[nIdx_0].v[0];

        float fInvDeterminant = 1.0f / (fCoeff[3] * fCoeff[0] - fCoeff[2] * fCoeff[1]);

        // BiNormal
		vP = vP * fInvDeterminant * fCoeff[2];
		vQ = vQ * fInvDeterminant * fCoeff[3];
		auto vB = vP + vQ;
		vB.normalize();

        // Tangent
		// X(T) = Y(B) x Z(N)
		auto vT = cross(vB, vecNml[nIdx_0]);

        VtxAdditional& sVtx = m_VtxList[nVtxIdx];
        sVtx.tangent.push_back(vT);
    }
#endif

    return true;
}

void GeometryChunkExporter::computeVtxParemters(aten::FbxImporter* pImporter)
{
    // TODO
}

// 頂点データを出力
uint32_t GeometryChunkExporter::exportVertices(
    FileOutputStream* pOut,
    aten::FbxImporter* pImporter)
{
    IoStreamSeekHelper seekHelper(pOut);

    uint32_t nVBCnt = 0;
    uint32_t nPrevFmt = 0;

    for (size_t i = 0; i < m_MeshList.size(); i++) {
        MeshInfo& sMesh = m_MeshList[i];

        pImporter->beginMesh((uint32_t)i);

        for (size_t n = 0; n < sMesh.subset.size(); n++) {
            PrimitiveSetParam& sPrimSet = sMesh.subset[n];

            aten::MeshVertex sVtxInfo;

            // Blank MeshVertex. 
            AT_VRETURN(seekHelper.skip(sizeof(sVtxInfo)), 0);

			AT_VRETURN(
                exportVertices(
                    pOut,
                    pImporter,
                    sMesh,
                    sPrimSet), 0);

            // returnTo to position of expoting MeshVertex.
            AT_VRETURN_FALSE(seekHelper.returnWithAnchor());

			sVtxInfo.sizeVtx = sMesh.sizeVtx;
            sVtxInfo.numVtx = (uint16_t)m_ExportedVtx.size();

            // Export MeshVertex.
            OUTPUT_WRITE_VRETURN(pOut, &sVtxInfo, 0, sizeof(sVtxInfo));

            AT_VRETURN_FALSE(seekHelper.returnToAnchor());

            sPrimSet.idxVB = nVBCnt;

            nVBCnt++;
            m_ExportedVtx.clear();
        }

        pImporter->endMesh();
    }

    m_ExportedVtx.clear();

    return nVBCnt;
}

// 指定された関節リストにおける関節インデックスの格納位置を探す
static inline int32_t _FindJointIdx(
    const std::set<uint32_t>& tsJoint,
    uint32_t nJointIdx)
{
    std::set<uint32_t>::const_iterator it = tsJoint.begin();

    for (uint32_t pos = 0; it != tsJoint.end(); it++, pos++) {
        uint32_t idx = *it;

        if (idx == nJointIdx) {
            return pos;
        }
    }

    AT_ASSERT(false);
    return -1;
}

// 頂点データを出力.
bool GeometryChunkExporter::exportVertices(
    FileOutputStream* pOut,
    aten::FbxImporter* pImporter,
    const MeshInfo& sMesh,
    PrimitiveSetParam& sPrimSet)
{
    // 頂点データサイズのテーブル
    static uint32_t tblVtxSize[] = {
        (uint32_t)aten::MeshVertexSize::Position,
		(uint32_t)aten::MeshVertexSize::Normal,
		(uint32_t)aten::MeshVertexSize::Color,
		(uint32_t)aten::MeshVertexSize::UV,
		(uint32_t)aten::MeshVertexSize::Tangent,
    };

	m_vMin = aten::vec4(AT_MATH_INF);
	m_vMax = aten::vec4(-AT_MATH_INF);

    bool bEnableSkin = (m_SkinList.size() > 0);

    uint16_t nMinIdx = UINT16_MAX;
    uint16_t nMaxIdx = 0;

    for (size_t i = 0; i < sPrimSet.tri.size(); i++) {
        // 三角形を取得
        uint32_t nTriIdx = sPrimSet.tri[i];
        TriangleParam& sTri = m_TriList[nTriIdx];

        for (size_t nVtxPos = 0; nVtxPos < 3; nVtxPos++) {
            // 頂点インデックスを取得
            uint32_t nVtxIdx = sTri.vtx[nVtxPos];

			// For debug.
			uint32_t vtxSize = 0;

            // 出力済み頂点かどうか
            auto itFind = std::find(
                m_ExportedVtx.begin(),
                m_ExportedVtx.end(),
                nVtxIdx);

            // 頂点データ出力に応じたインデックスに変換
            if (itFind != m_ExportedVtx.end()) {
                // Exported...
                sTri.vtx[nVtxPos] = (uint32_t)std::distance(m_ExportedVtx.begin(), itFind);
            }
            else {
                sTri.vtx[nVtxPos] = (uint32_t)m_ExportedVtx.size();
            }

            uint32_t nIdx = sTri.vtx[nVtxPos];
            AT_ASSERT(nIdx <= UINT16_MAX);

            nMinIdx = (nIdx < nMinIdx ? nIdx : nMinIdx);
            nMaxIdx = (nIdx > nMaxIdx ? nIdx : nMaxIdx);

            if (itFind != m_ExportedVtx.end()) {
                // 出力済み頂点なのでこれ以上は何もしない
                continue;
            }

            // 出力済み頂点リストに登録
            m_ExportedVtx.push_back(nVtxIdx);

            for (uint32_t nVtxFmt = 0; nVtxFmt < (uint32_t)aten::MeshVertexFormat::Num; nVtxFmt++) {
                aten::vec4 vec;

                // 指定された頂点における指定フォーマットのデータを取得.
                bool bIsExist = pImporter->getVertex(
					nVtxIdx,
                    vec, 
                    (aten::MeshVertexFormat)nVtxFmt);

                // NOTE
                // bIsExist means whether specified format is exist.

                if (bIsExist) {
                    // 指定された頂点に指定フォーマットのデータは存在する
                    AT_ASSERT(nVtxFmt < AT_COUNTOF(tblVtxSize));
                    AT_ASSERT(sizeof(vec) >= tblVtxSize[nVtxFmt]);

                    if (nVtxFmt == (uint32_t)aten::MeshVertexFormat::Color) {
                        // カラーの場合は変換してから出力
                        uint8_t r = (uint8_t)(vec.x * 255.0f);
                        uint8_t g = (uint8_t)(vec.y * 255.0f);
                        uint8_t b = (uint8_t)(vec.z * 255.0f);
                        uint8_t a = (uint8_t)(vec.w * 255.0f);
                        uint32_t color = AT_COLOR_RGBA(r, g, b, a);
						AT_VRETURN_FALSE(pOut->write(&color, 0, tblVtxSize[nVtxFmt]));
                    }
                    else {
						// Color以外はそのまま出力.
						AT_VRETURN_FALSE(pOut->write(&vec, 0, tblVtxSize[nVtxFmt]));
                    }

					vtxSize += tblVtxSize[nVtxFmt];

                    if (nVtxFmt == (uint32_t)aten::MeshVertexFormat::Position) {
						// 最大、最小位置を保持.
                        m_vMin.x = std::min(m_vMin.x, vec.x);
                        m_vMin.y = std::min(m_vMin.y, vec.y);
                        m_vMin.z = std::min(m_vMin.z, vec.z);

                        m_vMax.x = std::max(m_vMax.x, vec.x);
                        m_vMax.y = std::max(m_vMax.y, vec.y);
                        m_vMax.z = std::max(m_vMax.z, vec.z);
                    }
                }
            }

            // For skin.
            if (bEnableSkin) {
                // 指定頂点におけるスキニング情報を取得
                uint32_t nSkinIdx = pImporter->getSkinIdxAffectToVtx(nVtxIdx);
                const SkinParam& sSkin = m_SkinList[nSkinIdx];

				aten::vec4 vecJoint(0);
				aten::vec4 vecWeight(0);
                
                for (uint32_t n = 0; n < (uint32_t)sSkin.joint.size(); n++) {
#if 1
                    // プリミティブセット内での関節位置を探す
                    // これが描画時における関節インデックスとなる
                    vecJoint.p[n] = (float)_FindJointIdx(sPrimSet.joint, sSkin.joint[n]);
#else
                    vecJoint.p[n] = sSkin.joint[n];
#endif
                    vecWeight.p[n] = sSkin.weight[n];
                }

                AT_VRETURN_FALSE(pOut->write(&vecJoint, 0, (uint32_t)aten::MeshVertexSize::BlendIndices));
                AT_VRETURN_FALSE(pOut->write(&vecWeight, 0, (uint32_t)aten::MeshVertexSize::BlendWeight));

				vtxSize += (uint32_t)aten::MeshVertexSize::BlendIndices;
				vtxSize += (uint32_t)aten::MeshVertexSize::BlendWeight;
            }
        }
    }

    sPrimSet.minIdx = nMinIdx;
    sPrimSet.maxIdx = nMaxIdx;

    return true;
}

bool GeometryChunkExporter::exportMesh(
    FileOutputStream* pOut,
    aten::FbxImporter* pImporter)
{
    for (size_t i = 0; i < m_MeshList.size(); i++) {
        const MeshInfo& sMesh = m_MeshList[i];

        aten::MeshSet sMeshInfo;
        {
            sMeshInfo.numSubset = (uint32_t)sMesh.subset.size();
            sMeshInfo.fmt = sMesh.fmt;
        }

        // Blank MeshSet. 
        IoStreamSeekHelper seekHelper(pOut);
		AT_VRETURN_FALSE(seekHelper.skip(sizeof(sMeshInfo)));

        m_Header.numMeshSubset += sMeshInfo.numSubset;

        pImporter->beginMesh((uint32_t)i);

		aten::vec4 vMin(AT_MATH_INF);
		aten::vec4 vMax(-AT_MATH_INF);

        for (size_t n = 0; n < sMesh.subset.size(); n++) {
            const PrimitiveSetParam& sPrimSet = sMesh.subset[n];

			AT_VRETURN_FALSE(
                exportPrimitiveSet(
                    pOut,
                    pImporter,
                    sPrimSet));

            // Get min and max position in primitives.
            getMinMaxPos(
                pImporter,
                vMin, vMax,
                sPrimSet);
        }

        pImporter->endMesh();

        // Compute center position in meshset.
        sMeshInfo.center[0] = (vMin.x + vMax.x) * 0.5f;
        sMeshInfo.center[1] = (vMin.y + vMax.y) * 0.5f;
        sMeshInfo.center[2] = (vMin.z + vMax.z) * 0.5f;

        // Get material information.
        pImporter->getMaterialForMesh(
            static_cast<uint32_t>(i),
            sMeshInfo.mtrl);

        // returnTo to position of expoting MeshSet.
		AT_VRETURN_FALSE(seekHelper.returnWithAnchor());

        // Export PrimitiveSet.
        OUTPUT_WRITE_VRETURN(pOut, &sMeshInfo, 0, sizeof(sMeshInfo));

		AT_VRETURN_FALSE(seekHelper.returnToAnchor());
    }

    return true;
}

void GeometryChunkExporter::getMinMaxPos(
    aten::FbxImporter* pImporter,
    aten::vec4& vMin,
	aten::vec4& vMax,
    const PrimitiveSetParam& sPrimSet)
{
    for (size_t i = 0; i < sPrimSet.tri.size(); i++) {
        uint32_t nTriIdx = sPrimSet.tri[i];
        const TriangleParam& sTri = m_TriList[nTriIdx];

        for (uint32_t n = 0; n < 3; ++n) {
            aten::vec4 vec;

            // Get vertex's position.
            bool bIsExist = pImporter->getVertex(
                sTri.vtx[n],
                vec, 
                aten::MeshVertexFormat::Position);
            AT_ASSERT(bIsExist);

            vMin.x = std::min(vMin.x, vec.x);
            vMin.y = std::min(vMin.y, vec.y);
            vMin.z = std::min(vMin.z, vec.z);

            vMax.x = std::max(vMax.x, vec.x);
            vMax.y = std::max(vMax.y, vec.y);
            vMax.z = std::max(vMax.z, vec.z);
        }
    }
}

bool GeometryChunkExporter::exportPrimitiveSet(
    FileOutputStream* pOut,
    aten::FbxImporter* pImporter,
    const PrimitiveSetParam& sPrimSet)
{
    aten::PrimitiveSet sSubsetInfo;
    {
        sSubsetInfo.idxVB = sPrimSet.idxVB;
        sSubsetInfo.minIdx = sPrimSet.minIdx;
        sSubsetInfo.maxIdx = sPrimSet.maxIdx;

        sSubsetInfo.numJoints = (uint16_t)sPrimSet.joint.size();
    }

    // のべ所属関節インデックス数
    m_Header.numAllJointIndices += sSubsetInfo.numJoints;

    // Blank PrimitiveSet. 
    IoStreamSeekHelper seekHelper(pOut);
	AT_VRETURN_FALSE(seekHelper.skip(sizeof(sSubsetInfo)));

    // 所属関節へのインデックス
    {
        std::set<uint32_t>::const_iterator it = sPrimSet.joint.begin();
        for (; it != sPrimSet.joint.end(); it++) {
            uint16_t idx = *it;
            OUTPUT_WRITE_VRETURN(pOut, &idx, 0, sizeof(idx));
        }
    }

    // Export indices.
    sSubsetInfo.numIdx = exportIndices(
                            pOut,
                            pImporter,
                            sPrimSet);
	AT_VRETURN_FALSE(sSubsetInfo.numIdx > 0);

    // returnTo to position of expoting PrimitiveSet.
	AT_VRETURN_FALSE(seekHelper.returnWithAnchor());

    // Export PrimitiveSet.
    OUTPUT_WRITE_VRETURN(pOut, &sSubsetInfo, 0, sizeof(sSubsetInfo));

	AT_VRETURN_FALSE(seekHelper.returnToAnchor());

    return true;
}

uint32_t GeometryChunkExporter::exportIndices(
    FileOutputStream* pOut,
    aten::FbxImporter* pImporter,
    const PrimitiveSetParam& sPrimSet)
{
    bool result = false;
    uint32_t nIdxNum = 0;

    std::vector<uint32_t> tvIndices;
    tvIndices.reserve(sPrimSet.tri.size() * 3);

    // Gather indices.
    for (size_t i = 0; i < sPrimSet.tri.size(); i++) {
        uint32_t nTriIdx = sPrimSet.tri[i];
        const TriangleParam& sTri = m_TriList[nTriIdx];

        tvIndices.push_back(sTri.vtx[0]);
        tvIndices.push_back(sTri.vtx[1]);
        tvIndices.push_back(sTri.vtx[2]);
    }

#if 1
	// TriangleList
	nIdxNum = static_cast<uint32_t>(tvIndices.size());

	result = OUTPUT_WRITE(pOut, &tvIndices[0], 0, sizeof(uint32_t) * tvIndices.size());
	AT_ASSERT(result);
#else
    if (m_ExportTriList)
    {
        // TriangleList

        nIdxNum = static_cast<uint32_t>(tvIndices.size());

        result = pOut->write(&tvIndices[0], 0, sizeof(uint32_t) * tvIndices.size());
        AT_ASSERT(result);
    }
    else
    {
        // TriangleStrip

        PrimitiveGroup* pPrimGroup = nullptr;
        uint16_t nPrimGroupNum = 0;

        // Crteate triangle strip.
        AT_VRETURN(
            GenerateStrips(
                &tvIndices[0],
                (uint32_t)tvIndices.size(),
                &pPrimGroup,
                &nPrimGroupNum), 0);

		AT_VRETURN(nPrimGroupNum == 1, 0);
		AT_VRETURN(pPrimGroup != nullptr, 0);

        nIdxNum = pPrimGroup->numIndices;

        // Export indices.
        result = pOut->write(pPrimGroup->indices, 0, sizeof(UINT32) * pPrimGroup->numIndices);
        AT_ASSERT(result);

        SAFE_DELETE_ARRAY(pPrimGroup);
    }
#endif

    return (result ? nIdxNum : 0);
}
