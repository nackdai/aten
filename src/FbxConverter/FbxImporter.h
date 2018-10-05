#pragma once

#include "GeometryCommon.h"

#include "deformable/ANMFormat.h"
#include "deformable/SKLFormat.h"
#include "deformable/MSHFormat.h"

#include <vector>
#include <map>

class FbxDataManager;

namespace aten
{
    class FbxImporter {
        friend class aten::FbxImporter;

    public:
        FbxImporter();
        ~FbxImporter() { close(); }

    public:
        bool open(const char* pszName, bool isOpenForAnm = false);
        bool close();

        //////////////////////////////////
        // For geometry chunk.

        void exportGeometryCompleted();

        uint32_t getMeshNum();

        // メッシュに関する処理を開始.
        void beginMesh(uint32_t nIdx);

        // メッシュに関する処理を終了.
        void endMesh();

        // BeginMeshで指定されたメッシュに含まれスキニング情報を取得.
        void getSkinList(std::vector<SkinParam>& tvSkinList);

        // BeginMeshで指定されたメッシュに含まれる三角形を取得.
        uint32_t getTriangles(std::vector<TriangleParam>& tvTriList);

        // 指定された頂点に影響を与えるスキニング情報へのインデックスを取得.
        uint32_t getSkinIdxAffectToVtx(uint32_t nVtxIdx);

        // １頂点あたりのサイズを取得.
        // ただし、スキニングに関するサイズは含まない
        uint32_t getVtxSize();

        // 頂点フォーマットを取得.
        // ただし、スキニングに関するフォーマットは含まない
        uint32_t getVtxFmt();

        // 指定された頂点における指定フォーマットのデータを取得.
        bool getVertex(
            uint32_t nIdx,
            aten::vec4& vec,
            aten::MeshVertexFormat type);

        void getMaterialForMesh(
            uint32_t nMeshIdx,
            aten::MeshMaterial& sMtrl);

        //////////////////////////////////
        // For joint chunk.

        // 関節データの出力完了を通知.
        void exportJointCompleted();

        // 関節に関する処理を開始.
        bool beginJoint();

        // 関節に関する処理を終了.
        void endJoint();

        // 関節数を取得.
        uint32_t getJointNum();

        // 指定された関節の名前を取得.
        const char* getJointName(uint32_t nIdx);

        // 親関節へのインデックスを取得.    
        int32_t getJointParent(
            uint32_t nIdx,
            const std::vector<aten::JointParam>& tvJoint);

        // 指定された関節の逆マトリクスを取得.  
        void getJointInvMtx(
            uint32_t nIdx,
            aten::mat4& mtx);

        // 関節の姿勢を取得.
        void getJointTransform(
            uint32_t nIdx,
            const std::vector<aten::JointParam>& tvJoint,
            std::vector<JointTransformParam>& tvTransform);

        //////////////////////////////////
        // For animation.

        // モーションの対象となるモデルデータを指定.
        bool readBaseModel(const char* pszName);

        // ファイルに含まれるモーションの数を取得.
        uint32_t getAnmSetNum();

        // モーションに関する処理を開始.
        bool beginAnm(uint32_t nSetIdx);

        // モーションに関する処理を終了.
        bool endAnm();

        // モーションノード（適用ジョイント）の数を取得.
        uint32_t getAnmNodeNum();

        // アニメーションチャンネルの数を取得.
        // アニメーションチャンネルとは
        // ジョイントのパラメータ（ex. 位置、回転など）ごとのアニメーション情報のこと
        uint32_t getAnmChannelNum(uint32_t nNodeIdx);

        // モーションノード（適用ジョイント）の情報を取得.
        bool getAnmNode(
            uint32_t nNodeIdx,
            aten::AnmNode& sNode);

        // アニメーションチャンネルの情報を取得.
        // アニメーションチャンネルとは
        // ジョイントのパラメータ（ex. 位置、回転など）ごとのアニメーション情報のこと
        bool getAnmChannel(
            uint32_t nNodeIdx,
            uint32_t nChannelIdx,
            aten::AnmChannel& sChannel);

        // キーフレーム情報を取得.
        // キーフレームあたりのジョイントのパラメータに適用するパラメータを取得.
        bool getAnmKey(
            uint32_t nNodeIdx,
            uint32_t nChannelIdx,
            uint32_t nKeyIdx,
            aten::AnmKey& sKey,
            std::vector<float>& tvValue);

        //////////////////////////////////
        // For material.

        bool beginMaterial();

        bool endMaterial();

        uint32_t getMaterialNum();

        bool getMaterial(
            uint32_t nMtrlIdx,
            MaterialInfo& mtrl);

        void setIgnoreTexIdx(int32_t idx)
        {
            m_ignoreTexIdx = idx;
        }

    private:
        bool getFbxMatrial(
            uint32_t nMtrlIdx,
            std::vector<MaterialTex>& mtrlTex,
            std::vector<MaterialParam>& mtrlParam);

        bool getFbxMatrialByImplmentation(
            uint32_t nMtrlIdx,
            std::vector<MaterialTex>& mtrlTex,
            std::vector<MaterialParam>& mtrlParam);

    private:
        FbxDataManager* m_dataMgr{ nullptr };
        FbxDataManager* m_dataMgrBase{ nullptr };

        uint32_t m_curMeshIdx{ 0 };
        uint32_t m_posVtx{ 0 };

        uint32_t m_curAnmIdx{ 0 };

        struct MaterialShading {
            void* fbxMtrl{ nullptr };
            std::string name;
        };
        std::map<uint32_t, std::vector<MaterialShading>> m_mtrlShd;

        void getLambertParams(void* mtrl, std::vector<MaterialParam>& list);
        void getPhongParams(void* mtrl, std::vector<MaterialParam>& list);

        int32_t m_ignoreTexIdx{ -1 };

        enum ParamType {
            Tranlate,
            Scale,
            Rotate,

            Num,
        };

        struct AnmKey {
            uint32_t key;
            float value[4];

            AnmKey() {}
        };

        struct AnmChannel {
            uint32_t nodeIdx;
            ParamType type[ParamType::Num];

            std::vector<AnmKey> keys[ParamType::Num];

            bool isChecked{ false };

            AnmChannel()
            {
                for (uint32_t i = 0; i < ParamType::Num; i++) {
                    type[i] = ParamType::Num;
                }
            }
        };
        std::vector<AnmChannel> m_channels;
    };
}