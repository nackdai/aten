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
        bool open(std::string_view pszName, bool isOpenForAnm = false);
        bool close();

        //////////////////////////////////
        // For geometry chunk.

        void exportGeometryCompleted();

        uint32_t getMeshNum();

        // ���b�V���Ɋւ��鏈�����J�n.
        void beginMesh(uint32_t nIdx);

        // ���b�V���Ɋւ��鏈�����I��.
        void endMesh();

        // BeginMesh�Ŏw�肳�ꂽ���b�V���Ɋ܂܂�X�L�j���O�����擾.
        void getSkinList(std::vector<SkinParam>& tvSkinList);

        // BeginMesh�Ŏw�肳�ꂽ���b�V���Ɋ܂܂��O�p�`���擾.
        uint32_t getTriangles(std::vector<TriangleParam>& tvTriList);

        // �w�肳�ꂽ���_�ɉe����^����X�L�j���O���ւ̃C���f�b�N�X���擾.
        uint32_t getSkinIdxAffectToVtx(uint32_t nVtxIdx);

        // �P���_������̃T�C�Y���擾.
        // �������A�X�L�j���O�Ɋւ���T�C�Y�͊܂܂Ȃ�
        uint32_t getVtxSize();

        // ���_�t�H�[�}�b�g���擾.
        // �������A�X�L�j���O�Ɋւ���t�H�[�}�b�g�͊܂܂Ȃ�
        uint32_t getVtxFmt();

        // �w�肳�ꂽ���_�ɂ�����w��t�H�[�}�b�g�̃f�[�^���擾.
        bool GetVertex(
            uint32_t nIdx,
            aten::vec4& vec,
            aten::MeshVertexFormat type);

        void getMaterialForMesh(
            uint32_t nMeshIdx,
            aten::MeshMaterial& sMtrl);

        //////////////////////////////////
        // For joint chunk.

        // �֐߃f�[�^�̏o�͊�����ʒm.
        void exportJointCompleted();

        // �֐߂Ɋւ��鏈�����J�n.
        bool beginJoint();

        // �֐߂Ɋւ��鏈�����I��.
        void endJoint();

        // �֐ߐ����擾.
        uint32_t getJointNum();

        // �w�肳�ꂽ�֐߂̖��O���擾.
        const char* getJointName(uint32_t nIdx);

        // �e�֐߂ւ̃C���f�b�N�X���擾.
        int32_t getJointParent(
            uint32_t nIdx,
            const std::vector<aten::JointParam>& tvJoint);

        // �w�肳�ꂽ�֐߂̋t�}�g���N�X���擾.
        void getJointInvMtx(
            uint32_t nIdx,
            aten::mat4& mtx);

        // �֐߂̎p�����擾.
        void getJointTransform(
            uint32_t nIdx,
            const std::vector<aten::JointParam>& tvJoint,
            std::vector<JointTransformParam>& tvTransform);

        //////////////////////////////////
        // For animation.

        // ���[�V�����̑ΏۂƂȂ郂�f���f�[�^���w��.
        bool readBaseModel(std::string_view pszName);

        // �t�@�C���Ɋ܂܂�郂�[�V�����̐����擾.
        uint32_t getAnmSetNum();

        // ���[�V�����Ɋւ��鏈�����J�n.
        bool beginAnm(uint32_t nSetIdx);

        // ���[�V�����Ɋւ��鏈�����I��.
        bool endAnm();

        // ���[�V�����m�[�h�i�K�p�W���C���g�j�̐����擾.
        uint32_t getAnmNodeNum();

        // �A�j���[�V�����`�����l���̐����擾.
        // �A�j���[�V�����`�����l���Ƃ�
        // �W���C���g�̃p�����[�^�iex. �ʒu�A��]�Ȃǁj���Ƃ̃A�j���[�V�������̂���
        uint32_t getAnmChannelNum(uint32_t nNodeIdx);

        // ���[�V�����m�[�h�i�K�p�W���C���g�j�̏����擾.
        bool getAnmNode(
            uint32_t nNodeIdx,
            aten::AnmNode& sNode);

        // �A�j���[�V�����`�����l���̏����擾.
        // �A�j���[�V�����`�����l���Ƃ�
        // �W���C���g�̃p�����[�^�iex. �ʒu�A��]�Ȃǁj���Ƃ̃A�j���[�V�������̂���
        bool getAnmChannel(
            uint32_t nNodeIdx,
            uint32_t nChannelIdx,
            aten::AnmChannel& sChannel);

        // �L�[�t���[�������擾.
        // �L�[�t���[��������̃W���C���g�̃p�����[�^�ɓK�p����p�����[�^���擾.
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

        bool GetMaterial(
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
