#pragma once

#include "types.h"
#include "defs.h"
#include "math/vec3.h"

#include <set>
#include <vector>
#include <string>

///////////////////////////////////////

enum class JointTransform {
    Translate,
    Quaternion,
    Scale,

    AxisRot,
    Rotate,
    Skew,
};

///////////////////////////////////////

/** �O�p�`���
 */
struct TriangleParam {
    uint32_t vtx[3];             ///< �O�p�`���\�����钸�_�C���f�b�N�X
    std::set<uint32_t> joint;    ///< �O�p�`���\�����钸�_�ɉe����^����i�X�L�j���O����j�֐߃C���f�b�N�X

    /** �O�p�`�ɉe����^����֐߃C���f�b�N�X�����ӂɌ��܂�L�[���v�Z����.
     */
    uint32_t computeKey() const;

    /** �O�p�`�ɉe����^����֐ߐ����擾.
     */
    uint32_t getJointNum() const;

    /** �w�肳�ꂽ�֐߂��폜.
     */
    void eraseJoint(uint32_t idx);
};

///////////////////////////////////////

/** �X�L�j���O���.
 * �P���_���Ƃɑ��݂���
 */
struct SkinParam {
    uint32_t vtxId;
    std::vector<uint32_t> joint;     ///< �e����^����֐߂̃C���f�b�N�X
    std::vector<float> weight;       ///< �E�G�C�g�l

    /** �֐߂�o�^.
     */
    void add(uint32_t nJointIdx, float fWeight);

    /** �E�G�C�g�l�̍��v���P�ɂȂ�悤�ɐ��K������.
     */
    void normalize();

    /** �w�肳�ꂽ�֐߂��폜����.
     */
    bool eraseJoint(uint32_t idx);
};

///////////////////////////////////////

/** �v���~�e�B�u�Z�b�g.
 * �����֐߂��Ƃɂ܂Ƃ߂�ꂽ�O�p�`�Q
 */
struct PrimitiveSetParam {
    uint32_t key;                ///< �O�p�`�ɉe����^����֐߃C���f�b�N�X�����ӂɌ��܂�L�[
    std::vector<uint32_t> tri;   ///< �֐߂���e�����󂯂�O�p�`�Q

    std::set<uint32_t> joint;    ///< �O�p�`�ɉe����^����֐߃C���f�b�N�X

    uint32_t idxVB;
    uint16_t minIdx;
    uint16_t maxIdx;

    // For std::find
    bool operator==(uint32_t rhs)
    {
        return (key == rhs);
    }

    bool operator==(const PrimitiveSetParam& rhs);

#if 0
private:
    static std::vector<TriangleParam>* ptrTriList;

public:
    static void SetTriList(std::vector<TriangleParam>* pTriList) { ptrTriList = pTriList; }
    static std::vector<TriangleParam>* GetTriList()
    {
        AT_ASSERT(ptrTriList != nullptr);
        return ptrTriList;
    }
#endif
};

///////////////////////////////////////

/** ���b�V�����.
 */
struct MeshInfo {
    uint32_t startTri;   ///< ���b�V�����\������O�p�`�̊J�n�C���f�b�N�X
    uint32_t endTri;     ///< ���b�V�����\������O�p�`�̏I���C���f�b�N�X

    std::vector<PrimitiveSetParam> subset;

    uint32_t fmt;        ///< ���b�V���ɂ����钸�_�t�H�[�}�b�g
    uint32_t sizeVtx;    ///< ���b�V���ɂ�����P���_������̃T�C�Y
};

///////////////////////////////////////

struct VtxAdditional {
    std::vector<aten::vec3> nml;
    std::vector<aten::vec3> tangent;

    bool hasNormal() const
    {
        return !nml.empty();
    }

    bool hasTangent() const
    {
        return !tangent.empty();
    }

    void fixNormal()
    {
        if (nml.empty()) {
            return;
        }

        aten::vec3 v;

        for (size_t i = 0; i < nml.size(); i++) {
            v.x += nml[i].x;
            v.y += nml[i].y;
            v.z += nml[i].z;
        }

        float div = 1.0f / nml.size();
        v.x *= div;
        v.y *= div;
        v.z *= div;

        nml.clear();
        nml.push_back(v);
    }

    void getNormal(aten::vec3& v) const
    {
        // NOTE
        // Need to call "fixNormal" before call this function...
        AT_ASSERT(nml.size() == 1);
        v = nml[0];
    }

    void fixTangent()
    {
        if (tangent.empty()) {
            return;
        }

        aten::vec3 v;

        for (size_t i = 0; i < tangent.size(); i++) {
            v.x += tangent[i].x;
            v.y += tangent[i].y;
            v.z += tangent[i].z;
        }

        float div = 1.0f / tangent.size();
        v.x *= div;
        v.y *= div;
        v.z *= div;
    }

    void getTangent(aten::vec3& v) const
    {
        // NOTE
        // Need to call "fixTangent" before call this function...
        AT_ASSERT(tangent.size() == 1);
        v = tangent[0];
    }
};

///////////////////////////////////////

struct JointTransformParam {
    std::string name;
    JointTransform type;
    std::vector<float> param;
};

///////////////////////////////////////

struct TextureType {
    bool isSpecular{ false };
    bool isNormal{ false };
    bool is_translucent{ false };
};

struct MaterialTex {
    std::string name;
    TextureType type;
};

struct MaterialParam {
    std::string name;
    std::vector<float> values;
};

struct MaterialInfo {
    std::string name;
    std::vector<MaterialTex> tex;
    std::vector<MaterialParam> params;
};
