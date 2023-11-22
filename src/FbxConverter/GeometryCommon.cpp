#include "GeometryCommon.h"
#include "misc/key.h"

///////////////////////////////////////

// �O�p�`�ɉe����^����֐߃C���f�b�N�X�����ӂɌ��܂�L�[���v�Z����.
uint32_t TriangleParam::computeKey() const
{
    static uint32_t nJointTbl[4];

    uint32_t ret = 0;

    if (!joint.empty()) {
        size_t num = joint.size();

        std::set<uint32_t>::const_iterator it = joint.begin();
        for (size_t n = 0; n < num; n++, it++) {
            nJointTbl[n] = *it;
        }

        // Generate key value by joint indices.
        ret = aten::KeyValue::gen(
                nJointTbl,
                (uint32_t)num);
    }

    return ret;
}

// �O�p�`�ɉe����^����֐ߐ����擾.
uint32_t TriangleParam::getJointNum() const
{
    size_t ret = 0;

    if (!joint.empty()) {
        ret = joint.size();
    }

    return (uint32_t)ret;
}

// �w�肳�ꂽ�֐߂��폜.
void TriangleParam::eraseJoint(uint32_t idx)
{
    std::set<uint32_t>::iterator it = joint.begin();

    for (; it != joint.end(); it++) {
        if (*it == idx) {
            joint.erase(it);
            break;
        }
    }
}

///////////////////////////////////////

template <class _T>
static void eraseItem(
    std::vector<_T>& tvList,
    size_t pos)
{
    std::vector<_T>::iterator it = tvList.begin();
    std::advance(it, pos);
    tvList.erase(it);
}

// �֐߂�o�^.
void SkinParam::add(uint32_t nJointIdx, float fWeight)
{
    if (weight.size() < 4) {
        joint.push_back(nJointIdx);
        weight.push_back(fWeight);
    }
    else {
        // If num of skin is over 4, num of skin is limited to 4 by weight.
        size_t nMinIdx = 0;
        float fMinWeight = weight[0];

        for (size_t i = 1; i < weight.size(); i++) {
            float f = weight[i];
            if (fMinWeight > f) {
                fMinWeight = f;
                nMinIdx = i;
            }
        }

        if (fWeight > fMinWeight) {
            eraseItem(joint, nMinIdx);
            eraseItem(weight, nMinIdx);

            joint.push_back(nJointIdx);
            weight.push_back(fWeight);
        }
    }
}

// �E�G�C�g�l�̍��v���P�ɂȂ�悤�ɐ��K������.
void SkinParam::normalize()
{
    float fWeightSum = 0.0f;
    for (size_t i = 0; i < weight.size(); i++) {
        fWeightSum += weight[i];
    }

    for (size_t i = 0; i < weight.size(); i++) {
        weight[i] /= fWeightSum;
    }
}

// �w�肳�ꂽ�֐߂��폜����.
bool SkinParam::eraseJoint(uint32_t idx)
{
    for (size_t i = 0; i < joint.size(); i++) {
        if (joint[i] == idx) {
            std::vector<uint32_t>::iterator itJoint = joint.begin();
            std::advance(itJoint, i);
            joint.erase(itJoint);

            std::vector<float>::iterator itWeight = weight.begin();
            std::advance(itWeight, i);
            weight.erase(itWeight);

            return true;
        }
    }

    return false;
}

////////////////////////////////////////////

bool PrimitiveSetParam::operator==(const PrimitiveSetParam& rhs)
{
    return (this == &rhs);
}
