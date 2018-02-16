#pragma once

#include "types.h"
#include "defs.h"
#include "math/vec3.h"

#include <set>
#include <vector>

///////////////////////////////////////

enum JointTransform {
    Translate,
    Quaternion,
    Scale,

    AxisRot,
    Rotate,
    Skew,
};

///////////////////////////////////////

/** 三角形情報
 */
struct TriangleParam {
    uint32_t vtx[3];             ///< 三角形を構成する頂点インデックス
    std::set<uint32_t> joint;    ///< 三角形を構成する頂点に影響を与える（スキニングする）関節インデックス

    /** 三角形に影響を与える関節インデックスから一意に決まるキーを計算する.
     */
    uint32_t computeKey() const;

    /** 三角形に影響を与える関節数を取得.
     */ 
    uint32_t getJointNum() const;

    /** 指定された関節を削除.
     */
    void eraseJoint(uint32_t idx);
};

///////////////////////////////////////

/** スキニング情報.
 * １頂点ごとに存在する
 */
struct SkinParam {
    uint32_t vtxId;
    std::vector<uint32_t> joint;     ///< 影響を与える関節のインデックス
    std::vector<float> weight;       ///< ウエイト値

    /** 関節を登録.
     */
    void add(uint32_t nJointIdx, float fWeight);

    /** ウエイト値の合計が１になるように正規化する.
     */
    void normalize();

    /** 指定された関節を削除する.
     */
    bool eraseJoint(uint32_t idx);
};

///////////////////////////////////////

/** プリミティブセット.
 * 所属関節ごとにまとめられた三角形群
 */
struct PrimitiveSetParam {
    uint32_t key;                ///< 三角形に影響を与える関節インデックスから一意に決まるキー
    std::vector<uint32_t> tri;   ///< 関節から影響を受ける三角形群

    std::set<uint32_t> joint;    ///< 三角形に影響を与える関節インデックス

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

/** メッシュ情報.
 */
struct MeshInfo {
    uint32_t startTri;   ///< メッシュを構成する三角形の開始インデックス
    uint32_t endTri;     ///< メッシュを構成する三角形の終了インデックス

    std::vector<PrimitiveSetParam> subset;

    uint32_t fmt;        ///< メッシュにおける頂点フォーマット
    uint32_t sizeVtx;    ///< メッシュにおける１頂点あたりのサイズ
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
    const char* name;
    JointTransform type;
    std::vector<float> param;
};
