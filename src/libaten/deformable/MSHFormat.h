#pragma once

#include "types.h"

namespace aten
{
    enum class MeshVertexFormat : uint32_t {
        Position,
        Normal,
        Color,
        UV,
        Tangent,
        BlendIndices,
        BlendWeight,

        Num,
    };

    enum class MeshVertexSize : uint32_t {
        Position     = sizeof(float) * 4, ///< 頂点位置
        Normal       = sizeof(float) * 3, ///< 法線
        Color        = sizeof(uint8_t) * 4,     ///< 頂点カラー
        UV           = sizeof(float) * 2, ///< UV座標
        Tangent      = sizeof(float) * 3, ///< 接ベクトル
        BlendIndices = sizeof(float) * 4, ///< ブレンドマトリクスインデックス
        BlendWeight  = sizeof(float) * 4, ///< ブレンドウエイト
    };

    enum {
        MaxJointMtxNum = 4,
    };

    // フォーマット
    // +------------------------+
    // |         ヘッダ     　　|
    // +------------------------+
    // |    メッシュグループ    |
    // +------------------------+

    // メッシュグループ
    // +------------------------+
    // |     グループヘッダ     |
    // +------------------------+
    // |   頂点データテーブル   |
    // | +--------------------+ |
    // | |      ヘッダ        | |
    // | +--------------------+ |
    // | |     頂点データ     | |
    // | +--------------------+ |
    // |         ・・・         |
    // +------------------------+
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

    struct MeshHeader {
        uint32_t magic{ 0 };
        uint32_t version{ 0 };

        uint32_t sizeHeader{ 0 };
        uint32_t sizeFile{ 0 };

        float maxVtx[3];
        float minVtx[3];

        uint16_t numVB{ 0 };
        uint16_t numMeshGroup{ 0 };
        uint16_t numMeshSet{ 0 };
        uint16_t numMeshSubset{ 0 };

        uint32_t numAllJointIndices{ 0 }; ///< ジョイントインデックス総数

        struct {
            uint32_t isGPUSkinning : 1;
        };
    };

    /////////////////////////////////////////////////////////

    /**
     * @brief マテリアル情報
     */
    struct MeshMaterial {
        char name[32];  ///< マテリアル名
        uint32_t nameKey{ 0 };    ///< マテリアル名キー
    };

    /////////////////////////////////////////////////////////

    /**
     * @brief 頂点データ情報
     */
    struct MeshVertex {
        uint32_t sizeVtx{ 0 };  ///< １頂点あたりのサイズ
        uint32_t numVtx{ 0 };   ///< 頂点数
    };

    /**
     * @brief メッシュグループ情報
     */
    struct MeshGroup {
        uint16_t numVB{ 0 };        ///< 頂点バッファ数
        uint16_t numMeshSet{ 0 };   ///< メッシュセット数
    };

    /**
     * @brief メッシュセット情報
     */
    struct MeshSet {
        uint16_t numSubset{ 0 };

        uint16_t fmt{ 0 };      ///< 頂点フォーマット

        float center[3]; ///< 重心位置

        MeshMaterial mtrl;    ///< マテリアル情報
    };

    /**
     * @brief プリミティブセット情報
     */
    struct PrimitiveSet {
        uint16_t idxVB{ 0 };        ///< 利用する頂点バッファのインデックス
        uint16_t minIdx{ 0 };
        uint16_t maxIdx{ 0 };
        uint16_t numJoints{ 0 };    ///< ジョイント数

        uint32_t numIdx{ 0 };       ///< インデックス数
    };
}
