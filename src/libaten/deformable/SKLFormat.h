#pragma once

#include "defs.h"
#include "types.h"
#include "math/quaternion.h"

namespace aten
{
    enum class JointTransformType {
        Translate  = 1 << 0,
        Quaternion = 1 << 1,
        Scale      = 1 << 2,
    };

    // NOTE
    // フォーマット
    // +--------------------+
    // |   ファイルヘッダ   |
    // +--------------------+
    // | ジオメトリチャンク |
    // +--------------------+
    // | ジョイントチャンク |
    // +--------------------+
    // | マテリアルチャンク |
    // +--------------------+
    
    struct JointHeader {
		uint32_t magic{ 0 };
        uint32_t version{ 0 };

        uint32_t sizeHeader{ 0 };
        uint32_t sizeFile{ 0 };

        uint32_t numJoint{ 0 };
    };

    /////////////////////////////////////////////////////////
    // ジョイントチャンク

    // ジョイントチャンク
    // +----------------------+
    // |    チャンクヘッダ    |
    // +----------------------+
    // |  ジョイントチャンク  |
    // |       ヘッダ         |
    // +----------------------+
    // |  ジョイントテーブル  |
    // | +------------------+ |
    // | |     ジョイント   | |
    // | |+----------------+| |
    // | ||     ヘッダ     || |
    // | |+----------------+| |
    // | |+----------------+| |
    // | ||    変形データ  || |
    // | |+----------------+| |
    // | |       ・・・     | |
    // | +------------------+ |
    // |        ・・・        |
    // +----------------------+

    struct JointPose {
        float trans[3];
        quaternion<float> quat;
		float scale[3];
    };

    struct JointParam {
        char name[32];
        uint32_t nameKey{ 0 };

        // If parent is -1, joint has no parent.
        int16_t parent{ 0 };        ///< 親ジョイントのインデックス.
        uint16_t idx{ 0 };          ///< 自分自身のインデックス

        mat4 mtxInvBind;

        uint8_t validParam{ 0 };    ///< 有効なパラメータフラグ。ポーズパラメータの全てが有効とは限らない.
		uint8_t validAnmParam{ 0 }; ///< アニメ計算時の有効なパラメータフラグ。ポーズパラメータの全てが有効とは限らない.
		uint8_t reserved[2];

		JointPose pose;  ///< ポーズパラメータ
    };

	// TODO
	C_ASSERT(sizeof(mat4) == sizeof(float) * 16);
}
