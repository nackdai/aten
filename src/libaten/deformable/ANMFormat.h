#pragma once

#include "defs.h"
#include "types.h"

// ANM = ANiMation

namespace aten
{
    /** アニメーションの補間計算.
     */
    enum class AnmInterpType : uint8_t {
        Linear,
        Bezier,
        Hermite,
        Slerp,

        Num,

        Mask = 0x7f,
        UserCustom = 1 << 7,
    };
    AT_STATICASSERT(sizeof(AnmInterpType) == sizeof(uint8_t));

    /** アニメーションを行うパラメータ.
     */
    enum class AnmTransformType : uint32_t {
        Invalid = 0,

        ParamX  = 1 << 0,
        ParamY  = 1 << 1,
        ParamZ  = 1 << 2,
        ParamW  = 1 << 3,
        ParamXYZ  = ParamX | ParamY | ParamZ,
        ParamXYZW = ParamX | ParamY | ParamZ | ParamW,

        ParamShift = 0,
        ParamMask = 0x0f,

        Translate  = 1 << 4,
        Rotate     = 1 << 5,
        Quaternion = 1 << 6,
        Scale      = 1 << 7,

        TransformShift = 4,
        TransformMask = 0x7ffffff0,

        TranslateXYZ = Translate | ParamXYZ,
        TranslateX   = Translate | ParamX,
        TranslateY   = Translate | ParamY,
        TranslateZ   = Translate | ParamZ,

        RotateXYZ = Rotate | ParamXYZ,
        RotateX   = Rotate | ParamX,
        RotateY   = Rotate | ParamY,
        RotateZ   = Rotate | ParamZ,

        QuaternionXYZW = Quaternion | ParamXYZW,
        QuaternionXYZ  = Quaternion | ParamXYZ,
        QuaternionW    = Quaternion | ParamW,

        ScaleXYZ = Scale | ParamXYZ,
        ScaleX   = Scale | ParamX,
        ScaleY   = Scale | ParamY,
        ScaleZ   = Scale | ParamZ,
    };
    AT_STATICASSERT(sizeof(AnmTransformType) == sizeof(uint32_t));

    /** アニメーションキーフレーム.
     */
    enum AnmKeyType : uint32_t {
        Time,    ///< 時間ベース.
        Frame,       ///< フレームベース.
    };
    AT_STATICASSERT(sizeof(AnmKeyType) == sizeof(uint32_t));

    // NOTE
    // フォーマット
    // +----------------+
    // | ファイルヘッダ |
    // +----------------+
    // | アニメーション |
    // |     ノード     |
    // +----------------+
    // | アニメーション |
    // |   チャンネル   |
    // +----------------+
    // | アニメーション |
    // |      キー      |
    // +----------------+

    /**
    */
    struct AnmHeader {
        uint32_t magic{ 0 };
        uint32_t version{ 0 };

        uint32_t sizeHeader{ 0 };
        uint32_t sizeFile{ 0 };

        uint32_t numNodes{ 0 };
        uint32_t numChannels{ 0 };
        uint32_t numKeys{ 0 };

        AnmKeyType keyType{ AnmKeyType::Time };    ///< アニメーションキーフレーム.
        uint32_t reserved{ 0 };

        float time{ 0.0f };
    };

    /** キーフレーム情報.
     *
     * キーフレームあたりのジョイントのパラメータに適用するパラメータ.
     */
    struct AnmKey {
        float keyTime{ 0.0f };         ///< キー時間.

        uint8_t numParams{ 0 };     ///< アニメーションパラメータ数。位置、回転、スケールによって異なる.
        uint8_t stride{ 0 };        ///< １パラメータあたりのバイトサイズ.
        uint8_t reserved[2];

        // TODO
        // Work around...
        union {
            uint64_t value;
            float* params;
        };

        AnmKey()
        {
            value = 0;
        }
    };

    /** アニメーションチャンネル
     *
     * ジョイントのパラメータ（ex. 位置、回転など）ごとのアニメーション情報
     */
    struct AnmChannel {
        AnmInterpType interp{ AnmInterpType::Linear };    ///< 補間計算のタイプ.
        uint8_t stride{ 0 };
        uint16_t numKeys{ 0 };  ///< キーフレーム情報数.

        uint32_t keyIdx{ 0 };     ///< チャンネルが参照するキーの最初の位置.

        AnmTransformType type{ AnmTransformType::Invalid };       ///< アニメーションを行うパラメータのタイプ.
    };

    /** アニメーションノード.
     *
     * 適用ジョイントの情報
     */
    struct AnmNode {
        char target[32];       ///< 適用対象のジョイント名.
        uint32_t targetKey{ 0 };    ///< 適用対象のジョイントのキー値.

        uint16_t targetIdx{ 0 };    ///< 適用対象のジョイントのインデックス.
        uint16_t numChannels{ 0 };  ///< チャンネル数.

        uint32_t channelIdx{ 0 };   ///< ノードが参照するチャンネルの最初の位置.

        AnmNode()
        {
            target[0] = 0;
        }
    };
}
