#include "deformable/DeformAnimation.h"
#include "deformable/DeformAnimationInterp.h"

namespace aten
{
    bool DeformAnimation::read(std::string_view path)
    {
        FileInputStream file;
        AT_VRETURN_FALSE(file.open(path));

        return read(&file);
    }

    bool DeformAnimation::read(FileInputStream* stream)
    {
        AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_header, sizeof(m_header)));

        // ノード情報読み込み.
        {
            m_nodes.resize(m_header.numNodes);
            AT_VRETURN_FALSE(
                AT_STREAM_READ(
                    stream, &m_nodes[0],
                    static_cast<uint32_t>(sizeof(AnmNode) * m_nodes.size())));

            // ノードを探索しやすくするためにハッシュに登録.
            for (uint32_t i = 0; i < m_header.numNodes; i++) {
                auto& node = m_nodes[i];
                m_nodeMap.insert(std::pair<uint32_t, AnmNode*>(node.targetKey, &node));
            }
        }

        // チャンネル情報読み込み.
        {
            m_channels.resize(m_header.numChannels);
            AT_VRETURN_FALSE(
                AT_STREAM_READ(
                    stream, &m_channels[0],
                    static_cast<uint32_t>(sizeof(AnmChannel) * m_channels.size())));
        }

        m_keys.resize(m_header.numKeys);
        m_keyParams.resize(m_header.numKeys);

        for (uint32_t nodeIdx = 0; nodeIdx < m_header.numNodes; ++nodeIdx) {
            const auto& node = m_nodes[nodeIdx];

            for (uint32_t channelIdx = 0; channelIdx < node.numChannels; ++channelIdx) {
                const auto& channel = m_channels[node.channelIdx + channelIdx];

                for (uint32_t keyIdx = 0; keyIdx < channel.numKeys; ++keyIdx) {
                    // キー情報読み込み.
                    AT_VRETURN_FALSE(AT_STREAM_READ(stream, &m_keys[keyIdx + channel.keyIdx], sizeof(AnmKey)));

                    auto& key = m_keys[keyIdx + channel.keyIdx];
                    auto& keyParam = m_keyParams[keyIdx + channel.keyIdx];

                    // キー情報のパラメータ読み込み.
                    keyParam.resize(key.numParams);
                    AT_VRETURN_FALSE(
                        AT_STREAM_READ(
                            stream, &keyParam[0],
                            static_cast<uint32_t>(sizeof(float) * keyParam.size())));

                    // パラメータへのポインタへ実データを割り当てる.
                    key.params = &keyParam[0];
                }
            }
        }

        return true;
    }

    // 指定されたスケルトンにアニメーションを適用する.
    void DeformAnimation::applyAnimation(
        SkeletonController* skl,
        float time)
    {
        auto jointNum = skl->getJointNum();

        for (uint32_t i = 0; i < jointNum; ++i) {
            applyAnimationByIdx(skl, i, time);
        }
    }

    // 指定されたスケルトンの指定されたジョイントにアニメーションを適用する.
    void DeformAnimation::applyAnimationByIdx(
        SkeletonController* skl,
        uint32_t jointIdx,
        float time)
    {
        AnmNode* targetNode = nullptr;

        for (auto& node : m_nodes) {
            if (node.targetIdx == jointIdx) {
                targetNode = &node;
                break;
            }
        }

        if (targetNode) {
            applyAnimation(
                *targetNode,
                skl,
                jointIdx,
                time);
        }
    }

    void DeformAnimation::applyAnimation(
        const AnmNode& node,
        SkeletonController* skl,
        uint32_t jointIdx,
        float time)
    {
        uint32_t updateFlag = 0;
        vec4 param(float(0));

        // 姿勢情報更新開始.
        skl->beginUpdatePose(jointIdx);

        for (uint32_t channelIdx = 0; channelIdx < node.numChannels; ++channelIdx)
        {
            const auto& channel = m_channels[channelIdx + node.channelIdx];

            auto paramType = (AnmTransformType)((uint32_t)channel.type & (uint32_t)AnmTransformType::ParamMask);
            auto transformType = (AnmTransformType)((uint32_t)channel.type & (uint32_t)AnmTransformType::TransformMask);

            // どのパラメータが更新されるのかというフラグ.
            switch (transformType) {
            case AnmTransformType::Translate:
                updateFlag |= (uint32_t)JointTransformType::Translate;
                break;
            case AnmTransformType::Quaternion:
                updateFlag |= (uint32_t)JointTransformType::Quaternion;
                break;
            case AnmTransformType::Scale:
                updateFlag |= (uint32_t)JointTransformType::Scale;
                break;
            default:
                AT_ASSERT(false);
                break;
            }

            const auto interp = channel.interp;
            const auto keyNum = channel.numKeys;
            const auto* keys = (const AnmKey*)&m_keys[channel.keyIdx];

            // 補間計算したパラメータ値を取得.
            if (DeformAnimationInterp::isScalarInterp(interp)) {
                switch (paramType) {
                case AnmTransformType::ParamX:    // Xのみ.
                    param.v[0] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 0, keys);
                    break;
                case AnmTransformType::ParamY:    // Yのみ.
                    param.v[1] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 0, keys);
                    break;
                case AnmTransformType::ParamZ:    // Zのみ.
                    param.v[2] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 0, keys);
                    break;
                case AnmTransformType::ParamW:    // Wのみ.
                    param.v[3] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 0, keys);
                    break;
                case AnmTransformType::ParamXYZ:  // XWZのみ.
                    param.v[0] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 0, keys);
                    param.v[1] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 1, keys);
                    param.v[2] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 2, keys);
                    break;
                case AnmTransformType::ParamXYZW: // XYZWすべて.
                    param.v[0] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 0, keys);
                    param.v[1] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 1, keys);
                    param.v[2] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 2, keys);
                    param.v[3] = DeformAnimationInterp::computeInterp(interp, time, keyNum, 3, keys);
                    break;
                default:
                    AT_ASSERT(false);
                    break;
                }
            }
            else {
                // NOTE
                // 現状slerpを行う場合

                // TODO
                AT_ASSERT(paramType == AnmTransformType::ParamXYZW);
                AT_ASSERT(transformType == AnmTransformType::Quaternion);

                DeformAnimationInterp::computeInterp(
                    param,
                    interp,
                    time,
                    keyNum,
                    0,
                    keys);
            }

            // 計算した姿勢情報をスケルトンに渡す.
            skl->updatePose(
                jointIdx,
                transformType,
                paramType,
                param);
        }

        // 姿勢情報更新終了.
        skl->endUpdatePose(
            jointIdx,
            updateFlag);
    }
}
