#pragma once

#include <string>
#include <vector>
#include <map>

#include "ANMFormat.h"
#include "misc/stream.h"
#include "deformable/Skeleton.h"

namespace aten
{
    /**
     * @brief Animation for deformable mesh.
     */
    class DeformAnimation {
    public:
        DeformAnimation() {}
        virtual ~DeformAnimation() {}

    public:
        /**
         * @brief Read animation data from the specified file.
         */
        bool read(std::string_view path);

        /**
         * @brief 指定されたスケルトンにアニメーションを適用する.
         */
        void applyAnimation(
            SkeletonController* skl,
            float time);

        /**
         * @brief 指定されたスケルトンの指定されたジョイントにアニメーションを適用する.
         */
        void applyAnimationByIdx(
            SkeletonController* skl,
            uint32_t jointIdx,
            float time);

        const AnmHeader& getDesc() const
        {
            return m_header;
        }

    private:
        bool read(FileInputStream* stream);

        void applyAnimation(
            const AnmNode& node,
            SkeletonController* skl,
            uint32_t jointIdx,
            float time);

    private:
        AnmHeader m_header;

        std::vector<AnmNode> m_nodes;
        std::map<uint32_t, AnmNode*> m_nodeMap;

        std::vector<AnmChannel> m_channels;

        std::vector<AnmKey> m_keys;
        std::vector<std::vector<float>> m_keyParams;
    };
}
