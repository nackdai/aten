#pragma once

#include "ANMFormat.h"

#include <vector>
#include <map>

namespace aten
{
	class FileInputStream;
	class SkeletonController;

    /**
    */
    class DeformAnimation {
    public:
		DeformAnimation() {}
        virtual ~DeformAnimation() {}

	public:
		bool read(const char* path);

		// 指定されたスケルトンにアニメーションを適用する.
		void applyAnimation(
			SkeletonController* skl,
			float time);

		// 指定されたスケルトンの指定されたジョイントにアニメーションを適用する.
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
