#pragma once

#include "deformable/SKLFormat.h"
#include "math/mat4.h"

#include <vector>

namespace aten
{
	class FileInputStream;

	class Skeleton {
	private:
		Skeleton() {}
		~Skeleton() {}

	private:
		bool read(FileInputStream* stream);

	public:
		void buildPose(const mat4& mtxL2W);

	private:
		void buildLocalPose(uint32_t idx);

	private:
		JointHeader m_header;

		std::vector<JointParam> m_joints;
		std::vector<mat4> m_globalPose;
		std::vector<uint8_t> m_needUpdateJointFlag;
	};
}
