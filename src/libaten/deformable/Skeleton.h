#pragma once

#include "deformable/SKLFormat.h"
#include "deformable/ANMFormat.h"
#include "math/mat4.h"

#include <vector>

namespace aten
{
	class FileInputStream;

	class Skeleton {
		friend class deformable;

	private:
		Skeleton() {}
		~Skeleton() {}

	private:
		bool read(FileInputStream* stream);

	public:
		uint32_t getJointNum() const
		{
			return m_header.numJoint;
		}

		const std::vector<JointParam>& getJoints() const
		{
			return m_joints;
		}

	private:
		JointHeader m_header;

		std::vector<JointParam> m_joints;
	};

	///////////////////////////////////////////////////////

	class SkeletonController {
		friend class deformable;
		friend class DeformAnimation;

	private:
		SkeletonController() {}
		~SkeletonController() {}

	private:
		void init(Skeleton* skl);

	public:
		void buildPose(const mat4& mtxL2W);

		uint32_t getJointNum() const
		{
			AT_ASSERT(m_skl);
			return m_skl->getJointNum();
		}

		const mat4& getMatrix(uint32_t idx) const
		{
			return m_globalMatrix[idx];
		}

	private:
		void buildLocalMatrix(uint32_t idx);

		void beginUpdatePose(uint32_t idx);
		void endUpdatePose(uint32_t idx, uint8_t updateFlag);
		void updatePose(
			uint32_t idx,
			AnmTransformType transformType,
			AnmTransformType paramType,
			const vec4& param);

	private:
		Skeleton* m_skl{ nullptr };

		std::vector<JointPose> m_globalPose;
		std::vector<mat4> m_globalMatrix;
		std::vector<uint8_t> m_needUpdateJointFlag;

		bool m_isUpdatingPose{ false };
	};
}
