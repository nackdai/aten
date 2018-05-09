#pragma once

#include <vector>
#include "scene/hitable.h"
#include "math/frustum.h"

namespace aten {
	enum class AccelType {
		Bvh,
		Qbvh,
		Sbvh,
		ThreadedBvh,
		StacklessBvh,
		StacklessQbvh,
		UserDefs,

		Default,
	};

	class accelerator : public hitable {
		friend class object;
		friend class deformable;
		template<typename ACCEL> friend class AcceleratedScene;

	private:
		accelerator() {}

	protected:
		accelerator(AccelType type)
		{
			m_type = type;
		}
		virtual ~accelerator() {}

	private:
		static AccelType s_internalType;
		static std::function<accelerator*()> s_userDefsInternalAccelCreator;

		static accelerator* createAccelerator(AccelType type = AccelType::Default);

		static void setInternalAccelType(AccelType type);
		static AccelType getInternalAccelType();

		void asNested()
		{
			m_isNested = true;
		}

	public:
		static void setUserDefsInternalAccelCreator(std::function<accelerator*()> creator);

		virtual void build(
			hitable** list,
			uint32_t num,
			aabb* bbox = nullptr) = 0;

		virtual void buildVoxel(
			uint32_t exid,
			uint32_t offset)
		{
			// Nothing is done.
			AT_ASSERT(false);
		}

		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect,
			bool enableLod) const = 0;

		virtual void update()
		{
			AT_ASSERT(false);
		}

		virtual void drawAABB(
			aten::hitable::FuncDrawAABB func,
			const aten::mat4& mtxL2W)
		{
			AT_ASSERT(false);
		}

		struct ResultIntersectTestByFrustum {
			int ep{ -1 };	///< Entry Point.
			int ex{ -1 };	///< Layer Id.

			// 1つ上のレイヤーへの戻り先のノードID.
			int top{ -1 };	///< Upper layer id.

			int padding;

			ResultIntersectTestByFrustum() {}
		};

		virtual bool hitMultiLevel(
			const ResultIntersectTestByFrustum& fisect,
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const
		{
			AT_ASSERT(false);
			return false;
		}

		virtual ResultIntersectTestByFrustum intersectTestByFrustum(const frustum& f)
		{
			AT_ASSERT(false);
			return std::move(ResultIntersectTestByFrustum());
		}

		virtual bool exportTree(const char* path)
		{
			AT_ASSERT(false);
			return false;
		}

		virtual bool importTree(const char* path, int offsetTriIdx)
		{
			AT_ASSERT(false);
			return false;
		}

		AccelType getAccelType()
		{
			return m_type;
		}

	protected:
		bool isExporting() const
		{
			return m_isExporting;
		}
		void enableExporting()
		{
			m_isExporting = true;
		}

	protected:
		AccelType m_type{ AccelType::Bvh };
		bool m_isNested{ false };
		bool m_isExporting{ false };
	};
}
