#pragma once 

#include <vector>
#include <functional>

#include "types.h"
#include "math/aabb.h"
#include "math/vec3.h"
#include "material/material.h"
#include "sampler/sampler.h"

//#define ENABLE_TANGENTCOORD_IN_HITREC

namespace aten {
	class hitable;
	class accelerator;

	struct hitrecord {
		vec3 p;

		vec3 normal;

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
		// tangent coordinate.
		vec3 du;
		vec3 dv;
#endif

		// texture coordinate.
		real u{ real(0) };
		real v{ real(0) };

		real area{ real(1) };

		int objid{ -1 };
		int mtrlid{ -1 };
	};

	struct Intersection {
		real t{ AT_MATH_INF };

		int objid{ -1 };

		short mtrlid{ -1 };

		// for cube.
		short face;

		int meshid{ -1 };

		int primid{ -1 };
		real a, b;	// barycentric

		real padding;
	};

	using NotifyChanged = std::function<void(hitable*)>;

	class hitable {
	public:
		hitable(const char* name = nullptr)
			: m_name(name)
		{}
		virtual ~hitable() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			Intersection& isect) const = 0;

		virtual const aabb& getBoundingbox() const
		{
			return m_aabb;
		}
		void setBoundingBox(const aabb& bbox)
		{
			m_aabb = bbox;
		}

		virtual aabb getTransformedBoundingBox() const
		{
			return std::move(m_aabb);
		}

		virtual const hitable* getHasObject() const
		{
			return nullptr;
		}

		bool isInstance() const
		{
			return (getHasObject() != nullptr);
		}

		virtual int geomid() const
		{
			return -1;
		}

		virtual accelerator* getInternalAccelerator();

		struct SamplePosNormalPdfResult {
			aten::vec3 pos;
			aten::vec3 nml;
			real area;

			real a;
			real b;
			int primid{ -1 };
		};

		virtual void getSamplePosNormalArea(SamplePosNormalPdfResult* result, sampler* sampler) const
		{
			AT_ASSERT(false);
		}

		static void evalHitResult(
			const hitable* obj,
			const ray& r,
			hitrecord& rec,
			const Intersection& isect)
		{
			obj->evalHitResult(r, rec, isect);

			rec.objid = isect.objid;
			rec.mtrlid = isect.mtrlid;

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
			// tangent coordinate.
			rec.du = normalize(getOrthoVector(rec.normal));
			rec.dv = normalize(cross(rec.normal, rec.du));
#endif
		}

		using FuncPreDraw = std::function<void(const aten::mat4&, int, int)>;

		virtual void draw(
			FuncPreDraw func,
			const aten::mat4& mtxL2W,
			int parentId = -1)
		{
			// For rasterize rendering.
			AT_ASSERT(false);
		}

		using FuncDrawAABB = std::function<void(const aten::mat4&)>;

		virtual void drawAABB(
			FuncDrawAABB func,
			const aten::mat4& mtxL2W)
		{
			// For debug rendering.
			AT_ASSERT(false);
		}

		void setFuncNotifyChanged(NotifyChanged onNotifyChanged)
		{
			m_onNotifyChanged = onNotifyChanged;
		}

		virtual void update()
		{
			// Nothing is done...
		}

	protected:
		void onNotifyChanged()
		{
			if (m_onNotifyChanged) {
				m_onNotifyChanged(this);
			}
		}

	private:
		virtual void evalHitResult(
			const ray& r,
			hitrecord& rec,
			const Intersection& isect) const
		{
			AT_ASSERT(false);
		}

	private:
		const char* m_name;
		aabb m_aabb;

		NotifyChanged m_onNotifyChanged;
	};
}
