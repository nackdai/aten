#pragma once 

#include <vector>
#include "types.h"
#include "math/aabb.h"
#include "math/vec3.h"
#include "material/material.h"
#include "sampler/sampler.h"
#include "shape/shape.h"

//#define ENABLE_TANGENTCOORD_IN_HITREC

namespace aten {
	class hitable;

	struct hitrecord {
		real t{ AT_MATH_INF };

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

		union {
			// TODO
#ifdef __AT_CUDA__
			void* obj{ nullptr };
#else
			hitable* obj{ nullptr };
#endif
			int objid;
		};

		int mtrlid{ -1 };
	};

	struct Intersection {
		union {
			// cube.
			struct {
				int face;
			};
			// triangle.
			struct {
				int idx[3];
				real a, b;	// barycentric
			};
		};
	};

	class hitable {
	public:
		hitable(const char* name = nullptr);
		virtual ~hitable() {}

	public:
		virtual bool hit(
			const ray& r,
			real t_min, real t_max,
			hitrecord& rec,
			Intersection& isect) const = 0;

		virtual aabb getBoundingbox() const = 0;

		struct SamplePosNormalPdfResult {
			aten::vec3 pos;
			aten::vec3 nml;
			real area;

			real a;
			real b;
			int idx[3];
		};

		virtual void getSamplePosNormalArea(SamplePosNormalPdfResult* result, sampler* sampler) const
		{
			AT_ASSERT(false);
		}

		uint32_t id() const
		{
			return m_id;
		}

		static void evalHitResult(
			const hitable* obj,
			const ray& r,
			hitrecord& rec,
			const Intersection& isect)
		{
			obj->evalHitResult(r, rec, isect);

#ifdef ENABLE_TANGENTCOORD_IN_HITREC
			// tangent coordinate.
			rec.du = normalize(getOrthoVector(rec.normal));
			rec.dv = normalize(cross(rec.normal, rec.du));
#endif
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
		uint32_t m_id{ 0 };
		const char* m_name;
	};
}
